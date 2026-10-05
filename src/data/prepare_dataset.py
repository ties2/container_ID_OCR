"""Turn a CVAT export into YOLO datasets for per-character segmentation and detection.

Input (from CVAT "CVAT for images 1.1" exports with images, one sub-folder per
CVAT task, e.g. cvat_export/AH-G1/, cvat_export/AH-G2/; all are merged):
    annotations.xml  - polygons labelled `container_id` (one per code)
                       and boxes labelled `char` (one per character)
    images/          - the annotated images
    labels.csv       - columns: filename, container_id  (the true 11-char code)

What it does, per image:
    1. Assign every `char` box to the `container_id` polygon that contains its centre.
    2. Check each code: exactly 11 boxes and a valid ISO 6346 check digit, else REVIEW.
    3. Order the boxes along the code's own direction (works for vertical,
       horizontal and tilted codes) and give box i the class of character i.
    4. Extract a pixel mask inside each (padded) box with Otsu thresholding.
    5. Write YOLO-seg labels (mask polygons) and YOLO-detect labels (the
       bounding box of the same mask), so both models see identical annotations.
    6. Split train/val/test by container ID (no container in two sets), stratified
       by variation group G1..G8 from the selection sheets in data/metadata/, and
       persistent (old containers never change split when new data is added).
    7. Save overlay images and a report so every mask can be checked by eye.

Usage (from the project root):
    python -m src.data.prepare_dataset            # uses the default project paths
    make prepare                                  # same thing

Default paths:
    data/metadata/Label-camera-*.csv  selection sheets: file,group,layout,angle,condition,status,notes
                                      (status 'rejected' = removed from the dataset, reason in notes)
    data/02_interim/cvat_export/<task>/   one unzipped CVAT export per task (annotations.xml + images/)
    data/02_interim/labels.csv     filename,container_id
    data/03_processed/char_dataset/  output (yolo_seg/, yolo_det/, review/, report.csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.models.postprocess import validate_iso6346

log = logging.getLogger("prepare")

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
CODE_LEN = 11



# --------------------------------------------------------------------------- data

@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


@dataclass
class Code:
    polygon: np.ndarray                     # (N, 2) float32
    boxes: list[Box] = field(default_factory=list)


@dataclass
class Instance:
    cls: int
    contour: np.ndarray | None              # (N, 2) int, None if extraction failed
    flags: list[str] = field(default_factory=list)


# ------------------------------------------------------------------- small helpers

def iso6346_valid(code: str) -> bool:
    """True if `code` is a complete 11-character ISO 6346 ID with a correct check digit.

    Reuses the validator of the OCR pipeline (src/models/postprocess.py).
    """
    return len(code) == CODE_LEN and validate_iso6346(code).is_valid_checksum


def image_key(name: str) -> str:
    """Normalise a filename so CVAT names and CSV names match.

    '1-122830001-OCR-AH-A01_jpg.rf.5957....jpg' -> '1-122830001-OCR-AH-A01'
    """
    stem = Path(name).name
    stem = re.sub(r"\.rf\.[0-9A-Za-z]+", "", stem)
    stem = re.sub(r"\.(jpe?g|png)$", "", stem, flags=re.I)
    return re.sub(r"_jpe?g$", "", stem, flags=re.I)


def camera_of(name: str) -> str:
    """Camera code from the filename ('1-122830001-OCR-AH-A01' -> 'AH')."""
    part = image_key(name).split("-OCR-")
    return part[1][:2] if len(part) > 1 else "?"


META_FIELDS = ("group", "layout", "angle", "condition")


def load_metadata(paths: list[Path]) -> dict[str, dict[str, str]]:
    """Read the per-camera selection sheets (file,group,layout,angle,condition,...).

    Returns image key -> {camera, group, layout, angle, condition, status, notes}.
    Images that are not in any sheet get '?' (status 'ok') so they are still processed.
    status 'rejected' removes an image from the dataset (see main); notes give the reason.
    """
    meta: dict[str, dict[str, str]] = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = image_key(row["file"])
                meta[key] = {"camera": camera_of(key), **{f: (row.get(f) or "?").strip() for f in META_FIELDS},
                             "status": (row.get("status") or "ok").strip().lower(),
                             "notes": (row.get("notes") or "").strip()}
    return meta


def meta_for(meta: dict[str, dict[str, str]], key: str) -> dict[str, str]:
    return meta.get(key, {"camera": camera_of(key), **{f: "?" for f in META_FIELDS},
                          "status": "ok", "notes": ""})


def group_id(name: str) -> str:
    """Capture-event id from the filename ('1-122830001-OCR-...' -> '122830001')."""
    parts = image_key(name).split("-")
    return parts[1] if len(parts) > 1 else image_key(name)


# --------------------------------------------------------------------- annotation

def _points(poly) -> np.ndarray:
    return np.array([[float(v) for v in p.split(",")] for p in poly.get("points").split(";")], dtype=np.float32)


def _inside_any(point: tuple[float, float], polygons: list[np.ndarray]) -> bool:
    return any(cv2.pointPolygonTest(pg, point, False) >= 0 for pg in polygons)


def parse_ignore(xml_path: Path) -> dict[str, list[np.ndarray]]:
    """Polygons labelled `ignore` (e.g. a code cut off by the image border), per image."""
    return {img.get("name"): [_points(pg) for pg in img.iter("polygon") if pg.get("label") == "ignore"]
            for img in ET.parse(xml_path).getroot().iter("image")}


def parse_cvat(xml_path: Path) -> dict[str, tuple[int, int, list[Code], list[Box]]]:
    """Read a CVAT-for-images-1.1 file. Returns name -> (w, h, codes, unassigned boxes).

    Codes and character boxes whose centre lies inside an `ignore` polygon are dropped:
    such regions are painted out of the training images (see paint_out).
    """
    out = {}
    for img in ET.parse(xml_path).getroot().iter("image"):
        w, h = int(img.get("width")), int(img.get("height"))
        ignore = [_points(pg) for pg in img.iter("polygon") if pg.get("label") == "ignore"]
        codes = [Code(_points(pg)) for pg in img.iter("polygon") if pg.get("label") == "container_id"]
        codes = [c for c in codes if not _inside_any(tuple(map(float, c.polygon.mean(axis=0))), ignore)]
        boxes = [
            Box(*(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")))
            for b in img.iter("box") if b.get("label") == "char"
        ]
        boxes = [b for b in boxes if not _inside_any(b.centre, ignore)]
        loose = []
        for b in boxes:
            home = next((c for c in codes
                         if cv2.pointPolygonTest(c.polygon, b.centre, False) >= 0), None)
            (home.boxes if home else loose).append(b)
        out[img.get("name")] = (w, h, codes, loose)
    return out


def annotation_files(export: Path) -> list[Path]:
    """All annotations.xml files below the export folder (one per CVAT task export)."""
    files = sorted(export.rglob("annotations.xml"))
    if not files:
        raise SystemExit(f"No annotations.xml found below {export}")
    return files


def load_annotations(export: Path) -> dict[str, tuple[int, int, list[Code], list[Box]]]:
    """parse_cvat for every export below `export`, merged. An image annotated in two
    exports keeps the later one (sorted by folder name) and a warning is logged."""
    merged: dict[str, tuple] = {}
    seen: dict[str, Path] = {}
    for f in annotation_files(export):
        for name, value in parse_cvat(f).items():
            key = image_key(name)
            if key in seen:
                log.warning("%s is annotated in %s and %s; using the second", key, seen[key], f)
                merged = {n: v for n, v in merged.items() if image_key(n) != key}
            seen[key] = f
            merged[name] = value
    return merged


def load_ignore(export: Path) -> dict[str, list[np.ndarray]]:
    """parse_ignore for every export below `export`, merged by image key."""
    out: dict[str, list[np.ndarray]] = {}
    for f in annotation_files(export):
        for name, polys in parse_ignore(f).items():
            out[image_key(name)] = polys
    return out


def reading_order(boxes: list[Box]) -> list[Box]:
    """Sort boxes along the main direction of the code.

    The principal axis of the box centres gives the code's direction. A mostly
    vertical axis is read top-to-bottom, a mostly horizontal one left-to-right.
    """
    pts = np.array([b.centre for b in boxes], dtype=np.float64)
    centred = pts - pts.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]     # first principal direction
    if abs(axis[1]) > abs(axis[0]):                               # vertical code
        axis = axis if axis[1] > 0 else -axis
    else:                                                          # horizontal code
        axis = axis if axis[0] > 0 else -axis
    order = np.argsort(centred @ axis)
    return [boxes[i] for i in order]


# -------------------------------------------------------------- mask extraction

def _background_is_bright(gray: np.ndarray, box: Box, thr: float, pad_frac: float = 0.35) -> bool:
    """Decide the text polarity from the container surface around the box.

    A ring around the box (outside it) is mostly container surface. If most of
    that ring is brighter than the Otsu threshold, the background is bright and
    the characters are dark. Looking outside the box keeps this correct even
    when the box is drawn tightly on a printed frame.
    """
    H, W = gray.shape
    bw_, bh_ = box.x2 - box.x1, box.y2 - box.y1
    px, py = pad_frac * bw_ + 4, pad_frac * bh_ + 4
    x1, y1 = max(0, int(box.x1 - px)), max(0, int(box.y1 - py))
    x2, y2 = min(W, int(np.ceil(box.x2 + px))), min(H, int(np.ceil(box.y2 + py)))
    ctx = cv2.GaussianBlur(gray[y1:y2, x1:x2], (3, 3), 0)
    inside = np.zeros(ctx.shape, bool)
    inside[max(0, int(box.y1) - y1):int(np.ceil(box.y2)) - y1,
    max(0, int(box.x1) - x1):int(np.ceil(box.x2)) - x1] = True
    ring = ctx[~inside]
    return ring.size > 0 and float((ring > thr).mean()) > 0.5


def remove_frame(bw: np.ndarray) -> np.ndarray:
    """Erase a printed frame (around the check digit) from a binary crop.

    The digit often touches its frame, so both end up in one connected component,
    and in oblique views the frame is a parallelogram, not a rectangle. A component
    is treated as a frame when it spans most of the crop, its outline simplifies to
    a quadrilateral, and its line is thin compared with its size (this keeps a bold,
    unframed '0'). The line thickness is measured in the middle of each side (the
    thinnest side counts, because the digit may touch the others); the filled frame
    shape is then shrunk (eroded) by 1.5x that thickness, and only the pixels inside
    it are kept, which leaves the digit.
    """
    ch, cw = bw.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    for i in sorted(range(1, n), key=lambda k: -stats[k][cv2.CC_STAT_AREA]):
        fx, fy, fw, fh, _ = stats[i]
        if fw < 0.75 * cw or fh < 0.75 * ch:
            continue
        comp = (lab == i).astype(np.uint8)
        outline = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
                      key=cv2.contourArea)
        quad = cv2.approxPolyDP(outline, 0.04 * cv2.arcLength(outline, True), True)
        if len(quad) != 4:
            continue                                   # not a four-sided outline -> a character

        def run(line: np.ndarray) -> int:              # length of the first run of 1s
            on = np.flatnonzero(line)
            if not on.size:
                return 0
            off = np.flatnonzero(line[on[0]:] == 0)
            return int(off[0]) if off.size else len(line) - on[0]

        def central(a: int, length: int) -> range:
            return range(a + int(0.3 * length), a + int(0.7 * length) + 1)

        sides = [
            np.median([run(comp[fy:fy + fh, x]) for x in central(fx, fw)]),                # top
            np.median([run(comp[fy:fy + fh, x][::-1]) for x in central(fx, fw)]),          # bottom
            np.median([run(comp[y, fx:fx + fw]) for y in central(fy, fh)]),                # left
            np.median([run(comp[y, fx:fx + fw][::-1]) for y in central(fy, fh)]),          # right
        ]
        # The digit can touch the frame on some sides (making those runs long), so the
        # thinnest side gives the real line thickness.
        t = float(min(sides))
        if t < 1 or t / min(fw, fh) > 0.15:
            continue                                   # thick stroke -> a character, not a frame
        filled = np.zeros_like(bw)
        cv2.drawContours(filled, [outline], -1, 255, cv2.FILLED)
        m = int(np.ceil(1.5 * t)) + 1
        # borderValue=0: outside the crop counts as background, so a frame lying on the
        # crop edge is eroded as well
        inner = cv2.erode(filled, np.ones((2 * m + 1, 2 * m + 1), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.bitwise_and(bw, inner)
    return bw


def binarize(gray: np.ndarray, box: Box, pad_frac: float = 0.15,
             framed: bool = False) -> tuple[np.ndarray | None, int, int]:
    """Padded crop -> Otsu -> characters white (-> frame removed). Returns (binary, x1, y1)."""
    H, W = gray.shape
    bw_, bh_ = box.x2 - box.x1, box.y2 - box.y1
    px, py = (pad_frac * bw_ + 2, pad_frac * bh_ + 2) if pad_frac > 0 else (1, 1)
    x1, y1 = max(0, int(box.x1 - px)), max(0, int(box.y1 - py))
    x2, y2 = min(W, int(np.ceil(box.x2 + px))), min(H, int(np.ceil(box.y2 + py)))
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None, x1, y1
    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    thr, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if _background_is_bright(gray, box, thr):     # dark characters -> make them white
        bw = 255 - bw
    if framed:
        cleaned = remove_frame(bw)
        bw = cleaned if not np.array_equal(cleaned, bw) else remove_thin_lines(bw)
    return bw, x1, y1


def debug_strip(gray: np.ndarray, box: Box, pad_frac: float, framed: bool) -> np.ndarray:
    """Picture of a failed character: grey crop with the drawn box | binary after Otsu.
    Saved to review/failed/ so the reason for the failure can be seen."""
    bw, x1, y1 = binarize(gray, box, pad_frac, framed)
    if bw is None:
        return np.zeros((40, 80, 3), np.uint8)
    crop = cv2.cvtColor(gray[y1:y1 + bw.shape[0], x1:x1 + bw.shape[1]], cv2.COLOR_GRAY2BGR)
    cv2.rectangle(crop, (int(box.x1) - x1, int(box.y1) - y1), (int(box.x2) - x1, int(box.y2) - y1), (0, 0, 255), 1)
    strip = np.hstack([crop, np.full((crop.shape[0], 4, 3), 128, np.uint8), cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)])
    return cv2.resize(strip, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


def remove_thin_lines(bw: np.ndarray) -> np.ndarray:
    """Remove structures that are much thinner than the character strokes (a frame line
    around the check digit), whatever their shape. Used when remove_frame finds no
    clean four-sided frame (e.g. the frame is cut by the crop or bent in perspective).

    The thickest stroke is measured with a distance transform; a morphological opening
    with a disc of about half that radius deletes the thin lines and keeps the strokes;
    the kept strokes are then restored to their original outline.
    """
    r_char = float(cv2.distanceTransform((bw > 0).astype(np.uint8), cv2.DIST_L2, 3).max())
    if r_char < 3:
        return bw                                    # strokes too thin to tell apart
    r = max(1, int(0.45 * r_char))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.morphologyEx(bw, cv2.MORPH_OPEN, disc)
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 3, 2 * r + 3))
    return cv2.bitwise_and(bw, cv2.dilate(core, grow))


def extract_mask(gray: np.ndarray, box: Box, pad_frac: float = 0.15,
                 framed: bool = False, fallback: bool = False) -> tuple[np.ndarray | None, list[str]]:
    """Return the character contour (image coordinates) inside `box`, plus QA flags.

    `pad_frac` enlarges the box before thresholding so a tightly drawn box does
    not cut the character. For the check digit use pad_frac=0 and framed=True:
    a printed frame around the digit is then detected and erased (remove_frame).

    `fallback` (used for vertical codes): when dirt or a streak is connected to the
    character and touches the crop border, nothing survives the normal rules. Then the
    binary image is clipped to the drawn box and the largest piece is kept; the mask is
    flagged 'fallback' so it is checked by eye (red outline in the figures).

    Steps: pad the box -> Otsu -> make the character white (decided from a ring
    of container surface around the box) -> connected components -> drop components touching the
    crop border (pieces of neighbours), components that contain another one
    (the frame around the check digit), and specks -> union -> outer contour.
    """
    flags: list[str] = []
    bw_, bh_ = box.x2 - box.x1, box.y2 - box.y1
    bw, x1, y1 = binarize(gray, box, pad_frac, framed)
    if bw is None:
        return None, ["empty_crop"]

    n, lab, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    ch, cw = bw.shape
    cand = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x == 0 or y == 0 or x + w >= cw or y + h >= ch:
            continue                              # touches border -> neighbour / frame
        cx, cy = cents[i][0] + x1, cents[i][1] + y1
        if not (box.x1 <= cx <= box.x2 and box.y1 <= cy <= box.y2):
            continue                              # centre outside the drawn box
        cand.append(i)

    def contains(a: int, b: int) -> bool:
        """a's box encloses b, and b is a real piece (>= 10% of a's area), not a speck:
        that pattern is a frame around a character, so a is the frame."""
        ax, ay, aw, ah = stats[a][:4]
        bx, by, bw2, bh2 = stats[b][:4]
        enclosed = ax < bx and ay < by and ax + aw > bx + bw2 and ay + ah > by + bh2
        return enclosed and stats[b][cv2.CC_STAT_AREA] >= 0.1 * stats[a][cv2.CC_STAT_AREA]

    cand = [a for a in cand if not any(contains(a, b) for b in cand if b != a)]
    cand = [i for i in cand if stats[i][cv2.CC_STAT_AREA] >= 0.03 * bw_ * bh_]   # ignore specks
    if not cand and fallback:
        clip = np.zeros_like(bw)
        bx1, by1 = max(0, int(box.x1) - x1), max(0, int(box.y1) - y1)
        clip[by1:int(np.ceil(box.y2)) - y1, bx1:int(np.ceil(box.x2)) - x1] = 255
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_and(bw, clip), connectivity=8)
        cand = [i for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] >= 0.05 * bw_ * bh_]
        cand = sorted(cand, key=lambda i: -stats[i][cv2.CC_STAT_AREA])[:1]
        if cand:
            flags.append("fallback")
    if not cand:
        return None, ["no_character_found"]
    biggest = max(stats[i][cv2.CC_STAT_AREA] for i in cand)
    keep = [i for i in cand if stats[i][cv2.CC_STAT_AREA] >= 0.10 * biggest]
    if len(keep) > 1:
        flags.append("multi_part")

    mask = np.isin(lab, keep).astype(np.uint8) * 255
    fill = (mask > 0).sum() / max(1.0, bw_ * bh_)
    if fill < 0.08 or fill > 0.85:
        flags.append(f"odd_fill_{fill:.2f}")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 1:                         # stencil font: merge parts into one outline
        contour = cv2.convexHull(np.vstack(contours))
    else:
        contour = contours[0]
    contour = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
    if len(contour) < 3:
        return None, flags + ["degenerate_contour"]
    return contour + np.array([x1, y1]), flags


# ------------------------------------------------------------------------ outputs

def yolo_seg_line(cls: int, contour: np.ndarray, w: int, h: int) -> str:
    coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in contour)
    return f"{cls} {coords}"


def yolo_det_line(cls: int, contour: np.ndarray, w: int, h: int) -> str:
    x, y, bw, bh = cv2.boundingRect(contour.astype(np.int32))
    return f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def paint_out(img: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    """Fill `ignore` regions with the surrounding surface (OpenCV inpainting), so that
    unlabelled characters there are not learned as background."""
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pg.astype(np.int32) for pg in polygons], 255)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def label_strip(img: np.ndarray, code: Code, text: str, title: str) -> np.ndarray:
    """A horizontal code with the character each box was given, written above the box.
    An upside-down code shows labels that do not match the glyphs (e.g. 'M' above a '0')."""
    x, y, w, h = cv2.boundingRect(code.polygon.astype(np.int32))
    m = 6
    x0, y0 = max(0, x - m), max(0, y - m)
    crop = img[y0:y + h + m, x0:x + w + m].copy()
    scale = 900 / max(1, crop.shape[1])
    crop = cv2.resize(crop, None, fx=scale, fy=scale)
    head = np.full((46, crop.shape[1], 3), 255, np.uint8)
    for b, ch in zip(reading_order(code.boxes), text):
        cx = int((b.centre[0] - x0) * scale)
        cv2.putText(head, ch, (cx - 8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2, cv2.LINE_AA)
    cv2.putText(head, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, crop])


def draw_overlay(img: np.ndarray, insts: list[Instance], codes: list[Code], status: str) -> np.ndarray:
    vis = img.copy()
    layer = vis.copy()
    for code in codes:
        cv2.polylines(vis, [code.polygon.astype(np.int32)], True, (255, 160, 0), 2)
    for inst in insts:
        if inst.contour is None:
            continue
        colour = (0, 0, 255) if inst.flags else (0, 220, 0)
        cv2.fillPoly(layer, [inst.contour.astype(np.int32)], colour)
    vis = cv2.addWeighted(layer, 0.45, vis, 0.55, 0)
    for inst in insts:
        if inst.contour is None:
            continue
        x, y, w, _ = cv2.boundingRect(inst.contour.astype(np.int32))
        cv2.putText(vis, CLASSES[inst.cls], (x + w + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 200, 0) if status == "OK" else (0, 0, 255), 3, cv2.LINE_AA)
    return vis


SPLITS = ("train", "val", "test")


def grouped_split(keys: list[str], ratios: tuple[float, float, float], seed: int,
                  previous: dict[str, str] | None = None,
                  strata: dict[str, str] | None = None) -> dict[str, str]:
    """Assign whole containers to train/val/test. Returns container -> split.

    - Grouped: a container (all its images, from every camera) goes to one split.
    - Persistent: containers in `previous` keep their split, so adding annotations
      never moves an old container (a test container can never become training data).
    - Stratified: `strata` gives each container a variation group (G1..G8). Each new
      container goes to the split that is furthest below its target share *within its
      own group*, so every group is spread over train/val/test in the same proportions.
    """
    keys_set = set(keys)
    strata = strata or {}
    out = {g: sp for g, sp in (previous or {}).items() if g in keys_set}
    new = sorted(keys_set - set(out))
    random.Random(seed).shuffle(new)
    target = dict(zip(SPLITS, ratios))
    for g in new:
        same = [sp for c, sp in out.items() if strata.get(c, "") == strata.get(g, "")]
        total = len(same) + 1
        out[g] = max(SPLITS, key=lambda sp: target[sp] * total - same.count(sp))
    return out


def load_assignments(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {r["container_id"]: r["split"] for r in csv.DictReader(fh)}


def save_assignments(path: Path, assignments: dict[str, str], previous: dict[str, str]) -> None:
    merged = {**previous, **assignments}            # keep containers that are REVIEW for now
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["container_id", "split"])
        wr.writerows(sorted(merged.items()))


def write_split_summary(out: Path, rows: list[dict[str, str]]) -> None:
    """split_summary.csv: number of images per camera / group / layout / angle / condition
    and split - the diversity table for the report."""
    with open(out / "split_summary.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["variable", "value", *SPLITS, "total"])
        for var in ("camera", *META_FIELDS):
            for val in sorted({r[var] for r in rows}):
                n = [sum(r[var] == val and r["split"] == sp for r in rows) for sp in SPLITS]
                wr.writerow([var, val, *n, sum(n)])


def write_dataset_info(out: Path, prepared, truth: dict[str, str], split_of: dict[str, str],
                       n_annotated: int, n_excluded: int = 0) -> None:
    """Write dataset_info.json (sizes per split) and class_counts.csv (instances per class)."""
    info = {"annotated_images": n_annotated, "excluded_images": n_excluded,
            "review_images": n_annotated - n_excluded - len(prepared), "ok_images": len(prepared)}
    class_counts = {sp: [0] * len(CLASSES) for sp in SPLITS}
    for sp in SPLITS:
        keys = [k for k, *_ in prepared if split_of[truth[k]] == sp]
        info[f"{sp}_images"] = len(keys)
        info[f"{sp}_containers"] = len({truth[k] for k in keys})
    for key, _img, seg, _det in prepared:
        for line in seg:
            class_counts[split_of[truth[key]]][int(line.split()[0])] += 1
    for sp in SPLITS:
        info[f"{sp}_characters"] = sum(class_counts[sp])
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    with open(out / "class_counts.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", *SPLITS])
        wr.writerows([c, *(class_counts[sp][i] for sp in SPLITS)] for i, c in enumerate(CLASSES))


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, default=Path("data/02_interim/cvat_export"),
                    help="unzipped CVAT export folder")
    ap.add_argument("--csv", type=Path, default=Path("data/02_interim/labels.csv"),
                    help="labels.csv with filename,container_id")
    ap.add_argument("--out", type=Path, default=Path("data/03_processed/char_dataset"),
                    help="output folder (deleted and rebuilt on every run)")
    ap.add_argument("--split", type=float, nargs=3, default=(0.6, 0.2, 0.2), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--assignments", type=Path, default=Path("data/02_interim/split_assignments.csv"),
                    help="persistent container -> split table (kept between runs)")
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata"),
                    help="folder with Label-camera-*.csv selection sheets (group, layout, ...)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    images = {image_key(p.name): p for p in args.export.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    with open(args.csv, newline="") as fh:
        truth = {image_key(r["filename"]): r["container_id"].strip().upper().replace(" ", "")
                 for r in csv.DictReader(fh)}

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "review").mkdir(parents=True)

    prepared: list[tuple[str, Path, list[str], list[str]]] = []   # key, img, seg, det
    report_rows = []
    flag_rows = []                                                # one row per suspicious mask
    strips = []                                                   # horizontal codes, for the orientation check
    ignore_of = {k: pgs for k, pgs in load_ignore(args.export).items() if pgs}
    meta = load_metadata(sorted(args.metadata.glob("Label-camera-*.csv"))) if args.metadata.exists() else {}
    if not meta:
        log.warning("No metadata in %s: split is grouped but not stratified by variation group.", args.metadata)

    for name, (w, h, codes, loose) in load_annotations(args.export).items():
        key = image_key(name)
        if meta_for(meta, key)["status"] == "rejected":
            # Removed on purpose: status 'rejected' in the selection sheet (reason in notes).
            note = meta_for(meta, key)["notes"] or "no reason given"
            report_rows.append([name, "EXCLUDED", truth.get(key, ""), 0, 0, f"rejected in selection sheet: {note}"])
            log.info("%-8s %-40s %s", "EXCLUDED", key, note)
            continue
        reasons: list[str] = []
        text = truth.get(key)
        if text is None:
            reasons.append("not_in_csv")
        elif not iso6346_valid(text):
            reasons.append(f"csv_code_invalid:{text}")
        if key not in images:
            reasons.append("image_file_missing")
        if not codes:
            reasons.append("no_container_id_polygon")
        if loose:
            reasons.append(f"{len(loose)}_char_boxes_outside_any_code")
        for j, code in enumerate(codes):
            if len(code.boxes) != CODE_LEN:
                reasons.append(f"code{j}_has_{len(code.boxes)}_boxes")

        insts: list[Instance] = []
        img = cv2.imread(str(images[key])) if key in images else None
        if img is not None and text and len(text) == CODE_LEN:
            for j, code in enumerate(codes):
                _, _, cw_, ch_ = cv2.boundingRect(code.polygon.astype(np.int32))
                if cw_ > ch_ and len(code.boxes) == CODE_LEN:
                    strips.append(label_strip(img, code, text, f"{key}  code{j}"))
        if not reasons and img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            failed_at: list[str] = []
            for j, code in enumerate(codes):
                _, _, cw_, ch_ = cv2.boundingRect(code.polygon.astype(np.int32))
                vertical = ch_ > cw_
                for pos, (box, ch) in enumerate(zip(reading_order(code.boxes), text)):
                    # The check digit (last position) often sits in a printed frame:
                    # no padding there, so the frame stays outside the crop.
                    is_check = pos == CODE_LEN - 1
                    contour, flags = extract_mask(gray, box, pad_frac=0.0 if is_check else 0.15,
                                                  framed=is_check, fallback=vertical)
                    insts.append(Instance(CLASS_ID[ch], contour, flags))
                    if contour is None:
                        failed_at.append(f"code{j}:{pos + 1}={ch}")
                        fail_dir = args.out / "review" / "failed"
                        fail_dir.mkdir(exist_ok=True)
                        cv2.imwrite(str(fail_dir / f"{key}_code{j}_pos{pos + 1:02d}_{ch}.png"),
                                    debug_strip(gray, box, 0.0 if is_check else 0.15, is_check))
            if failed_at:   # e.g. "masks_failed code0:1=S code0:11=6" (position 1-11 = character)
                reasons.append("masks_failed " + " ".join(failed_at))

        status = "OK" if not reasons else "REVIEW"
        n_flagged = sum(bool(i.flags) for i in insts)
        report_rows.append([name, status, text or "", len(insts), n_flagged, "; ".join(reasons)])
        if img is not None:
            overlay = draw_overlay(img, insts, codes, status)
            for pg in ignore_of.get(key, []):
                cv2.polylines(overlay, [pg.astype(np.int32)], True, (160, 160, 160), 3)
            cv2.imwrite(str(args.out / "review" / f"{status}_{key}.jpg"), overlay)
        if status == "OK":
            flag_rows += [[key, n, CLASSES[i.cls], "; ".join(i.flags)]
                          for n, i in enumerate(insts) if i.flags]
            seg = [yolo_seg_line(i.cls, i.contour, w, h) for i in insts]
            det = [yolo_det_line(i.cls, i.contour, w, h) for i in insts]
            prepared.append((key, images[key], seg, det))
        log.info("%-8s %-40s %s", status, key, "; ".join(reasons))

    with open(args.out / "report.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "status", "container_id", "n_chars", "n_flagged_masks", "reasons"])
        wr.writerows(report_rows)

    if strips:   # one sheet: check that the red labels match the characters (upside-down codes do not)
        cv2.imwrite(str(args.out / "review" / "horizontal_codes.jpg"), np.vstack(strips))
        log.info("Check the orientation of %d horizontal codes in %s", len(strips),
                 args.out / "review" / "horizontal_codes.jpg")

    with open(args.out / "mask_flags.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "instance", "char", "flags"])   # instance = line in the label file
        wr.writerows(flag_rows)

    if not prepared:
        log.error("No image passed the checks - see %s", args.out / "report.csv")
        return

    strata = {truth[k]: meta_for(meta, k)["group"] for k, *_ in prepared}
    previous = load_assignments(args.assignments)
    split_of = grouped_split([truth[k] for k, *_ in prepared], tuple(args.split), args.seed,
                             previous, strata)
    save_assignments(args.assignments, split_of, previous)
    for task, idx in (("seg", 2), ("det", 3)):
        root = args.out / f"yolo_{task}"
        for key, img_path, *labels in prepared:
            sp = split_of[truth[key]]
            (root / "images" / sp).mkdir(parents=True, exist_ok=True)
            (root / "labels" / sp).mkdir(parents=True, exist_ok=True)
            target = root / "images" / sp / f"{key}{img_path.suffix}"
            if key in ignore_of:      # paint out regions that are deliberately not annotated
                cv2.imwrite(str(target), paint_out(cv2.imread(str(img_path)), ignore_of[key]))
            else:
                shutil.copy(img_path, target)
            (root / "labels" / sp / f"{key}.txt").write_text("\n".join(labels[idx - 2]) + "\n")
        names = "\n".join(f"  {i}: '{c}'" for i, c in enumerate(CLASSES))
        (root / "data.yaml").write_text(
            f"path: {root.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")

    with open(args.out / "split.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        split_rows = [{"image": k, "container_id": truth[k], "split": split_of[truth[k]], **meta_for(meta, k)}
                      for k, *_ in prepared]
        cols = ["image", "container_id", "split", "camera", *META_FIELDS]
        wr.writerow(cols)
        wr.writerows([r[c] for c in cols] for r in split_rows)
    write_split_summary(args.out, split_rows)

    counts = {s: sum(1 for k, *_ in prepared if split_of[truth[k]] == s) for s in SPLITS}
    n_excluded = sum(r[1] == "EXCLUDED" for r in report_rows)
    write_dataset_info(args.out, prepared, truth, split_of, len(report_rows), n_excluded)
    log.info("OK images: %d / %d  | split %s", len(prepared), len(report_rows), counts)
    log.info("Check the overlays in %s before training.", args.out / "review")


if __name__ == "__main__":
    main()