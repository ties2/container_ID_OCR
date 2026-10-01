"""Turn a CVAT export into YOLO datasets for per-character segmentation and detection.

Input (from CVAT "CVAT for images 1.1" export with images):
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
    6. Split train/val/test by container ID (no container in two sets).
    7. Save overlay images and a report so every mask can be checked by eye.

Usage (from the project root):
    python -m src.data.prepare_dataset            # uses the default project paths
    make prepare                                  # same thing

Default paths:
    data/02_interim/cvat_export/   unzipped CVAT export (annotations.xml + images/)
    data/02_interim/labels.csv     filename,container_id
    data/03_processed/char_dataset/  output (yolo_seg/, yolo_det/, review/, report.csv)
"""
from __future__ import annotations

import argparse
import csv
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
    stem = re.sub(r"\.rf\.[0-9a-f]+", "", stem)
    stem = re.sub(r"\.(jpe?g|png)$", "", stem, flags=re.I)
    return re.sub(r"_jpe?g$", "", stem, flags=re.I)


def group_id(name: str) -> str:
    """Capture-event id from the filename ('1-122830001-OCR-...' -> '122830001')."""
    parts = image_key(name).split("-")
    return parts[1] if len(parts) > 1 else image_key(name)


# --------------------------------------------------------------------- annotation

def parse_cvat(xml_path: Path) -> dict[str, tuple[int, int, list[Code], list[Box]]]:
    """Read a CVAT-for-images-1.1 file. Returns name -> (w, h, codes, unassigned boxes)."""
    out = {}
    for img in ET.parse(xml_path).getroot().iter("image"):
        w, h = int(img.get("width")), int(img.get("height"))
        codes = [
            Code(np.array([[float(v) for v in p.split(",")] for p in poly.get("points").split(";")],
                          dtype=np.float32))
            for poly in img.iter("polygon") if poly.get("label") == "container_id"
        ]
        boxes = [
            Box(*(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")))
            for b in img.iter("box") if b.get("label") == "char"
        ]
        loose = []
        for b in boxes:
            home = next((c for c in codes
                         if cv2.pointPolygonTest(c.polygon, b.centre, False) >= 0), None)
            (home.boxes if home else loose).append(b)
        out[img.get("name")] = (w, h, codes, loose)
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
    """Erase a printed rectangular frame (around the check digit) from a binary crop.

    The digit often touches its frame, so both end up in one connected
    component. A component is treated as a frame when it spans most of the
    crop, its outline is close to a rectangle and its line is thin compared
    with its size (this keeps a bold, unframed '0'). The line thickness is
    measured in the middle of each side; everything within 1.5x that
    thickness of the frame's edge is erased, which leaves only the digit.
    """
    ch, cw = bw.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    for i in sorted(range(1, n), key=lambda k: -stats[k][cv2.CC_STAT_AREA]):
        fx, fy, fw, fh, _ = stats[i]
        if fw < 0.75 * cw or fh < 0.75 * ch:
            continue
        comp = (lab == i).astype(np.uint8)
        outline = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                      key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(outline)
        if rw * rh == 0 or cv2.contourArea(outline) / (rw * rh) < 0.88:
            continue                                   # not rectangular -> a character

        def run(line: np.ndarray) -> int:              # length of the first run of 1s
            off = np.flatnonzero(line == 0)
            return int(off[0]) if off.size else len(line)

        def central(a: int, length: int) -> range:
            return range(a + int(0.3 * length), a + int(0.7 * length) + 1)

        sides = [
            np.median([run(comp[fy:fy + fh, x]) for x in central(fx, fw)]),                # top
            np.median([run(comp[fy:fy + fh, x][::-1]) for x in central(fx, fw)]),          # bottom
            np.median([run(comp[y, fx:fx + fw]) for y in central(fy, fh)]),                # left
            np.median([run(comp[y, fx:fx + fw][::-1]) for y in central(fy, fh)]),          # right
        ]
        t = float(np.median(sides))
        if t < 1 or t / min(fw, fh) > 0.12:
            continue                                   # thick stroke -> a character, not a frame
        m = int(np.ceil(1.5 * t)) + 1
        out = np.zeros_like(bw)
        out[fy + m:fy + fh - m, fx + m:fx + fw - m] = bw[fy + m:fy + fh - m, fx + m:fx + fw - m]
        return out
    return bw


def extract_mask(gray: np.ndarray, box: Box, pad_frac: float = 0.15,
                 framed: bool = False) -> tuple[np.ndarray | None, list[str]]:
    """Return the character contour (image coordinates) inside `box`, plus QA flags.

    `pad_frac` enlarges the box before thresholding so a tightly drawn box does
    not cut the character. For the check digit use pad_frac=0 and framed=True:
    a printed frame around the digit is then detected and erased (remove_frame).

    Steps: pad the box -> Otsu -> make the character white (decided from a ring
    of container surface around the box) -> connected components -> drop components touching the
    crop border (pieces of neighbours), components that contain another one
    (the frame around the check digit), and specks -> union -> outer contour.
    """
    flags: list[str] = []
    H, W = gray.shape
    bw_, bh_ = box.x2 - box.x1, box.y2 - box.y1
    px, py = (pad_frac * bw_ + 2, pad_frac * bh_ + 2) if pad_frac > 0 else (1, 1)
    x1, y1 = max(0, int(box.x1 - px)), max(0, int(box.y1 - py))
    x2, y2 = min(W, int(np.ceil(box.x2 + px))), min(H, int(np.ceil(box.y2 + py)))
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None, ["empty_crop"]

    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    thr, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if _background_is_bright(gray, box, thr):     # dark characters -> make them white
        bw = 255 - bw
    if framed:
        bw = remove_frame(bw)

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
        ax, ay, aw, ah = stats[a][:4]
        bx, by, bw2, bh2 = stats[b][:4]
        return ax < bx and ay < by and ax + aw > bx + bw2 and ay + ah > by + bh2

    cand = [a for a in cand if not any(contains(a, b) for b in cand if b != a)]
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


def grouped_split(keys: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    """Assign whole containers to train/val/test. Returns container -> split."""
    groups = sorted(set(keys))
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_val = max(1, round(ratios[1] * n))
    n_test = max(1, round(ratios[2] * n))
    out = {}
    for i, g in enumerate(groups):
        out[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


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
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    xml_path = args.export / "annotations.xml"
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
    for name, (w, h, codes, loose) in parse_cvat(xml_path).items():
        key = image_key(name)
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
        if not reasons and img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for code in codes:
                for pos, (box, ch) in enumerate(zip(reading_order(code.boxes), text)):
                    # The check digit (last position) often sits in a printed frame:
                    # no padding there, so the frame stays outside the crop.
                    is_check = pos == CODE_LEN - 1
                    contour, flags = extract_mask(gray, box, pad_frac=0.0 if is_check else 0.15,
                                                  framed=is_check)
                    insts.append(Instance(CLASS_ID[ch], contour, flags))
            failed = [i for i in insts if i.contour is None]
            if failed:
                reasons.append(f"{len(failed)}_masks_failed")

        status = "OK" if not reasons else "REVIEW"
        n_flagged = sum(bool(i.flags) for i in insts)
        report_rows.append([name, status, text or "", len(insts), n_flagged, "; ".join(reasons)])
        if img is not None:
            cv2.imwrite(str(args.out / "review" / f"{status}_{key}.jpg"), draw_overlay(img, insts, codes, status))
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

    with open(args.out / "mask_flags.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "instance", "char", "flags"])   # instance = line in the label file
        wr.writerows(flag_rows)

    if not prepared:
        log.error("No image passed the checks - see %s", args.out / "report.csv")
        return

    split_of = grouped_split([truth[k] for k, *_ in prepared], tuple(args.split), args.seed)
    for task, idx in (("seg", 2), ("det", 3)):
        root = args.out / f"yolo_{task}"
        for key, img_path, *labels in prepared:
            sp = split_of[truth[key]]
            (root / "images" / sp).mkdir(parents=True, exist_ok=True)
            (root / "labels" / sp).mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, root / "images" / sp / f"{key}{img_path.suffix}")
            (root / "labels" / sp / f"{key}.txt").write_text("\n".join(labels[idx - 2]) + "\n")
        names = "\n".join(f"  {i}: '{c}'" for i, c in enumerate(CLASSES))
        (root / "data.yaml").write_text(
            f"path: {root.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")

    with open(args.out / "split.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "container_id", "split"])
        wr.writerows([k, truth[k], split_of[truth[k]]] for k, *_ in prepared)

    counts = {s: sum(1 for k, *_ in prepared if split_of[truth[k]] == s) for s in ("train", "val", "test")}
    log.info("OK images: %d / %d  | split %s", len(prepared), len(report_rows), counts)
    log.info("Check the overlays in %s before training.", args.out / "review")


if __name__ == "__main__":
    main()