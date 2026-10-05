"""
src/visualization/plot_char_dataset.py
======================================
Show the character dataset as rows of panels:

    image  |  ground truth (CVAT)  |  character masks  [|  model prediction]

- image:          the original photo, cropped around the container codes
- ground truth:   what was drawn in CVAT (code polygon + one box per character)
- masks:          the pixel masks made by prepare_dataset.py (one colour per
                  character, with its class), i.e. what YOLO is trained on.
                  Masks flagged as suspicious (mask_flags.csv) get a red outline.
- prediction:     optional, the masks predicted by a trained model (--weights)

Usage (from the project root):
    python -m src.visualization.plot_char_dataset                 # 6 random images
    python -m src.visualization.plot_char_dataset --n 4 --split test
    python -m src.visualization.plot_char_dataset --split test \\
        --weights results/yolo/seg-s42/weights/best.pt           # adds a 4th panel
    python -m src.visualization.plot_char_dataset --show         # also open the figures

Output: one PNG per image + overview.png (all rows in one figure) in --out.
With --show the figures are also displayed (PyCharm shows them in its Plots
tool window; on a machine without a screen, matplotlib only saves them).
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data.prepare_dataset import CLASSES, image_key, load_annotations

log = logging.getLogger("plot_char_dataset")

_CMAP = plt.get_cmap("tab20")


def _colour(i: int) -> tuple[int, int, int]:
    r, g, b, _ = _CMAP(i % 20)
    return int(255 * r), int(255 * g), int(255 * b)


def load_seg_labels(path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    """Read a YOLO-seg label file -> [(class_id, polygon in pixels)]."""
    out = []
    for line in path.read_text().splitlines():
        vals = line.split()
        if len(vals) < 7:
            continue
        pts = np.array(vals[1:], dtype=np.float32).reshape(-1, 2) * [w, h]
        out.append((int(vals[0]), pts.astype(np.int32)))
    return out


def crop_window(polys: list[np.ndarray], w: int, h: int, margin: int = 40) -> tuple[int, int, int, int]:
    """Bounding window around all code polygons (whole image if there are none)."""
    if not polys:
        return 0, 0, w, h
    pts = np.vstack(polys)
    x1, y1 = np.maximum(pts.min(axis=0) - margin, 0).astype(int)
    x2, y2 = np.minimum(pts.max(axis=0) + margin, [w, h]).astype(int)
    return x1, y1, x2, y2


def draw_ground_truth(img: np.ndarray, codes, loose_boxes) -> np.ndarray:
    vis = img.copy()
    for code in codes:
        cv2.polylines(vis, [code.polygon.astype(np.int32)], True, (255, 140, 0), 3)
        for b in code.boxes:
            cv2.rectangle(vis, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 200, 255), 2)
    for b in loose_boxes:   # boxes outside every code polygon (an annotation error)
        cv2.rectangle(vis, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (255, 0, 0), 3)
    return vis


def draw_masks(shape: tuple[int, int], instances: list[tuple[int, np.ndarray]],
               flagged: set[int] | None = None) -> np.ndarray:
    """Black canvas, each character filled in its own colour and labelled with its class.

    Instances whose index is in `flagged` get a red outline (check them by eye).
    """
    canvas = np.zeros((*shape, 3), np.uint8)
    for i, (cls, poly) in enumerate(instances):
        cv2.fillPoly(canvas, [poly], _colour(i))
        if flagged and i in flagged:
            cv2.polylines(canvas, [poly], True, (255, 0, 0), 3)
    for cls, poly in instances:
        x, y, w, _ = cv2.boundingRect(poly)
        cv2.putText(canvas, CLASSES[cls], (x + w + 3, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def predict_instances(model, img_bgr: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Run a trained YOLO model and return [(class_id, polygon)].

    Segmentation models give mask outlines; detection models give boxes, which are
    returned as 4-point polygons so both can be drawn the same way.
    """
    res = model.predict(img_bgr, verbose=False)[0]
    classes = [int(c) for c in res.boxes.cls.tolist()]
    if res.masks is not None:
        return [(c, p.astype(np.int32)) for c, p in zip(classes, res.masks.xy) if len(p) >= 3]
    return [(c, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32))
            for c, (x1, y1, x2, y2) in zip(classes, res.boxes.xyxy.tolist())]


def render(dataset: Path, export: Path, out: Path, split: str = "all", n: int = 6, seed: int = 0,
           weights: Path | None = None, show: bool = False, overview_name: str = "overview.png",
           per_image: bool = True, predictions: dict | None = None,
           row_notes: dict[str, str] | None = None, title: str | None = None) -> Path | None:
    """Draw the rows and save them. Returns the path of the overview figure (None if no images).

    predictions: image key -> [(class_id, polygon)], already computed (e.g. by evaluate.py);
                 if not given and `weights` is set, the model is run here.
    row_notes:   image key -> text shown above that row (e.g. "GT ... | pred ...").
    title:       text shown above the whole overview (e.g. the main metrics).
    """
    with open(dataset / "split.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if split in ("all", r["split"])]
    if not rows:
        log.error("No images for split '%s' in %s", split, dataset / "split.csv")
        return None
    random.Random(seed).shuffle(rows)
    if n:
        rows = rows[:n]

    annotations = {image_key(name): v for name, v in load_annotations(export).items()}
    flagged: dict[str, set[int]] = {}
    flags_csv = dataset / "mask_flags.csv"
    if flags_csv.exists():
        with open(flags_csv, newline="") as fh:
            for fr in csv.DictReader(fh):
                flagged.setdefault(fr["image"], set()).add(int(fr["instance"]))
    model = None
    if weights and predictions is None:
        from ultralytics import YOLO   # imported only when needed
        model = YOLO(str(weights))
    with_pred = model is not None or predictions is not None
    row_notes = row_notes or {}

    titles = ["Image", "Ground truth (CVAT)", "Character masks"] + (["Prediction"] if with_pred else [])
    out.mkdir(parents=True, exist_ok=True)
    all_panels = []

    for r in rows:
        key, sp = r["image"], r["split"]
        img_path = next((dataset / "yolo_seg" / "images" / sp).glob(f"{key}.*"))
        img_bgr = cv2.imread(str(img_path))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        _, _, codes, loose = annotations[key]
        masks = load_seg_labels(dataset / "yolo_seg" / "labels" / sp / f"{key}.txt", w, h)

        x1, y1, x2, y2 = crop_window([c.polygon for c in codes], w, h)
        panels = [img, draw_ground_truth(img, codes, loose), draw_masks((h, w), masks, flagged.get(key))]
        if predictions is not None:
            panels.append(draw_masks((h, w), predictions.get(key, [])))
        elif model:
            panels.append(draw_masks((h, w), predict_instances(model, img_bgr)))
        panels = [p[y1:y2, x1:x2] for p in panels]
        all_panels.append((row_notes.get(key, f"{key}  [{sp}]  {r['container_id']}"), panels))

        if per_image:
            fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.5))
            for ax, p, t in zip(axes, panels, titles):
                ax.imshow(p)
                ax.set_title(t, fontsize=11)
                ax.axis("off")
            fig.suptitle(all_panels[-1][0], fontsize=10)
            fig.tight_layout()
            fig.savefig(out / f"{key}.png", dpi=150)
            if not show:
                plt.close(fig)

    # One sub-figure per image: its own heading (image, GT vs prediction) and four panels.
    cols = len(titles)
    fig = plt.figure(figsize=(3.4 * cols, 3.6 * len(all_panels) + (0.5 if title else 0)), layout="constrained")
    subfigs = np.atleast_1d(fig.subfigures(len(all_panels), 1))
    for sf, (label, panels) in zip(subfigs, all_panels):
        sf.suptitle(label, fontsize=8, family="monospace", x=0.01, ha="left")
        for ax, p, t in zip(np.atleast_1d(sf.subplots(1, cols)), panels, titles):
            ax.imshow(p)
            ax.set_title(t, fontsize=9)
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10, family="monospace", fontweight="bold")
    overview = out / overview_name
    fig.savefig(overview, dpi=150)
    log.info("Saved %d rows to %s", len(all_panels), overview)
    if show:
        plt.show()
    plt.close("all")
    return overview


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("data/03_processed/char_dataset"))
    ap.add_argument("--export", type=Path, default=Path("data/02_interim/cvat_export"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/char_samples"))
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    ap.add_argument("--n", type=int, default=6, help="number of images (0 = all)")
    ap.add_argument("--weights", type=Path, help="trained YOLO-seg weights -> adds a prediction panel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="also display the figures (e.g. in PyCharm)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    render(args.dataset, args.export, args.out, args.split, args.n, args.seed, args.weights, args.show)


if __name__ == "__main__":
    main()