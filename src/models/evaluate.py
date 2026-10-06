"""
src/models/evaluate.py
======================
Evaluate a trained run on the validation set (default) or, once at the end, the test set.

Two kinds of numbers:
    1. Localisation (Ultralytics): precision, recall, mAP50, mAP50-95 for boxes (B)
       and, for segmentation, masks (M).
    2. Reading (the research question): predicted characters inside each annotated
       code region are ordered along the code's direction and compared with the
       typed code:
           char_acc       1 - edit distance / 11  (share of the code read correctly)
           full_code_acc  share of codes with all 11 characters correct
           checkdigit_ok  share of reads that pass the ISO 6346 check digit
       Prediction uses class-agnostic NMS, so one location gives one character.
       Reads are also reported after an ISO 6346 *format* correction (positions 1-4
       must be letters, 5-11 digits: 0->O, 1->I, ... and back), and per code layout
       (vertical / horizontal).

Output (results/history/, same stamp as the training run):
    <stamp>_<run>_<split>-eval.txt        all metrics + every read
    <stamp>_<run>_<split>-eval_reads.csv  one row per code: truth, prediction, scores
    <stamp>_<run>_<split>-eval.png        image | ground truth | masks | prediction,
                                          main metrics above, GT vs prediction per row
The figure is also shown (PyCharm: Plots window). Metrics are added to the run in MLflow.

Usage (from the project root):
    python -m src.models.evaluate --run seg-s42                 # validation set
    python -m src.models.evaluate --run det-s42 --n 0           # all validation images
    python -m src.models.evaluate --run seg-s42 --split test    # final numbers, once
    make evaluate RUN=seg-s42
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.data.prepare_dataset import CLASSES, Box, image_key, iso6346_valid, load_annotations, reading_order

log = logging.getLogger("evaluate")

HISTORY = Path("results/history")
PROJECT = Path("results/yolo")
EXPERIMENT = "char-seg-vs-det"


# ----------------------------------------------------------------- reading logic

def levenshtein(a: str, b: str) -> int:
    """Edit distance: insertions, deletions and substitutions needed to turn a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def read_code(instances: list[tuple[int, np.ndarray]], polygon: np.ndarray) -> str:
    """Characters whose centre lies inside the code polygon, in reading order."""
    boxes, chars = [], {}
    for cls, poly in instances:
        x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
        box = Box(x, y, x + w, y + h)
        if cv2.pointPolygonTest(polygon.astype(np.float32), box.centre, False) >= 0:
            boxes.append(box)
            chars[id(box)] = CLASSES[cls]
    if not boxes:
        return ""
    if len(boxes) == 1:
        return chars[id(boxes[0])]
    return "".join(chars[id(b)] for b in reading_order(boxes))


# Characters that look alike. ISO 6346: positions 1-4 are letters, 5-11 are digits.
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
_LETTER_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}


def iso_format_correct(text: str) -> str:
    """Swap look-alike characters that are impossible at their position.

    Only applied to complete 11-character reads (otherwise the positions are unknown).
    """
    if len(text) != 11:
        return text
    return ("".join(_DIGIT_TO_LETTER.get(c, c) for c in text[:4])
            + "".join(_LETTER_TO_DIGIT.get(c, c) for c in text[4:]))


def code_layout(polygon: np.ndarray) -> str:
    _, _, w, h = cv2.boundingRect(polygon.astype(np.int32))
    return "vertical" if h > w else "horizontal"


def score_read(pred: str, truth: str) -> dict[str, float]:
    return {
        "char_acc": max(0.0, 1 - levenshtein(pred, truth) / len(truth)),
        "exact": float(pred == truth),
        "checkdigit_ok": float(iso6346_valid(pred)),
    }


# ------------------------------------------------------------------ predictions

def predict(model, img_bgr: np.ndarray, conf: float, imgsz: int) -> list[tuple[int, np.ndarray]]:
    """[(class_id, polygon)] - mask outlines for seg models, box corners for det models."""
    res = model.predict(img_bgr, conf=conf, imgsz=imgsz, agnostic_nms=True, verbose=False)[0]
    classes = [int(c) for c in res.boxes.cls.tolist()]
    if res.masks is not None:
        return [(c, p.astype(np.int32)) for c, p in zip(classes, res.masks.xy) if len(p) >= 3]
    return [(c, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32))
            for c, (x1, y1, x2, y2) in zip(classes, res.boxes.xyxy.tolist())]


def training_stamp(run: str) -> str:
    """Reuse the stamp of the training run so all files of a run sort together."""
    found = sorted(HISTORY.glob(f"*_{run}.txt"))
    return found[-1].name.split("_")[0] if found else datetime.now().strftime("%Y%m%d-%H%M%S")


# ------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name, e.g. seg-s42 or det-1cls-s7")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--n", type=int, default=6, help="rows in the figure (0 = all images)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for reading")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dataset", type=Path, default=Path("data/03_processed/char_dataset"))
    ap.add_argument("--export", type=Path, default=Path("data/02_interim/cvat_export"))
    ap.add_argument("--mlflow-uri", default="sqlite:///results/mlflow.db")
    ap.add_argument("--no-show", action="store_true", help="only save the figure")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    task = args.run.split("-")[0]
    single_cls = "-1cls" in args.run
    weights = PROJECT / args.run / "weights" / "best.pt"
    data_yaml = args.dataset / f"yolo_{task}" / "data.yaml"
    for path in (weights, data_yaml):
        if not path.exists():
            raise SystemExit(f"{path} not found.")

    from ultralytics import YOLO
    model = YOLO(str(weights))

    # 1. Localisation metrics on the chosen split.
    res = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, device=args.device,
                    single_cls=single_cls, project=str(PROJECT.resolve()),
                    name=f"{args.run}-{args.split}", exist_ok=True, verbose=False)
    loc = {k.replace("(", "").replace(")", ""): float(v) for k, v in res.results_dict.items()}

    # 2. Reading metrics: predictions grouped by the annotated code polygons.
    with open(args.dataset / "split.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == args.split]
    annotations = {image_key(n): v for n, v in load_annotations(args.export).items()}
    predictions, notes, reads = {}, {}, []
    for r in rows:
        key, truth = r["image"], r["container_id"]
        img_path = next((args.dataset / f"yolo_{task}" / "images" / args.split).glob(f"{key}.*"))
        predictions[key] = predict(model, cv2.imread(str(img_path)), args.conf, args.imgsz)
        texts = []
        for j, code in enumerate(annotations[key][2]):
            pred = read_code(predictions[key], code.polygon)
            fixed = iso_format_correct(pred)
            sc = score_read(pred, truth)
            fx = score_read(fixed, truth)
            reads.append({"image": key, "code": j, "layout": code_layout(code.polygon),
                          "truth": truth, "pred": pred, **sc,
                          "pred_fmt": fixed, **{f"{k}_fmt": v for k, v in fx.items()},
                          "group": r.get("group", "?"), "camera": r.get("camera", "?")})
            verdict = "OK" if sc["exact"] else f"{sc['char_acc']:.0%}"
            texts.append(f"{pred or '-':11s} {verdict}")
        notes[key] = f"{key}  GT {truth}  pred " + "  /  ".join(texts)

    def summarise(subset: list[dict], prefix: str) -> dict[str, float]:
        if not subset:
            return {}
        return {
            f"{prefix}/char_acc": float(np.mean([x["char_acc"] for x in subset])),
            f"{prefix}/full_code_acc": float(np.mean([x["exact"] for x in subset])),
            f"{prefix}/checkdigit_ok": float(np.mean([x["checkdigit_ok"] for x in subset])),
            f"{prefix}_fmt/char_acc": float(np.mean([x["char_acc_fmt"] for x in subset])),
            f"{prefix}_fmt/full_code_acc": float(np.mean([x["exact_fmt"] for x in subset])),
            f"{prefix}/n_codes": float(len(subset)),
        }

    reading = {}
    if reads and not single_cls:
        reading = summarise(reads, "reading")
        for lay in ("vertical", "horizontal"):
            reading |= summarise([x for x in reads if x["layout"] == lay], f"reading_{lay}")

    # 3. Report: text, CSV, figure.
    HISTORY.mkdir(parents=True, exist_ok=True)
    prefix = HISTORY / f"{training_stamp(args.run)}_{args.run}_{args.split}-eval"
    key_metrics = [("mask mAP50", loc.get("metrics/mAP50M")), ("box mAP50", loc.get("metrics/mAP50B")),
                   ("box mAP50-95", loc.get("metrics/mAP50-95B")),
                   ("char acc", reading.get("reading/char_acc")),
                   ("full-code acc", reading.get("reading/full_code_acc"))]
    title = (f"{args.run}  |  {args.split}: {len(rows)} images, {len(reads)} codes  |  "
             + "  ".join(f"{name} {v:.2f}" for name, v in key_metrics if v is not None))

    lines = [title, "", "== Localisation (Ultralytics) =="]
    lines += [f"{k:28s}: {v:.4f}" for k, v in loc.items()]
    lines += ["", f"== Reading (conf >= {args.conf}, class-agnostic NMS) =="]
    lines += [f"{k:28s}: {v:.4f}" for k, v in reading.items()] or ["(single-class model: no reading)"]
    lines += ["", "== Reads ==", f"{'image':28s} {'layout':10s} {'truth':12s} {'pred':12s} {'pred_fmt':12s} char_acc"]
    lines += [f"{x['image']:28s} {x['layout']:10s} {x['truth']:12s} {x['pred'] or '-':12s} "
              f"{x['pred_fmt'] or '-':12s} {x['char_acc']:.2f}" for x in reads]
    (prefix.parent / f"{prefix.name}.txt").write_text("\n".join(lines) + "\n")
    with open(f"{prefix}_reads.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(reads[0]) if reads else ["image"])
        wr.writeheader()
        wr.writerows(reads)
    print("\n".join(lines))

    from src.visualization.plot_char_dataset import render
    fig = render(args.dataset, args.export, HISTORY, split=args.split, n=args.n,
                 show=not args.no_show, overview_name=f"{prefix.name}.png", per_image=False,
                 predictions=predictions, row_notes=notes, title=title)

    # 4. MLflow: add the numbers and files to the training run.
    import mlflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    found = mlflow.search_runs(experiment_names=[EXPERIMENT],
                               filter_string=f"tags.mlflow.runName = '{args.run}'",
                               order_by=["start_time DESC"], max_results=1)
    if found.empty:
        log.warning("MLflow run '%s' not found; results are in %s*", args.run, prefix)
        return
    with mlflow.start_run(run_id=found.iloc[0]["run_id"]):
        mlflow.log_metrics({f"eval_{args.split}/{k}": v for k, v in {**loc, **reading}.items()})
        for f in (prefix.parent / f"{prefix.name}.txt", Path(f"{prefix}_reads.csv"), fig):
            if f and Path(f).exists():
                mlflow.log_artifact(str(f), artifact_path="history")
    log.info("Evaluation saved to %s* and added to MLflow run '%s'.", prefix, args.run)


if __name__ == "__main__":
    main()
