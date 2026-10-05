"""
src/models/train_yolo.py
========================
Train one per-character model (segmentation or detection) on the dataset built
by src/data/prepare_dataset.py, with MLflow tracking and a saved run history.

Both tasks use identical settings, so the only difference between a `seg` run
and a `det` run is the label type (masks vs boxes of the same characters).

During training, Ultralytics' MLflow integration logs per epoch:
    train/box_loss, train/cls_loss, train/dfl_loss (+ train/seg_loss for seg)
    val/... losses, metrics/precisionB, metrics/recallB, metrics/mAP50B, metrics/mAP50-95B
    (+ the same metrics with suffix M for masks), and the learning rates.

After training, this script writes a run history to results/history/, all files
with the same prefix <stamp>_<run>:
    <prefix>.txt            settings, dataset sizes, best and final epoch, diagnosis
    <prefix>_curves.png     loss / metric curves per epoch (Ultralytics)
    <prefix>_epochs.csv     all numbers per epoch
    <prefix>_console.log    full console output (when started via make)
The test set is NOT touched here. Evaluate afterwards with src/models/evaluate.py
(validation set while developing, test set once at the end).

Usage (from the project root, inside the .venv):
    python -m src.models.train_yolo --task seg --seed 42
    python -m src.models.train_yolo --task det --seed 42 --epochs 3       # quick timing test
    python -m src.models.train_yolo --task seg --single-cls                # localisation only
    python -m src.models.train_yolo --task seg --arch yolo11n              # another model family
    make train-seg SEED=42                                                # same via make

Then:  make evaluate RUN=seg-s42          (figure + reading metrics on validation)
       make mlflow-ui                     (http://127.0.0.1:5000)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("train_yolo")

# Model family/size. The same architecture is always used for seg and det, so a
# seg/det pair differs only in the head: e.g. yolov8n-seg.pt vs yolov8n.pt.
ARCHS = ("yolov8n", "yolo11n", "yolov8s", "yolo11s")


def pretrained_weights(arch: str, task: str) -> str:
    return f"{arch}-seg.pt" if task == "seg" else f"{arch}.pt"
EXPERIMENT = "char-seg-vs-det"
HISTORY = Path("results/history")


def _clean(metrics: dict) -> dict[str, float]:
    """MLflow metric names may not contain parentheses."""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in metrics.items()}


def read_epochs(results_csv: Path) -> list[dict[str, float]]:
    """Ultralytics results.csv -> one dict per epoch (column names stripped)."""
    with open(results_csv, newline="") as fh:
        return [{k.strip(): float(v) for k, v in row.items() if v.strip()} for row in csv.DictReader(fh)]


def fitness(row: dict[str, float], task: str) -> float:
    """Ultralytics' checkpoint criterion: 0.1*mAP50 + 0.9*mAP50-95 (box, + mask for seg)."""
    f = 0.1 * row.get("metrics/mAP50(B)", 0) + 0.9 * row.get("metrics/mAP50-95(B)", 0)
    if task == "seg":
        f += 0.1 * row.get("metrics/mAP50(M)", 0) + 0.9 * row.get("metrics/mAP50-95(M)", 0)
    return f


def diagnose(epochs: list[dict[str, float]]) -> list[str]:
    """Simple, explainable checks on the loss curves (a heuristic, not a proof)."""
    notes = []
    first, last = epochs[0], epochs[-1]
    t0, t1 = first.get("train/cls_loss"), last.get("train/cls_loss")
    vals = [e["val/cls_loss"] for e in epochs if "val/cls_loss" in e]
    if t0 and t1:
        notes.append(f"train/cls_loss {t0:.3f} -> {t1:.3f} ({100 * (t1 - t0) / t0:+.0f}%)")
        if t1 > 0.8 * t0:
            notes.append("UNDERFITTING: the class loss barely went down; the model has not "
                         "learned to tell characters apart (too few examples per class?).")
    if vals:
        best_i = min(range(len(vals)), key=vals.__getitem__)
        notes.append(f"val/cls_loss lowest {vals[best_i]:.3f} at epoch {best_i + 1}, final {vals[-1]:.3f}")
        if vals[-1] > 1.2 * vals[best_i] and t1 and t0 and t1 < t0:
            notes.append("POSSIBLE OVERFITTING: training loss keeps falling while the validation "
                         "loss rose >20% above its minimum.")
    return notes


def write_history(prefix: Path, args: argparse.Namespace, run_name: str, run_id: str | None,
                  save_dir: Path, best: Path, dataset_info: dict, epochs: list[dict[str, float]],
                  minutes: float) -> Path:
    best_row = max(epochs, key=lambda r: fitness(r, args.task))
    lines = [
        f"Run            : {run_name}",
        f"Stamp          : {args.stamp}",
        f"MLflow run id  : {run_id or '-'}",
        f"Duration       : {minutes:.1f} min",
        f"Weights        : {best}",
        f"Ultralytics dir: {save_dir}",
        "",
        "== Settings ==",
        *(f"{k:15s}: {v}" for k, v in sorted(vars(args).items())),
        "",
        "== Dataset ==",
        *(f"{k:20s}: {v}" for k, v in dataset_info.items()),
        "",
        f"== Best epoch (checkpoint): {int(best_row.get('epoch', 0))} of {len(epochs)} ==",
        *(f"{k:28s}: {v:.4f}" for k, v in best_row.items() if k != "epoch" and not k.startswith("lr/")),
        "",
        f"== Final epoch: {len(epochs)} ==",
        *(f"{k:28s}: {v:.4f}" for k, v in epochs[-1].items() if k != "epoch" and not k.startswith("lr/")),
        "",
        "== Diagnosis (heuristic) ==",
        *diagnose(epochs),
        "",
    ]
    path = prefix.with_suffix(".txt")
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["seg", "det"], required=True)
    ap.add_argument("--arch", choices=ARCHS, default="yolov8n",
                    help="model family and size (n = nano, s = small)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1280, help="characters are small; 640 loses detail")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu | 0 (first GPU) | mps")
    ap.add_argument("--single-cls", action="store_true",
                    help="treat all 36 characters as one class (localisation only)")
    ap.add_argument("--dataset", type=Path, default=Path("data/03_processed/char_dataset"))
    ap.add_argument("--mlflow-uri", default="sqlite:///results/mlflow.db",
                    help="MLflow tracking URI (MLflow 3 needs a database, e.g. SQLite)")
    ap.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d-%H%M%S"),
                    help="prefix that links all history files of this run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # e.g. seg-s42, det-1cls-s7, seg-yolo11n-s42 (the default yolov8n is not written)
    arch_tag = "" if args.arch == "yolov8n" else f"-{args.arch}"
    run_name = f"{args.task}{arch_tag}{'-1cls' if args.single_cls else ''}-s{args.seed}"
    # Absolute path: Ultralytics puts a *relative* project under its global runs_dir
    # (~/.config/Ultralytics/settings.json), which may point to another project.
    project = Path("results/yolo").resolve()
    data_yaml = args.dataset / f"yolo_{args.task}" / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found - run `make prepare` first.")
    info_file = args.dataset / "dataset_info.json"
    dataset_info = json.loads(info_file.read_text()) if info_file.exists() else {}

    # MLflow settings are read by Ultralytics' callback from the environment.
    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT
    os.environ["MLFLOW_RUN"] = run_name

    import mlflow
    from ultralytics import YOLO, settings

    # Create the experiment once, with artifacts (weights, plots) under results/.
    Path("results").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(args.mlflow_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT,
                                 artifact_location=Path("results/mlflow-artifacts").resolve().as_uri())
    settings.update({"mlflow": True})

    t_start = time.time()
    model = YOLO(pretrained_weights(args.arch, args.task))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        single_cls=args.single_cls,
        fliplr=0.0,              # a mirrored letter is not the same letter
        project=str(project),
        name=run_name,
        exist_ok=True,
    )
    minutes = (time.time() - t_start) / 60
    save_dir = Path(model.trainer.save_dir)
    best = Path(model.trainer.best)          # where this run really saved its weights
    log.info("Best checkpoint: %s", best)

    # Run history: text summary + curves, all with the same prefix.
    HISTORY.mkdir(parents=True, exist_ok=True)
    prefix = HISTORY / f"{args.stamp}_{run_name}"
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT],
                              filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["start_time DESC"], max_results=1)
    run_id = None if runs.empty else runs.iloc[0]["run_id"]
    epochs = read_epochs(save_dir / "results.csv")
    files = [write_history(prefix, args, run_name, run_id, save_dir, best, dataset_info, epochs, minutes)]
    for src, suffix in ((save_dir / "results.png", "_curves.png"), (save_dir / "results.csv", "_epochs.csv")):
        if src.exists():
            files.append(Path(shutil.copy(src, f"{prefix}{suffix}")))
    log.info("History written: %s*", prefix)

    # Attach dataset sizes and history files to the MLflow run.
    if run_id is not None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_params({f"data/{k}": v for k, v in dataset_info.items()} | {"stamp": args.stamp})
            for f in files:
                mlflow.log_artifact(str(f), artifact_path="history")
    log.info("Next: make evaluate RUN=%s   (validation figure + reading metrics)", run_name)


if __name__ == "__main__":
    main()