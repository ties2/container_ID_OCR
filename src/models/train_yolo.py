"""
src/models/train_yolo.py
========================
Train one per-character model (segmentation or detection) on the dataset built
by src/data/prepare_dataset.py, with MLflow tracking.

Both tasks use identical settings, so the only difference between a `seg` run
and a `det` run is the label type (masks vs boxes of the same characters).

During training, Ultralytics' MLflow integration logs per epoch:
    train/box_loss, train/cls_loss, train/dfl_loss (+ train/seg_loss for seg)
    val/... losses, metrics/precisionB, metrics/recallB, metrics/mAP50B, metrics/mAP50-95B
    (+ the same metrics with suffix M for masks), and the learning rates.
After training, this script evaluates best.pt on the TEST split and adds those
numbers to the same MLflow run with the prefix `test/`.

Usage (from the project root, inside the .venv):
    python -m src.models.train_yolo --task seg --seed 42
    python -m src.models.train_yolo --task det --seed 42 --epochs 3     # quick timing test
    make train-seg SEED=42                                              # same via make

View the runs:  make mlflow-ui   ->  http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

log = logging.getLogger("train_yolo")

PRETRAINED = {"seg": "yolov8n-seg.pt", "det": "yolov8n.pt"}
EXPERIMENT = "char-seg-vs-det"


def _clean(metrics: dict) -> dict[str, float]:
    """MLflow metric names may not contain parentheses."""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in metrics.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["seg", "det"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="characters are small; 640 loses detail")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu | 0 (first GPU) | mps")
    ap.add_argument("--dataset", type=Path, default=Path("data/03_processed/char_dataset"))
    ap.add_argument("--mlflow-uri", default="sqlite:///results/mlflow.db",
                    help="MLflow tracking URI (MLflow 3 needs a database, e.g. SQLite)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_name = f"{args.task}-s{args.seed}"
    # Absolute path: Ultralytics puts a *relative* project under its global runs_dir
    # (~/.config/Ultralytics/settings.json), which may point to another project.
    project = Path("results/yolo").resolve()
    data_yaml = args.dataset / f"yolo_{args.task}" / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found - run `make prepare` first.")

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

    model = YOLO(PRETRAINED[args.task])
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        fliplr=0.0,              # a mirrored letter is not the same letter
        project=str(project),
        name=run_name,
        exist_ok=True,
    )

    # Evaluate the best checkpoint on the held-out test split.
    best = Path(model.trainer.best)          # where this run really saved its weights
    log.info("Best checkpoint: %s", best)
    test = YOLO(str(best)).val(data=str(data_yaml), split="test", imgsz=args.imgsz,
                               batch=args.batch, device=args.device,
                               project=str(project), name=f"{run_name}-test", exist_ok=True)
    test_metrics = {f"test/{k}": v for k, v in _clean(test.results_dict).items()}
    for k, v in test_metrics.items():
        log.info("%-35s %.4f", k, v)

    # Attach the test numbers to the MLflow run that the training just created.
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT],
                              filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        log.warning("MLflow run '%s' not found - test metrics only printed above.", run_name)
        return
    with mlflow.start_run(run_id=runs.iloc[0]["run_id"]):
        mlflow.log_metrics(test_metrics)
    log.info("Test metrics added to MLflow run '%s'.", run_name)


if __name__ == "__main__":
    main()