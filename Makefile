# ============================================================
# Container ID OCR Pipeline – Makefile
# ============================================================
.PHONY: install install-dev prepare plot-data train-setup mlflow-ui train-seg train-det train-all predict test lint format docker-build docker-run clean

# bash + pipefail: a training error still stops make when output is piped to `tee`
SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c

# ── Variables ────────────────────────────────────────────────
CONFIG  ?= configs/model_config.yaml
IMAGE   ?= data/01_raw/sample.jpg
VIDEO   ?= data/01_raw/sample.mp4
# DEVICE: cpu | 0 (first GPU) | mps (Apple)
DEVICE  ?= cpu

# Character dataset (built by `make prepare`) and training settings
CHARSET  ?= data/03_processed/char_dataset
EPOCHS   ?= 300
# characters are small: 640 px loses detail
IMGSZ    ?= 1280
BATCH    ?= 4
SEED     ?= 42
# MLflow store (MLflow 3 no longer accepts a plain folder)
MLFLOW_URI ?= sqlite:///results/mlflow.db
# SINGLE_CLS=1: all characters as one class (localisation only)
SINGLE_CLS ?= 0
# one time stamp per make call; links the history files of a run
STAMP    := $(shell date +%Y%m%d-%H%M%S)
TRAIN_ARGS = --seed $(SEED) --epochs $(EPOCHS) --imgsz $(IMGSZ) --batch $(BATCH) \
             --device $(DEVICE) --dataset $(CHARSET) --mlflow-uri $(MLFLOW_URI) --stamp $(STAMP) \
             $(if $(filter 1,$(SINGLE_CLS)),--single-cls,)
RUN_TAG  = $(if $(filter 1,$(SINGLE_CLS)),-1cls,)-s$(SEED)

# ── Environment ──────────────────────────────────────────────

install:
	pip install -e ".[train]"

install-dev:
	pip install -e ".[train,vlm,dev]"
	pre-commit install

# ── Pipeline ─────────────────────────────────────────────────

## Run inference on a single image
predict:
	python -m src.models.pipeline \
		--config $(CONFIG) \
		--image  $(IMAGE)  \
		--output results/prediction.json

## Run inference on a video file
predict-video:
	python -m src.models.pipeline \
		--config $(CONFIG) \
		--video  $(VIDEO)  \
		--output results/video_reads.json

# ── Character segmentation vs detection ──────────────────────

## Build YOLO datasets from the CVAT export (see src/data/prepare_dataset.py)
prepare:
	python3 -m src.data.prepare_dataset

## Plot rows of: image | ground truth | masks  (results/figures/char_samples)
plot-data:
	python3 -m src.visualization.plot_char_dataset --n 6

## One-time: install the training stack (PyTorch, Ultralytics, MLflow)
train-setup:
	pip install ultralytics mlflow

## Open the MLflow dashboard at http://127.0.0.1:5000
mlflow-ui:
	mkdir -p results
	mlflow server --backend-store-uri $(MLFLOW_URI) --port 5000

## Train per-character instance segmentation (YOLOv8n-seg), logged to MLflow
train-seg:
	mkdir -p results/history
	python3 -m src.models.train_yolo --task seg $(TRAIN_ARGS) 2>&1 \
		| tee results/history/$(STAMP)_seg$(RUN_TAG)_console.log

## Train per-character detection on the same annotations (YOLOv8n), logged to MLflow
train-det:
	mkdir -p results/history
	python3 -m src.models.train_yolo --task det $(TRAIN_ARGS) 2>&1 \
		| tee results/history/$(STAMP)_det$(RUN_TAG)_console.log

## Both models with three seeds (6 runs)
train-all:
	for s in 42 7 123; do \
		$(MAKE) train-seg SEED=$$s && $(MAKE) train-det SEED=$$s || exit 1; \
	done

# ── Quality ───────────────────────────────────────────────────

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

# ── Docker ───────────────────────────────────────────────────

docker-build:
	docker build -t container-ocr:latest .

docker-run:
	docker run --rm --gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/results:/app/results \
		container-ocr:latest \
		python -m src.models.pipeline \
			--config configs/model_config.yaml \
			--image  data/01_raw/sample.jpg

# ── Housekeeping ─────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache dist build *.egg-info