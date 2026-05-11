# ============================================================
# Container ID OCR Pipeline – Makefile
# ============================================================
.PHONY: install install-dev train predict test lint format docker-build docker-run clean

# ── Variables ────────────────────────────────────────────────
CONFIG  ?= configs/model_config.yaml
IMAGE   ?= data/01_raw/sample.jpg
VIDEO   ?= data/01_raw/sample.mp4
DEVICE  ?= cuda

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

## Train the CRNN recogniser
train:
	python -m src.models.train \
		--config $(CONFIG) \
		--device $(DEVICE)

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
