# Container ID OCR Pipeline

An end-to-end OCR pipeline for reading ISO 6346 shipping container identification codes from images and video streams.

```
Pipeline: Image → [Detection] → [Preprocessing] → [Recognition] → [Post-processing] → [Confidence Gate] → Result
```

---

## Pipeline Stages

| Stage | Method | Key Output |
|-------|--------|-----------|
| **1. Detection** | YOLOv8 / RT-DETR | Cropped plate ROI |
| **2. Preprocessing** | Deskew + CLAHE + Sharpen | Normalised plate image |
| **3. Recognition** | CRNN (CNN+BiLSTM+CTC) | Raw text string |
| **4. Post-processing** | ISO 6346 checksum | Validated container ID |
| **5. Confidence Gate** | Threshold + human flag | Committed or flagged read |

---

## ISO 6346 Format

```
 B I C U  1 2 3 4 5 6  8
 \_____/  \__________/  \__ check digit
 owner+cat   serial
```

- **Owner code**: 3 alpha letters (e.g. `BIC`)
- **Category**: `U` (freight), `J` (detachable), `Z` (trailer)
- **Serial**: 6 digits
- **Check digit**: computed mod-11 over first 10 characters

---

## Quickstart

```bash
# 1. Install dependencies
make install

# 2. Run inference on a single image
make predict IMAGE=data/01_raw/sample.jpg

# 3. Train the CRNN recogniser
make train

# 4. Run the full test suite
make test

# 5. Build & run with Docker
make docker-build
make docker-run
```

---

## Project Structure

```
container_ocr_pipeline/
├── .github/workflows/      # CI (lint + test on every push)
├── configs/                # All hyperparameters and paths in one place
├── data/
│   ├── 01_raw/             # Original images – never modified
│   ├── 02_interim/         # Cropped / deskewed plates
│   └── 03_processed/       # Feature-ready tensors / annotation CSVs
├── models/                 # Saved weights (.pt) and ONNX exports
├── notebooks/              # Exploration notebooks (not production code)
├── src/
│   ├── data/               # Loaders and dataset classes
│   ├── features/           # Preprocessing transforms
│   ├── models/             # Detector, recogniser, full pipeline
│   └── visualization/      # Plotting utilities and dashboards
└── tests/                  # PyTest unit + integration tests
```

---

## Configuration

All tuneable knobs live in `configs/model_config.yaml`. Override any value on the CLI:

```bash
python -m src.models.pipeline --config configs/model_config.yaml \
    recognition.confidence_threshold=0.92
```
