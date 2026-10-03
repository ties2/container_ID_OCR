# Container ID Reading: Character Segmentation vs Detection

Reading ISO 6346 shipping-container codes from port-gate camera images.

The repository has two parts:

1. **Research workflow (current focus):** each character of the code is treated as its own
   instance. We compare **per-character instance segmentation (YOLOv8n-seg)** with
   **per-character detection (YOLOv8n)**, trained on exactly the same annotations, and
   measure how well each reads the full code.
2. **OCR pipeline prototype (earlier work):** detection → preprocessing → CRNN → ISO 6346
   check → confidence gate. The code is in `src/models/`, but no trained weights are included.

A full description of the method, data and design decisions is in
[`project_info.md`](project_info.md).

```
CVAT annotations ──► prepare_dataset ──► YOLO-seg / YOLO-det datasets ──► train (MLflow) ──► evaluate
 (code polygon +      (Otsu masks, checks,     (same images, same split)
  box per character)   grouped split)
```

---

## ISO 6346 format

```
 B M O U   6 7 2 5 7 3   7
 \_____/   \_________/   \__ check digit (mod-11 over the first 10 characters)
 owner + category  serial
```

- **Owner code**: 3 letters, **category**: `U` (freight), `J` (detachable), `Z` (trailer)
- **Serial**: 6 digits, **check digit**: 1 digit
- Validation is implemented in `src/models/postprocess.py` and reused by the dataset builder.

---

## Quickstart (research workflow)

```bash
# 1. System dependencies (once)
sudo apt install libgl1 libglib2.0-0

# 2. Create the environment
uv venv --python 3.14.7

# 3. Activate it
source .venv/bin/activate

# 4. Install everything in one go using uv
uv pip install opencv-python numpy matplotlib pytest

```

### The Standard Python 3.12 Path

If you decide to ditch `uv` and go back to the standard Python tools, use this:

```bash
# 1. System dependencies (once)
sudo apt install libgl1 libglib2.0-0
sudo apt install python3.12-venv

# 2. Create the environment
python3.12 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Install everything in one go using standard pip
pip install opencv-python numpy matplotlib pytest
> **A quick tip for PyCharm:** If you activate the environment in your standard terminal (`source .venv/bin/activate`) and then launch PyCharm from that same terminal by typing `pycharm-community` or `pycharm-professional`, PyCharm will often automatically detect and use that `.venv` as the default interpreter for the project.

# 1. Put the data in place
#    data/02_interim/cvat_export/   unzipped CVAT export ("CVAT for images 1.1", with images)
#    data/02_interim/labels.csv     filename,container_id

# 2. Build the datasets and check them
make prepare          # -> data/03_processed/char_dataset/
pytest tests/test_prepare_dataset.py

make plot-data        # -> results/figures/char_samples/ (image | ground truth | masks)
python3 -m src.visualization.plot_char_dataset --show
python3 -m src.visualization.plot_char_dataset --split test --weights results/yolo/seg-s42/weights/best.pt

# 3. Train with MLflow tracking (installs PyTorch + Ultralytics + MLflow once)
make train-setup
make train-seg SEED=42 EPOCHS=3     # short timing test first
make train-seg SEED=42              # segmentation
make train-det SEED=42              # detection, same data and settings
make train-all                      # both models x 3 seeds (42, 7, 123)

# 4. Watch training
make mlflow-ui                      # http://127.0.0.1:5000

# 5. Tests
pytest tests/
```

Useful variables: `DEVICE=0` (GPU), `IMGSZ=1280`, `EPOCHS=300`, `BATCH=4`, `SEED=42`,
`SINGLE_CLS=1` (all characters as one class: localisation only).
MLflow uses an SQLite store (`results/mlflow.db`); artifacts go to `results/mlflow-artifacts/`.

To show the figures interactively (e.g. in PyCharm's Plots window):

```bash
python3 -m src.visualization.plot_char_dataset --show
python3 -m src.visualization.plot_char_dataset --split test --show \
    --weights results/yolo/seg-s42/weights/best.pt      # adds a prediction panel
```

---

## Outputs

| Path | Content |
|---|---|
| `data/03_processed/char_dataset/yolo_seg/` | images + mask polygons, `data.yaml` (36 classes: 0–9, A–Z) |
| `data/03_processed/char_dataset/yolo_det/` | the same images + boxes derived from the same masks |
| `data/03_processed/char_dataset/report.csv` | per image: `OK` / `REVIEW` and the reason |
| `data/03_processed/char_dataset/mask_flags.csv` | masks that look suspicious (check by eye) |
| `data/03_processed/char_dataset/split.csv` | split per image (grouped by container ID) |
| `data/03_processed/char_dataset/dataset_info.json`, `class_counts.csv` | images / containers / characters per split, instances per class |
| `data/02_interim/split_assignments.csv` | **persistent** container → split table: new annotations never move old containers |
| `data/03_processed/char_dataset/review/` | overlays of every image with its masks |
| `results/figures/char_samples/` | rows of image / ground truth / masks (/ prediction) |
| `results/yolo/<task>-s<seed>/` | Ultralytics run: weights, curves, confusion matrix |
| `results/history/<stamp>_<run>.*` | one record per run: `.txt` summary, `_curves.png`, `_epochs.csv`, `_test_predictions.png`, `_console.log` |
| `results/mlflow.db`, `results/mlflow-artifacts/` | MLflow store: losses and metrics per epoch, test metrics, weights |

---

## Project structure

```
.
├── configs/model_config.yaml       # settings of the OCR pipeline prototype
├── data/
│   ├── 01_raw/                     # original images, never modified
│   ├── 02_interim/                 # CVAT export + labels.csv
│   └── 03_processed/char_dataset/  # generated YOLO datasets and reports
├── notebooks/                      # data exploration
├── results/                        # training runs, figures, MLflow store
├── src/
│   ├── data/
│   │   ├── dataset.py              # CRNN dataset (prototype)
│   │   └── prepare_dataset.py      # CVAT -> masks -> checks -> grouped split -> YOLO
│   ├── features/preprocess.py      # deskew / CLAHE / sharpen (prototype)
│   ├── models/
│   │   ├── train_yolo.py           # train seg or det, MLflow logging, test evaluation
│   │   ├── detector.py, recognizer.py, pipeline.py   # OCR pipeline (prototype)
│   │   └── postprocess.py          # ISO 6346 validation + confidence gate
│   └── visualization/
│       ├── plot_char_dataset.py    # image | ground truth | masks (| prediction)
│       └── plots.py                # plots for the prototype pipeline
├── tests/                          # pytest
├── Makefile
└── project_info.md                 # full description for reviewers
```

---

## Data and licence

Images: *container-id-number* dataset by kerofox (Roboflow Universe, CC BY 4.0), also on Kaggle
as "Container Number Recognition". The character annotations (polygons, boxes, text) are our own.
Data files are not tracked in git.