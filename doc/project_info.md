# Project Information: Character Segmentation vs Detection for Container-ID Reading

This document explains the whole project step by step, so that a reviewer can follow
every decision: the problem, the data, the annotation, how masks are made, how the data
is split, how the models are trained and how they are evaluated. Sections marked
**Status** say what is done and what is still planned.

Course context: Computer Vision theme week "Segmentation and Object Detection"
(NHL Stenden, MSc Computer Vision & Data Science). Deliverable: a two-page IMRaD
extended abstract.

---

## 1. Summary

Shipping containers carry an 11-character ID code (ISO 6346). We read this code from
port-gate camera images by treating **every character as its own instance**: the model
finds each letter and digit, and the code is read by ordering the characters along the
code's direction. This needs no separate OCR model and handles vertical codes without
special treatment.

The research question is whether **instance segmentation** reads the code better than
**object detection** when both are trained on exactly the same annotations.
We build our own annotated dataset (CVAT), generate pixel masks semi-automatically,
split the data by container to prevent leakage, train YOLOv8n-seg and YOLOv8n with
identical settings and track all runs in MLflow.

---

## 2. Problem

### 2.1 The code

```
 B M O U   6 7 2 5 7 3   7
 owner(3) + category(1)   serial(6)   check digit(1)
```

The check digit is computed from the first 10 characters (letters have fixed values,
weights are powers of 2, result mod 11, 10 → 0). It lets us validate a read without
ground truth, and it catches typing errors in our own labels.

### 2.2 Why it is hard in this data

- Codes are often **vertical** (characters stacked, each upright) as well as horizontal.
- Cameras look at the container **at an angle**, so code columns are tilted.
- Characters are **white on dark** or **dark on light**, on corrugated, rusty surfaces.
- The code sits **next to other markings**, e.g. the size/type code (`22G1`).
- The check digit is often printed inside a **frame**.
- One image may show **the same code twice** (on the door and on the roof edge).

### 2.3 How the approach evolved (and why)

1. First idea: segment the **whole code region** with one polygon and read it with OCR.
   The lecturer pointed out that a 4-point polygon around a narrow code column is
   essentially a rotated bounding box, so segmentation adds little there.
2. Following his suggestion, the project moved to **one mask per character**. Here the
   mask follows the shape of each letter, which a box cannot do. This also removes the
   separate OCR step.

---

## 3. Research question and design

**Research question:** *Does per-character instance segmentation read container codes
more accurately than per-character detection trained on the same annotations?*

**Design (one variable changes):**

| | Segmentation | Detection |
|---|---|---|
| Model | YOLOv8n-seg (3.4 M params) | YOLOv8n (3.2 M params) |
| Labels | character masks | bounding boxes **of the same masks** |
| Images, split, classes | identical | identical |
| Training settings | identical | identical |

Both models predict a class per instance (36 classes: `0–9`, `A–Z`), so both can read the
code directly. Whether the extra mask supervision helps reading is an open question.
The experiment answers it, rather than assuming it.

---

## 4. Data

### 4.1 Source

*container-id-number* by kerofox (Roboflow Universe, CC BY 4.0), mirrored on Kaggle.
1,107 images (1920×1080) from five gate cameras (`AH-A01`, `AS-B01`, `LB-C02`,
`LF-C01`, `RF-D01`), split 775 / 221 / 111. Labels: bounding boxes, one class
(`container-numbers`). There are no text transcriptions.

### 4.2 Data exploration findings

| Finding | Evidence | Consequence |
|---|---|---|
| Filenames encode a capture timestamp shared by all cameras (`1-123603001-OCR-RF-D01`) | Same code visible in `RF` and `LF` images with the same number (checked visually) | One container appears in up to 5 images (503 capture events) |
| The original random split leaks | 73% of test images (81/111) and 71% of validation images (158/221) share a capture event with a training image | We do not use the original split; we split by container |
| Original labels mark the type code as well | Label files contain a second, shorter box next to the ID (height ratio ≈ 4/11) | Our annotation covers the 11-character ID only |
| All images come from one gate within a few hours of one afternoon | Timestamps in filenames (12:xx–15:xx) | Lighting varies little; reported as a limitation |

---

## 5. Own annotated dataset

### 5.1 Selection

Images are selected for variation, not at random. Variation axes: camera (viewpoint),
layout (vertical / horizontal), angle (straight / tilted), surface (clean / rust, glare,
dirt). **Status:** 30 images from camera `AH` are annotated. Target: 30–40 images
spread over the cameras.

### 5.2 Annotation protocol (CVAT, "CVAT for images 1.1")

| Label | Shape | Rule |
|---|---|---|
| `container_id` | polygon | around each complete, readable copy of the 11-character ID; type code and other markings excluded |
| `char` | rectangle | one per character, drawn around the letter; for the check digit around the digit or its frame |

- Every visible copy of the code is annotated (instance segmentation: several instances per image).
- The true code is typed once per image in `labels.csv` and validated with the check digit.
- Shapes are drawn as *Shape*, not *Track* (a track would copy the object to other frames).

### 5.3 From rectangles to pixel masks (semi-automatic)

Drawing a polygon around every letter by hand is slow, and no SAM model was used. Because characters are high-contrast paint on metal, the mask inside each
rectangle is extracted with classical image processing and then checked by eye:

1. **Pad** the rectangle by 15% (so a tight box does not cut the letter). For the check
   digit, no padding.
2. **Otsu threshold** on the grey crop: splits it into "character" and "background"
   without a manual threshold.
3. **Polarity**: decide whether the characters are bright or dark by looking at a ring
   of container surface *around* the box. This stays correct when the box lies on a
   printed frame.
4. **Frame removal (check digit only)**: a component that spans most of the crop, has a
   rectangular outline and a thin line (thickness < 12% of its size) is a frame. Its line
   thickness is measured in the middle of each side and everything within 1.5× that
   thickness of the edge is erased. A bold unframed `0` is not removed (thick stroke).
5. **Connected components**: drop pieces touching the crop border (parts of neighbouring
   letters), pieces whose centre lies outside the drawn box, and pieces that contain
   another piece. Keep every remaining piece with ≥ 10% of the largest area (stencil
   fonts split letters into parts).
6. **Outline**: the outer contour becomes the polygon. Several parts are joined with a
   convex hull.

**Quality flags** (written to `mask_flags.csv`, shown in red in the figures): mask
covers < 8% or > 85% of its box, letter made of several parts, nothing found.

**Known limitation:** the YOLO polygon format cannot store holes, so letters like
`0`, `4`, `A`, `D` are filled. Segmentation is therefore of the character's outer shape.

### 5.4 Class assignment and checks

- Each `char` box is assigned to the `container_id` polygon that contains its centre.
- Boxes are ordered along the **principal axis** of their centres: a mostly vertical axis is
  read top to bottom, a horizontal one left to right. This works for tilted codes and does
  not depend on the order of drawing.
- Box *i* gets character *i* of the typed code.
- An image is marked **REVIEW** and left out of the dataset if: a code does not have exactly
  11 boxes (otherwise all following labels would shift), a box lies outside every code,
  the typed code fails the check digit, the image is missing, or a mask could not be made.

### 5.5 Split

Train / validation / test = 60 / 20 / 20, **grouped by container ID**: all images of one
container go to the same split, whichever camera they come from. The original Kaggle
train/valid/test folders are ignored (they leak, Section 4.2).

The assignment is **persistent** (`data/02_interim/split_assignments.csv`): when new images
are annotated, containers that already have a split keep it, and each new container goes
to the split that is furthest below its target share. So adding data never moves a test
container into training, and runs on different dataset versions stay comparable.
The detection dataset uses the same images and the same split. Outputs: `split.csv`,
`dataset_info.json` (sizes per split), `class_counts.csv` (instances per class and split).

### 5.6 Visual check

`plot_char_dataset.py` draws one row per image: **image | ground truth (CVAT) | masks**.
Every image is checked before training.

---

## 6. Models and training

| Setting | Value | Reason |
|---|---|---|
| Models | YOLOv8n-seg, YOLOv8n | smallest versions: few parameters for a small dataset; same family, so only the head differs |
| Initialisation | COCO-pretrained weights | small dataset; COCO has no character classes, so no label leakage |
| Image size | 1280 | characters are 30–70 px high in 1920×1080 images; 640 loses detail |
| Epochs / batch | 300 / 4 | few images give only ~3 iterations per epoch; best checkpoint chosen on validation, test never used for choices |
| Horizontal flip | **off** (`fliplr=0`) | a mirrored letter is not the same letter |
| Other augmentation | Ultralytics defaults (mosaic, HSV, scale, translate) | identical for both models |
| Seeds | 42, 7, 123, `deterministic=True` | the test set is small; results are reported as mean ± std |

Models not used, and why: larger YOLO variants (overfit on ~20 training images),
Mask R-CNN (different framework and training pipeline, so differences would no longer
come from segmentation vs detection alone), RT-DETR (detection only; transformers need
much more data).

---

## 7. Evaluation

### 7.1 Localisation (reported by Ultralytics, on the test split)

- Detection: precision, recall, mAP@50, mAP@50–95 for boxes (suffix `B`).
- Segmentation: the same for masks (suffix `M`), plus its own box metrics.

Note: mask metrics are measured against the Otsu masks, which are semi-automatic
ground truth (see Section 9).

### 7.2 Reading (the main comparison) — **Status: planned**

Predictions are grouped per code, ordered along the code's axis and compared with the
typed code:

- **Character accuracy**: share of the 11 positions read correctly.
- **Full-code accuracy**: share of codes with all 11 characters correct.
- **Check-digit validity**: share of reads that pass ISO 6346.

These metrics do not depend on the Otsu masks; they use the typed text only. Results
are reported per model as mean ± std over the three seeds, and separately for vertical
and horizontal codes.

---

## 8. Experiment tracking (MLflow)

Ultralytics' MLflow integration logs per epoch: training losses (`train/box_loss`,
`train/cls_loss`, `train/dfl_loss`, and `train/seg_loss` for segmentation), validation
losses, precision, recall, mAP and learning rates. `train_yolo.py` evaluates the best
checkpoint on the test split afterwards and adds those numbers to the same run with the
prefix `test/`. Runs are named `seg-s42`, `det-s42`, … in the experiment
`char-seg-vs-det`.

Every run also leaves a record in `results/history/`, all files sharing the prefix
`<stamp>_<run>`: a text summary (settings, dataset sizes, best and final epoch, test
metrics, a heuristic under/overfitting diagnosis), the loss/metric curves, the per-epoch
numbers, the test-set predictions figure and the full console log.

**Reading the curves.** Underfitting: training and validation losses both stay high
(the model has not learned). Overfitting: training loss keeps falling while validation
loss rises again and validation mAP drops. Decisions (epochs, settings) are made on the
validation set only; the test set is looked at once, for the final numbers.

**First result (30 annotated images, AH camera, 36 classes).** Localisation works
(segmentation recall ≈ 0.6–0.75), but `cls_loss` stays near 4 for training and validation:
the model cannot yet tell characters apart because most classes have only a few examples.
This is underfitting caused by too little data per class, not a bug.

---

## 9. Limitations and threats to validity

- **Small dataset**: about 30–40 annotated images and a test set of 6–8 images. Mitigated
  by three seeds; conclusions are stated with caution.
- **One gate, one afternoon**: little variation in lighting and location.
- **Semi-automatic masks**: mask metrics partly measure agreement with Otsu. The reading
  metrics use typed text only and are not affected.
- **Filled holes** in masks (format limitation).
- **Class imbalance**: digits are frequent; most letters (except `U`) are rare.
- **Selection**: images were chosen for variation, so the test set is not a random sample
  of gate traffic.

---

## 10. Reproducing the results

```bash
source .venv/bin/activate
make prepare            # CVAT export + labels.csv -> datasets, reports, overlays
make plot-data          # visual check
make train-setup        # once
make train-all          # 2 models x 3 seeds, logged to MLflow
make mlflow-ui          # compare runs
pytest tests/
```

---

## 11. Code map

| File | Responsibility |
|---|---|
| `src/data/prepare_dataset.py` | parse CVAT, assign and order characters, checks, Otsu masks, grouped split, YOLO datasets, reports |
| `src/models/train_yolo.py` | train one model (seg or det), MLflow, test evaluation |
| `src/visualization/plot_char_dataset.py` | image / ground truth / masks / prediction figures |
| `src/models/postprocess.py` | ISO 6346 validation (shared with the OCR prototype) |
| `tests/test_prepare_dataset.py` | filenames, check digit, reading order, split, mask extraction, frame removal |

---

## 12. Mapping to the course norms

| Norm | Where it is addressed |
|---|---|
| 67: challenge and well-reasoned choice of detection or segmentation | Sections 2–3: per-character segmentation chosen, detection trained as the controlled comparison |
| 68: diverse, representative dataset, appropriate format, separate train/val/test | Sections 4–5: own annotations, variation axes, leakage analysis, grouped split, YOLO format |
| 69: IMRaD extended abstract, max. two pages, with figures and tables | Abstract built from Sections 2–9; figures from `plot_char_dataset.py`, tables from MLflow |

---

## 13. References

- ISO 6346:2022, Freight containers: coding, identification and marking.
- kerofox, *container-id-number* dataset, Roboflow Universe, CC BY 4.0.
- G. Jocher, A. Chaurasia, J. Qiu, *Ultralytics YOLOv8*, 2023.
- CVAT.ai Corporation, *CVAT: Computer Vision Annotation Tool*.
- N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. SMC*, 1979.