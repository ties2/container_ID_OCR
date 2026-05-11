"""
src/data – Dataset classes and loading utilities
================================================
Covers:
  - ContainerPlateDataset   : PyTorch Dataset for CRNN training
  - AnnotationLoader        : reads YOLO-style or custom JSON annotations
  - split_dataset           : train / val split utility
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Annotation schema
# ──────────────────────────────────────────────────────────────────────

class Annotation:
    """One labelled plate crop."""
    __slots__ = ("image_path", "container_id", "bbox")

    def __init__(
        self,
        image_path: str,
        container_id: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self.image_path   = image_path
        self.container_id = container_id.upper().replace(" ", "")
        self.bbox         = bbox   # (x1, y1, x2, y2) or None for pre-cropped images


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class ContainerPlateDataset:
    """
    PyTorch-compatible dataset (no torch dependency at import time).

    Each sample returns ``(image_array, label_str)`` where image_array
    is a (1, H, W) float32 numpy array ready for the CRNN.

    Parameters
    ----------
    annotations : list[Annotation]
    preprocessor : callable
        Typically ``PlatePreprocessor.process``.
    augment : bool
        Whether to apply random augmentation (training only).
    """

    def __init__(
        self,
        annotations: List[Annotation],
        preprocessor: Callable,
        augment: bool = False,
    ) -> None:
        self.annotations  = annotations
        self.preprocessor = preprocessor
        self.augment      = augment

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        ann = self.annotations[idx]
        img = cv2.imread(ann.image_path)
        if img is None:
            raise FileNotFoundError(ann.image_path)

        # Crop if a bounding box is provided (full-frame annotation)
        if ann.bbox is not None:
            x1, y1, x2, y2 = ann.bbox
            img = img[y1:y2, x1:x2]

        if self.augment:
            img = self._augment(img)

        tensor = self.preprocessor(img)
        return tensor, ann.container_id

    # ------------------------------------------------------------------

    @staticmethod
    def _augment(img: np.ndarray) -> np.ndarray:
        """Light augmentation to improve generalisation."""
        # Random brightness / contrast
        alpha = np.random.uniform(0.7, 1.3)   # contrast
        beta  = np.random.randint(-30, 30)     # brightness
        img   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Random perspective warp
        h, w  = img.shape[:2]
        margin = int(min(h, w) * 0.05)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.random.randint(-margin, margin, src.shape).astype(np.float32)
        M   = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Gaussian noise
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img


# ──────────────────────────────────────────────────────────────────────
# Annotation loaders
# ──────────────────────────────────────────────────────────────────────

def load_annotations_csv(csv_path: str) -> List[Annotation]:
    """
    Load annotations from a CSV with columns:
        image_path, container_id[, x1, y1, x2, y2]
    """
    annotations = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bbox = None
            if all(k in row for k in ("x1", "y1", "x2", "y2")):
                bbox = (int(row["x1"]), int(row["y1"]),
                        int(row["x2"]), int(row["y2"]))
            annotations.append(
                Annotation(row["image_path"], row["container_id"], bbox)
            )
    logger.info("Loaded %d annotations from %s", len(annotations), csv_path)
    return annotations


def load_annotations_json(json_path: str) -> List[Annotation]:
    """
    Load annotations from a JSON list:
    [{"image_path": "...", "container_id": "...", "bbox": [x1,y1,x2,y2]}, ...]
    """
    with open(json_path) as f:
        records = json.load(f)
    annotations = [
        Annotation(
            r["image_path"],
            r["container_id"],
            tuple(r["bbox"]) if "bbox" in r else None,
        )
        for r in records
    ]
    logger.info("Loaded %d annotations from %s", len(annotations), json_path)
    return annotations


def split_dataset(
    annotations: List[Annotation],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Annotation], List[Annotation]]:
    """Deterministic train/val split."""
    import random
    rng = random.Random(seed)
    shuffled = annotations[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]
