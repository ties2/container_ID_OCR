"""
Stage 1 – Detection
===================
Locates the container ID plate region in a full-frame image using
YOLOv8 or RT-DETR.  Returns cropped ROI(s) sorted by confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """One detected plate region."""
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2  (pixel coords)
    confidence: float
    crop: np.ndarray = field(repr=False)


class PlateDetector:
    """
    Wraps either ultralytics YOLOv8 or RT-DETR behind a unified interface.

    Parameters
    ----------
    cfg : dict
        ``detection`` section of model_config.yaml.
    """

    SUPPORTED_MODELS = {"yolov8", "rtdetr"}

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model_type = cfg["model"].lower()
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model must be one of {self.SUPPORTED_MODELS}")

        self.conf_thr = cfg["confidence_threshold"]
        self.iou_thr  = cfg["iou_threshold"]
        self.imgsz    = tuple(cfg["input_size"])
        self.device   = cfg["device"]
        self.half     = cfg.get("half_precision", False)

        self._model = self._load_model(cfg["weights"])
        logger.info("Detector loaded: %s  device=%s", self.model_type, self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on a BGR image (numpy array).

        Returns a list of :class:`Detection` objects sorted by confidence
        descending.  Returns an empty list when no plate is found.
        """
        results = self._model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf_thr,
            iou=self.iou_thr,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    crop=crop,
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.debug("Detected %d plate(s)", len(detections))
        return detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, weights: str):
        """Load model from weights path or download pretrained backbone."""
        try:
            from ultralytics import YOLO, RTDETR  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Install ultralytics: `pip install ultralytics`"
            ) from exc

        weights_path = Path(weights)
        if not weights_path.exists():
            logger.warning(
                "Weights not found at %s – loading pretrained backbone. "
                "Fine-tune before production use.",
                weights_path,
            )
            weights = "yolov8n.pt" if self.model_type == "yolov8" else "rtdetr-l.pt"

        ModelClass = YOLO if self.model_type == "yolov8" else RTDETR
        return ModelClass(weights)
