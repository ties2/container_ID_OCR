"""
Stage 2 – Preprocessing
========================
Transforms a raw plate crop into a clean, normalised image ready
for the recognition stage.

Steps applied in order
----------------------
1. Grayscale conversion
2. Deskew       – corrects camera/mounting tilt via Hough-line angle
3. CLAHE        – Contrast Limited Adaptive Histogram Equalisation
4. Sharpen      – unsharp mask to enhance character edges
5. Resize       – fixed H × W expected by the CRNN
6. Normalise    – zero-mean / unit-std tensor
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PlatePreprocessor:
    """
    Applies the full preprocessing chain to a cropped plate image.

    Parameters
    ----------
    cfg : dict
        ``preprocessing`` section of model_config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self.target_h   = cfg["target_height"]
        self.target_w   = cfg["target_width"]
        self.grayscale  = cfg.get("grayscale", True)

        deskew_cfg      = cfg.get("deskew", {})
        self.do_deskew  = deskew_cfg.get("enabled", True)
        self.max_angle  = deskew_cfg.get("max_angle_deg", 15)

        clahe_cfg       = cfg.get("clahe", {})
        self.do_clahe   = clahe_cfg.get("enabled", True)
        self.clahe      = cv2.createCLAHE(
            clipLimit=clahe_cfg.get("clip_limit", 2.0),
            tileGridSize=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
        )

        sharpen_cfg     = cfg.get("sharpen", {})
        self.do_sharpen = sharpen_cfg.get("enabled", True)
        self.sharpk     = sharpen_cfg.get("kernel_size", 3)
        self.sharps     = sharpen_cfg.get("strength", 1.5)

        norm_cfg        = cfg.get("normalize", {})
        self.mean       = norm_cfg.get("mean", 0.5)
        self.std        = norm_cfg.get("std", 0.5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, crop: np.ndarray) -> np.ndarray:
        """
        Full preprocessing chain.

        Parameters
        ----------
        crop : np.ndarray
            BGR or grayscale plate crop (uint8).

        Returns
        -------
        np.ndarray
            Float32 array of shape (1, target_h, target_w) normalised to
            approximately [-1, 1], ready for the CRNN.
        """
        img = self._to_gray(crop)
        if self.do_deskew:
            img = self._deskew(img)
        if self.do_clahe:
            img = self._apply_clahe(img)
        if self.do_sharpen:
            img = self._sharpen(img)
        img = self._resize(img)
        img = self._normalize(img)
        return img  # shape: (1, H, W)  float32

    # ------------------------------------------------------------------
    # Individual transform steps (also usable standalone)
    # ------------------------------------------------------------------

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if self.grayscale and img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Estimate text-line angle via Hough transform on Canny edges,
        then rotate to horizontal.
        """
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

        if lines is None:
            return gray

        angles = []
        for line in lines[:20]:   # use the 20 strongest lines only
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if abs(angle) <= self.max_angle:
                angles.append(angle)

        if not angles:
            return gray

        angle = float(np.median(angles))
        logger.debug("Deskew angle: %.2f°", angle)

        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE for illumination normalisation."""
        return self.clahe.apply(gray)

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        """Unsharp mask: sharpened = original + strength × (original − blur)."""
        blurred = cv2.GaussianBlur(gray, (self.sharpk, self.sharpk), 0)
        return cv2.addWeighted(gray, 1 + self.sharps, blurred, -self.sharps, 0)

    def _resize(self, gray: np.ndarray) -> np.ndarray:
        return cv2.resize(
            gray, (self.target_w, self.target_h),
            interpolation=cv2.INTER_LINEAR,
        )

    def _normalize(self, gray: np.ndarray) -> np.ndarray:
        """Scale to [0,1] then apply (x - mean) / std.  Returns float32."""
        img = gray.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img[np.newaxis, ...]   # (1, H, W)
