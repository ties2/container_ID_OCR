"""
tests/test_preprocess.py
========================
Tests for Stage 2 – preprocessing transforms.
"""

import numpy as np
import pytest

from src.features.preprocess import PlatePreprocessor

DEFAULT_CFG = {
    "target_height": 64,
    "target_width":  512,
    "grayscale":     True,
    "deskew":        {"enabled": True,  "max_angle_deg": 15},
    "clahe":         {"enabled": True,  "clip_limit": 2.0, "tile_grid_size": [8, 8]},
    "sharpen":       {"enabled": True,  "kernel_size": 3,  "strength": 1.5},
    "normalize":     {"mean": 0.5,      "std": 0.5},
}


@pytest.fixture
def preprocessor():
    return PlatePreprocessor(DEFAULT_CFG)


@pytest.fixture
def sample_bgr():
    """Synthetic 120×400 BGR plate image."""
    return (np.random.rand(120, 400, 3) * 255).astype(np.uint8)


class TestPlatePreprocessor:

    def test_output_shape(self, preprocessor, sample_bgr):
        out = preprocessor.process(sample_bgr)
        assert out.shape == (1, 64, 512), f"Unexpected shape: {out.shape}"

    def test_output_dtype(self, preprocessor, sample_bgr):
        out = preprocessor.process(sample_bgr)
        assert out.dtype == np.float32

    def test_output_range(self, preprocessor, sample_bgr):
        """Normalised output should be roughly in [-3, 3] for valid images."""
        out = preprocessor.process(sample_bgr)
        assert out.min() >= -5.0
        assert out.max() <=  5.0

    def test_grayscale_input_accepted(self, preprocessor):
        gray = (np.random.rand(120, 400) * 255).astype(np.uint8)
        out  = preprocessor.process(gray)
        assert out.shape == (1, 64, 512)

    def test_deskew_disabled(self, sample_bgr):
        cfg = {**DEFAULT_CFG, "deskew": {"enabled": False}}
        pre = PlatePreprocessor(cfg)
        out = pre.process(sample_bgr)
        assert out.shape == (1, 64, 512)

    def test_clahe_disabled(self, sample_bgr):
        cfg = {**DEFAULT_CFG, "clahe": {"enabled": False}}
        pre = PlatePreprocessor(cfg)
        out = pre.process(sample_bgr)
        assert out.shape == (1, 64, 512)

    def test_sharpen_disabled(self, sample_bgr):
        cfg = {**DEFAULT_CFG, "sharpen": {"enabled": False}}
        pre = PlatePreprocessor(cfg)
        out = pre.process(sample_bgr)
        assert out.shape == (1, 64, 512)

    def test_deterministic(self, preprocessor, sample_bgr):
        """Same input must always produce the same output."""
        out1 = preprocessor.process(sample_bgr.copy())
        out2 = preprocessor.process(sample_bgr.copy())
        np.testing.assert_array_equal(out1, out2)
