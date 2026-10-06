"""
tests/test_evaluate.py
======================
Tests for the reading metrics in src/models/evaluate.py.

Run with:  pytest tests/test_evaluate.py -v
"""

import numpy as np
import pytest

from src.data.prepare_dataset import CLASS_ID
from src.models.evaluate import code_layout, iso_format_correct, levenshtein, read_code, score_read


def _square(cx, cy, r=10):
    return np.array([[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]], np.int32)


def _vertical_code(text, x=100, y0=50, step=40):
    return [(CLASS_ID[ch], _square(x, y0 + i * step)) for i, ch in enumerate(text)]


CODE_POLY = np.array([[70, 20], [130, 20], [130, 500], [70, 500]], np.float32)


class TestLevenshtein:
    @pytest.mark.parametrize("a,b,d", [("MEDU", "MEDU", 0), ("MEDU", "MEDO", 1), ("MEDU", "MED", 1), ("", "ABC", 3)])
    def test_distance(self, a, b, d):
        assert levenshtein(a, b) == d


class TestReadCode:
    def test_vertical_code_in_shuffled_order(self):
        inst = _vertical_code("MEDU4024195")
        assert read_code(inst[::-1], CODE_POLY) == "MEDU4024195"

    def test_characters_outside_the_code_are_ignored(self):
        inst = _vertical_code("MEDU4024195") + [(CLASS_ID["2"], _square(400, 100))]   # e.g. '22G1'
        assert read_code(inst, CODE_POLY) == "MEDU4024195"

    def test_nothing_found(self):
        assert read_code([], CODE_POLY) == ""


class TestScore:
    def test_perfect_read(self):
        sc = score_read("MEDU4024195", "MEDU4024195")
        assert sc == {"char_acc": 1.0, "exact": 1.0, "checkdigit_ok": 1.0}

    def test_one_wrong_character(self):
        sc = score_read("MEDU4024196", "MEDU4024195")
        assert sc["exact"] == 0.0 and sc["char_acc"] == pytest.approx(10 / 11)
        assert sc["checkdigit_ok"] == 0.0      # the check digit catches the error

    def test_missing_character(self):
        assert score_read("MEDU402195", "MEDU4024195")["char_acc"] == pytest.approx(10 / 11)


class TestFormatCorrection:
    @pytest.mark.parametrize("raw,fixed", [
        ("BS1U3121575", "BSIU3121575"),     # digit 1 where a letter must be
        ("BM0U4131633", "BMOU4131633"),     # digit 0 where a letter must be
        ("MEDU4O24195", "MEDU4024195"),     # letter O where a digit must be
        ("TCHU0537770", "TCHU0537770"),     # G/C confusion cannot be fixed
    ])
    def test_corrections(self, raw, fixed):
        assert iso_format_correct(raw) == fixed

    def test_incomplete_read_is_left_alone(self):
        assert iso_format_correct("T1U0535670") == "T1U0535670"


def test_layout_from_polygon():
    assert code_layout(CODE_POLY) == "vertical"
    assert code_layout(CODE_POLY[:, ::-1].copy()) == "horizontal"

