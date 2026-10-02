"""
tests/test_prepare_dataset.py
=============================
Tests for the character-segmentation dataset builder (src/data/prepare_dataset.py).

Run with:  pytest tests/test_prepare_dataset.py -v
"""

import cv2
import numpy as np
import pytest

from src.data.prepare_dataset import (
    Box,
    extract_mask,
    group_id,
    grouped_split,
    image_key,
    iso6346_valid,
    reading_order,
)


class TestNames:
    def test_image_key_strips_roboflow_hash(self):
        name = "1-122830001-OCR-AH-A01_jpg.rf.5957b57c4c0e9ee82077bbfb3b710db6.jpg"
        assert image_key(name) == "1-122830001-OCR-AH-A01"

    def test_image_key_accepts_short_name(self):
        assert image_key("1-122830001-OCR-AH-A01.jpg") == "1-122830001-OCR-AH-A01"

    def test_group_id_is_capture_timestamp(self):
        assert group_id("1-123603001-OCR-LF-C01_jpg.rf.abc.jpg") == "123603001"


class TestCheckDigit:
    @pytest.mark.parametrize("code", ["MEDU4024195", "AMFU3254527", "DRYU2031301"])
    def test_real_codes_from_dataset_are_valid(self, code):
        assert iso6346_valid(code)

    def test_typo_is_rejected(self):
        assert not iso6346_valid("MEDU4024196")

    def test_wrong_length_is_rejected(self):
        assert not iso6346_valid("MEDU402419")


def _boxes_along(points):
    return [Box(x - 10, y - 10, x + 10, y + 10) for x, y in points]


class TestReadingOrder:
    def test_vertical_code_reads_top_to_bottom(self):
        pts = [(100 + i, 50 + 40 * i) for i in range(11)]
        shuffled = _boxes_along(pts[::-1])
        ordered = reading_order(shuffled)
        assert [b.centre[1] for b in ordered] == sorted(b.centre[1] for b in ordered)

    def test_horizontal_code_reads_left_to_right(self):
        pts = [(50 + 30 * i, 200 - i) for i in range(11)]
        ordered = reading_order(_boxes_along(pts[::-1]))
        assert [b.centre[0] for b in ordered] == sorted(b.centre[0] for b in ordered)

    def test_tilted_vertical_code(self):
        pts = [(300 - 9 * i, 100 + 60 * i) for i in range(11)]   # leans left, like AMFU
        ordered = reading_order(_boxes_along(pts[5:] + pts[:5]))
        assert ordered[0].centre == (300.0, 100.0)


class TestSplit:
    def test_container_never_in_two_splits(self):
        keys = ["A", "A", "B", "C", "C", "D", "E", "F", "G", "H"]
        split = grouped_split(keys, (0.6, 0.2, 0.2), seed=1)
        assert set(split) == set(keys)
        assert {"train", "val", "test"} <= set(split.values())

    def test_old_containers_keep_their_split_when_data_grows(self):
        first = grouped_split([f"C{i}" for i in range(10)], (0.6, 0.2, 0.2), seed=1)
        grown = grouped_split([f"C{i}" for i in range(25)], (0.6, 0.2, 0.2), seed=1, previous=first)
        assert all(grown[c] == first[c] for c in first)

    def test_split_shares_follow_targets(self):
        split = grouped_split([f"C{i}" for i in range(50)], (0.6, 0.2, 0.2), seed=3)
        counts = {sp: list(split.values()).count(sp) for sp in ("train", "val", "test")}
        assert counts == {"train": 30, "val": 10, "test": 10}


class TestExtractMask:
    @staticmethod
    def _canvas(dark_text: bool):
        bg, fg = (210, 30) if dark_text else (50, 240)
        img = np.full((200, 200), bg, np.uint8)
        cv2.putText(img, "M", (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.0, fg, 6)
        return img

    @pytest.mark.parametrize("dark_text", [True, False])
    def test_character_found_for_both_polarities(self, dark_text):
        contour, flags = extract_mask(self._canvas(dark_text), Box(68, 82, 130, 132))
        assert contour is not None
        assert "no_character_found" not in flags

    def test_frame_around_check_digit_is_ignored(self):
        img = np.full((200, 200), 50, np.uint8)
        cv2.putText(img, "5", (80, 125), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 240, 6)
        cv2.rectangle(img, (65, 60), (135, 140), 240, 3)
        contour, _ = extract_mask(img, Box(72, 68, 128, 132), pad_frac=0.0)
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
        assert w < 60 and h < 60   # the digit, not the 70x80 frame

    def test_empty_box_is_reported(self):
        img = np.full((100, 100), 128, np.uint8)
        contour, flags = extract_mask(img, Box(20, 20, 60, 60))
        assert contour is None


class TestFramedCheckDigit:
    @staticmethod
    def _framed_digit(touching: bool):
        img = np.full((200, 200), 50, np.uint8)
        cv2.rectangle(img, (60, 50), (130, 145), 240, 3)             # printed frame
        y = 128 if touching else 125
        cv2.putText(img, "7", (72, y), cv2.FONT_HERSHEY_SIMPLEX, 2.4, 240, 7)
        if touching:
            cv2.line(img, (78, 52), (118, 52), 240, 7)                # digit merges with frame
        return img

    @pytest.mark.parametrize("touching", [False, True])
    def test_box_drawn_around_frame_returns_digit_not_square(self, touching):
        img = self._framed_digit(touching)
        contour, _ = extract_mask(img, Box(58, 48, 132, 147), pad_frac=0.0, framed=True)
        assert contour is not None
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
        assert w < 0.8 * 70     # narrower than the 70 px frame -> the frame was removed

    def test_bold_unframed_zero_is_kept(self):
        img = np.full((200, 200), 50, np.uint8)
        cv2.ellipse(img, (100, 100), (30, 42), 0, 0, 360, 240, 14)   # thick '0'
        contour, _ = extract_mask(img, Box(58, 46, 142, 154), pad_frac=0.0, framed=True)
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
        assert w > 55 and h > 80      # the whole zero survives