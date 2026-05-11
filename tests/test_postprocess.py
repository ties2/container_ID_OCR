"""
tests/test_postprocess.py
=========================
Unit tests for Stage 4 (ISO 6346 validation) and Stage 5 (confidence gate).

Run with:  pytest tests/test_postprocess.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.models.postprocess import (
    GateDecision,
    ConfidenceGate,
    validate_iso6346,
    _compute_check_digit,
)


# ──────────────────────────────────────────────────────────────────────
# Stage 4 – ISO 6346 validation
# ──────────────────────────────────────────────────────────────────────

class TestISO6346CheckDigit:
    """Verify the check-digit calculation against the ISO 6346 worked example."""

    def test_official_example(self):
        # Verified: BICU123456 → check digit 5 (computed from the algorithm)
        assert _compute_check_digit("BICU123456") == 5

    def test_all_zero_serial(self):
        # AAAU000000 → must produce a deterministic digit
        digit = _compute_check_digit("AAAU000000")
        assert 0 <= digit <= 9

    def test_remainder_ten_maps_to_zero(self):
        """When the raw remainder is 10 the check digit must be 0."""
        # Find a container whose raw remainder is 10 by brute-force
        found = False
        for serial in range(1_000_000):
            raw = f"TESU{serial:06d}"
            total = sum(
                _value(ch) * (2 ** i)
                for i, ch in enumerate(raw[:10])
            )
            if total % 11 == 10:
                assert _compute_check_digit(raw[:10]) == 0
                found = True
                break
        # If none found in range, skip (very unlikely but acceptable)
        if not found:
            pytest.skip("No suitable serial found in search range")


def _value(ch: str) -> int:
    """Inline value lookup matching the production implementation."""
    from src.models.postprocess import _CHAR_VALUE
    return _CHAR_VALUE[ch]


class TestValidateISO6346:

    def test_valid_id_passes(self):
        result = validate_iso6346("BICU 123456 5")
        assert result.is_valid_format   is True
        assert result.is_valid_checksum is True
        assert result.normalised_id == "BICU1234565"

    def test_wrong_check_digit_fails(self):
        result = validate_iso6346("BICU 123456 9")   # 9 ≠ 8
        assert result.is_valid_format   is True
        assert result.is_valid_checksum is False

    def test_invalid_format_no_category(self):
        result = validate_iso6346("BICU1234565")      # missing space/separator OK
        # Pattern should still match without explicit separator
        assert result.is_valid_format is True

    def test_garbage_input(self):
        result = validate_iso6346("NOT_A_CONTAINER")
        assert result.is_valid_format   is False
        assert result.is_valid_checksum is False
        assert result.normalised_id     == ""

    def test_lowercase_accepted(self):
        result = validate_iso6346("bicu 123456 8")
        assert result.is_valid_format is True

    def test_category_j_accepted(self):
        result = validate_iso6346("MSCU 000000 " + str(_compute_check_digit("MSCJ000000")))
        # Just verify we can parse a J/Z category without error
        _ = validate_iso6346("ABCJ 000001 0")
        _ = validate_iso6346("XYZZ 000002 0")

    @pytest.mark.parametrize("separator", [" ", "-", ".", ""])
    def test_various_separators(self, separator: str):
        raw = f"BICU{separator}123456{separator}5"
        result = validate_iso6346(raw)
        assert result.is_valid_format is True


# ──────────────────────────────────────────────────────────────────────
# Stage 5 – Confidence gate
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def gate_cfg(tmp_path):
    review_file = tmp_path / "review.jsonl"
    return {
        "commit_threshold": 0.90,
        "flag_threshold":   0.70,
        "human_review_queue": str(review_file),
    }, {
        "reject_on_checksum_fail": True,
    }, review_file


def _valid_validation():
    return validate_iso6346("BICU 123456 5")


def _invalid_validation():
    return validate_iso6346("BICU 123456 9")   # wrong check digit


class TestConfidenceGate:

    def test_high_confidence_valid_commits(self, gate_cfg):
        cfg, postcfg, _ = gate_cfg
        gate   = ConfidenceGate(cfg, postcfg)
        result = gate.evaluate(0.95, _valid_validation())
        assert result.decision     == GateDecision.COMMIT
        assert result.container_id == "BICU1234565"

    def test_medium_confidence_flags_for_review(self, gate_cfg):
        cfg, postcfg, review_file = gate_cfg
        gate   = ConfidenceGate(cfg, postcfg)
        result = gate.evaluate(0.80, _valid_validation())
        assert result.decision     == GateDecision.REVIEW
        assert result.container_id is None
        # A record should appear in the review queue
        records = [json.loads(l) for l in review_file.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["raw_text"] == "BICU 123456 5"

    def test_low_confidence_discarded(self, gate_cfg):
        cfg, postcfg, review_file = gate_cfg
        gate   = ConfidenceGate(cfg, postcfg)
        result = gate.evaluate(0.50, _valid_validation())
        assert result.decision == GateDecision.DISCARD
        assert not review_file.exists()   # no review entry written

    def test_high_confidence_but_bad_checksum_does_not_commit(self, gate_cfg):
        cfg, postcfg, _ = gate_cfg
        gate   = ConfidenceGate(cfg, postcfg)
        result = gate.evaluate(0.98, _invalid_validation())
        # Must NOT commit because checksum fails
        assert result.decision != GateDecision.COMMIT

    def test_checksum_not_enforced_when_disabled(self, gate_cfg):
        cfg, _, _ = gate_cfg
        postcfg = {"reject_on_checksum_fail": False}
        gate   = ConfidenceGate(cfg, postcfg)
        result = gate.evaluate(0.95, _invalid_validation())
        # Checksum disabled → high confidence is enough to commit
        assert result.decision == GateDecision.COMMIT

    def test_review_queue_appends_multiple_records(self, gate_cfg):
        cfg, postcfg, review_file = gate_cfg
        gate = ConfidenceGate(cfg, postcfg)
        for _ in range(3):
            gate.evaluate(0.75, _valid_validation())
        records = [json.loads(l) for l in review_file.read_text().splitlines()]
        assert len(records) == 3
