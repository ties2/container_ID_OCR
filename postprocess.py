"""
Stage 4 – Post-processing  &  Stage 5 – Confidence Gate
=========================================================

ISO 6346 Container ID Format
-----------------------------
  [Owner 3α][Category 1α][Serial 6d][CheckDigit 1d]
  Example: BICU 123456 8

Check-digit algorithm (ISO 6346 §3.1.2)
-----------------------------------------
1. Map each of the first 10 characters to a numeric value:
       0–9  → face value
       A    → 10,  B → 12,  C → 13  …  (skip multiples of 11)
2. Multiply by position weight  2^(position-1)  where position ∈ [1..10]
3. Sum all products, divide by 11, take remainder.
4. If remainder == 10, check digit is 0 (not X).
5. Compare computed value to the stated check digit.

Confidence Gate
---------------
  score ≥ commit_threshold   → COMMIT (write to output)
  flag_threshold ≤ score     → FLAG   (send to human review queue)
  score < flag_threshold     → DISCARD
"""

from __future__ import annotations

import json
import logging
import re
import string
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Stage 4 – ISO 6346 Validation
# ──────────────────────────────────────────────────────────────────────

# Pre-compute char → numeric value map.
# Letters skip multiples of 11: A=10 B=12 C=13 D=14 … K=21(skip 22→23) …
def _build_char_value_map() -> dict[str, int]:
    mapping: dict[str, int] = {}
    # Digits 0-9 map to themselves
    for d in string.digits:
        mapping[d] = int(d)
    # Letters: start at 10, skip multiples of 11
    val = 10
    for ch in string.ascii_uppercase:
        while val % 11 == 0:
            val += 1
        mapping[ch] = val
        val += 1
    return mapping


_CHAR_VALUE = _build_char_value_map()

# Regex to pull apart a raw OCR string into ISO 6346 components.
# Tolerates spaces, dashes, dots inserted by the OCR engine.
_ID_PATTERN = re.compile(
    r"^([A-Z]{3})"          # owner code (3 alpha)
    r"([UJZ])"              # equipment category
    r"[\s\-\.]?"            # optional separator
    r"(\d{6})"              # serial number
    r"[\s\-\.]?"            # optional separator
    r"(\d)$",               # check digit
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    raw_text: str
    owner: str               = ""
    category: str            = ""
    serial: str              = ""
    check_digit: int         = -1
    computed_check: int      = -1
    is_valid_format: bool    = False
    is_valid_checksum: bool  = False
    normalised_id: str       = ""   # e.g. "BICU1234568"


def validate_iso6346(raw_text: str) -> ValidationResult:
    """
    Parse and checksum-validate a raw OCR string against ISO 6346.

    Parameters
    ----------
    raw_text : str
        Whatever the recognition stage produced, e.g. ``"BICU 123456 8"``.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult(raw_text=raw_text)
    text = raw_text.upper().strip()
    m = _ID_PATTERN.match(text)

    if not m:
        logger.debug("Format mismatch: '%s'", raw_text)
        return result

    owner, category, serial, check_str = m.group(1, 2, 3, 4)
    check_digit   = int(check_str)
    computed      = _compute_check_digit(owner + category + serial)

    result.owner             = owner
    result.category          = category.upper()
    result.serial            = serial
    result.check_digit       = check_digit
    result.computed_check    = computed
    result.is_valid_format   = True
    result.is_valid_checksum = (check_digit == computed)
    result.normalised_id     = f"{owner}{category}{serial}{check_digit}"

    if not result.is_valid_checksum:
        logger.debug(
            "Checksum fail for '%s': stated=%d computed=%d",
            result.normalised_id, check_digit, computed,
        )

    return result


def _compute_check_digit(ten_chars: str) -> int:
    """ISO 6346 check-digit over the first 10 characters."""
    assert len(ten_chars) == 10, f"Expected 10 chars, got {len(ten_chars)}"
    total = sum(
        _CHAR_VALUE[ch] * (2 ** i)
        for i, ch in enumerate(ten_chars.upper())
    )
    remainder = total % 11
    return 0 if remainder == 10 else remainder


# ──────────────────────────────────────────────────────────────────────
# Stage 5 – Confidence Gate
# ──────────────────────────────────────────────────────────────────────

class GateDecision(Enum):
    COMMIT  = auto()   # high confidence → write to output
    REVIEW  = auto()   # medium confidence → send to human
    DISCARD = auto()   # low confidence → drop


@dataclass
class GatedRead:
    """Final pipeline output for one detection."""
    decision:     GateDecision
    container_id: Optional[str]        # populated on COMMIT
    confidence:   float
    validation:   ValidationResult
    source_image: Optional[str] = None
    frame_id:     Optional[int] = None


class ConfidenceGate:
    """
    Applies threshold-based gating and writes low-confidence reads to a
    human-review queue (JSONL file).

    Parameters
    ----------
    cfg : dict
        ``confidence_gate`` section of model_config.yaml.
    postproc_cfg : dict
        ``postprocessing`` section (to know whether to enforce checksum).
    """

    def __init__(self, cfg: dict, postproc_cfg: dict) -> None:
        self.commit_thr    = cfg["commit_threshold"]
        self.flag_thr      = cfg["flag_threshold"]
        self.review_path   = Path(cfg["human_review_queue"])
        self.enforce_check = postproc_cfg.get("reject_on_checksum_fail", True)
        self.review_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        confidence: float,
        validation: ValidationResult,
        source_image: Optional[str] = None,
        frame_id: Optional[int] = None,
    ) -> GatedRead:
        """
        Decide whether to commit, flag for review, or discard a read.

        The checksum must pass (when ``reject_on_checksum_fail=true``)
        even if confidence is high.
        """
        checksum_ok = (
            validation.is_valid_checksum
            if self.enforce_check
            else True
        )

        if confidence >= self.commit_thr and checksum_ok:
            decision     = GateDecision.COMMIT
            container_id = validation.normalised_id
            logger.info("COMMIT  %s  conf=%.3f", container_id, confidence)

        elif confidence >= self.flag_thr:
            decision     = GateDecision.REVIEW
            container_id = None
            logger.warning(
                "REVIEW  '%s'  conf=%.3f  checksum=%s",
                validation.raw_text, confidence, checksum_ok,
            )
            self._write_review(validation, confidence, source_image, frame_id)

        else:
            decision     = GateDecision.DISCARD
            container_id = None
            logger.debug("DISCARD '%s'  conf=%.3f", validation.raw_text, confidence)

        return GatedRead(
            decision=decision,
            container_id=container_id,
            confidence=confidence,
            validation=validation,
            source_image=source_image,
            frame_id=frame_id,
        )

    # ------------------------------------------------------------------

    def _write_review(
        self,
        val: ValidationResult,
        confidence: float,
        source_image: Optional[str],
        frame_id: Optional[int],
    ) -> None:
        record = {
            "raw_text":    val.raw_text,
            "normalised":  val.normalised_id,
            "confidence":  round(confidence, 4),
            "checksum_ok": val.is_valid_checksum,
            "source":      source_image,
            "frame_id":    frame_id,
        }
        with self.review_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
