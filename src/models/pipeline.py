"""
End-to-End Container ID OCR Pipeline
======================================
Orchestrates all five stages for a single image or a video stream.

Usage (CLI)
-----------
    python -m src.models.pipeline --image data/01_raw/container.jpg
    python -m src.models.pipeline --video data/01_raw/yard_cam.mp4
    python -m src.models.pipeline --config configs/model_config.yaml --image img.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np
import yaml

from src.models.detector   import PlateDetector, Detection
from src.features.preprocess import PlatePreprocessor
from src.models.recognizer import ContainerRecogniser, RecognitionResult
from src.models.postprocess import (
    ConfidenceGate,
    GatedRead,
    GateDecision,
    validate_iso6346,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Pipeline result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Complete output for one image."""
    source: str
    reads: List[GatedRead]
    elapsed_ms: float

    @property
    def committed(self) -> List[str]:
        return [
            r.container_id
            for r in self.reads
            if r.decision == GateDecision.COMMIT and r.container_id
        ]

    def to_dict(self) -> dict:
        out = {"source": self.source, "elapsed_ms": round(self.elapsed_ms, 2), "reads": []}
        for r in self.reads:
            out["reads"].append({
                "decision":     r.decision.name,
                "container_id": r.container_id,
                "confidence":   round(r.confidence, 4),
                "raw_text":     r.validation.raw_text,
                "checksum_ok":  r.validation.is_valid_checksum,
            })
        return out


# ──────────────────────────────────────────────────────────────────────
# Pipeline class
# ──────────────────────────────────────────────────────────────────────

class ContainerIDPipeline:
    """
    Five-stage OCR pipeline.

    Parameters
    ----------
    config_path : str | Path
        Path to ``configs/model_config.yaml``.
    """

    def __init__(self, config_path: str | Path = "configs/model_config.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg

        # Stage 1 – Detection
        self.detector     = PlateDetector(cfg["detection"])

        # Stage 2 – Preprocessing
        self.preprocessor = PlatePreprocessor(cfg["preprocessing"])

        # Stage 3 – Recognition
        self.recogniser   = ContainerRecogniser(cfg["recognition"])

        # Stage 5 – Confidence Gate  (Stage 4 is a pure function inside run())
        self.gate         = ConfidenceGate(
            cfg["confidence_gate"],
            cfg["postprocessing"],
        )

        logger.info("Pipeline initialised.  Config: %s", config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image: np.ndarray,
        source_label: Optional[str] = None,
        frame_id: Optional[int] = None,
    ) -> PipelineResult:
        """
        Process a single BGR image through all five stages.

        Parameters
        ----------
        image : np.ndarray
            Full-frame BGR image as loaded by cv2.
        source_label : str, optional
            Human-readable label for logging (file path, camera ID …).
        frame_id : int, optional
            Frame index when processing video.

        Returns
        -------
        PipelineResult
        """
        t0 = time.perf_counter()
        reads: List[GatedRead] = []

        # ── Stage 1: Detect plate regions ──────────────────────────────
        detections: List[Detection] = self.detector.detect(image)
        if not detections:
            logger.info("No plates detected in %s", source_label)
            return PipelineResult(
                source=source_label or "",
                reads=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        for det in detections:
            # ── Stage 2: Preprocess crop ───────────────────────────────
            plate_tensor = self.preprocessor.process(det.crop)

            # ── Stage 3: Recognise text ────────────────────────────────
            rec: RecognitionResult = self.recogniser.recognise(plate_tensor)
            logger.debug("Raw OCR: '%s'  conf=%.3f", rec.text, rec.confidence)

            # ── Stage 4: ISO 6346 validation (checksum) ────────────────
            validation = validate_iso6346(rec.text)

            # ── Stage 5: Confidence gate ───────────────────────────────
            gated = self.gate.evaluate(
                confidence=rec.confidence,
                validation=validation,
                source_image=source_label,
                frame_id=frame_id,
            )
            reads.append(gated)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Pipeline finished in %.1f ms — %d read(s), %d committed",
            elapsed, len(reads), len([r for r in reads if r.decision == GateDecision.COMMIT]),
        )
        return PipelineResult(source=source_label or "", reads=reads, elapsed_ms=elapsed)

    def run_video(self, video_path: str) -> Iterator[PipelineResult]:
        """
        Generator that yields a :class:`PipelineResult` for every frame
        that contains at least one detection.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = self.run(frame, source_label=video_path, frame_id=frame_id)
                if result.reads:
                    yield result
                frame_id += 1
        finally:
            cap.release()


# ──────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Container ID OCR Pipeline")
    p.add_argument("--config", default="configs/model_config.yaml")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image")
    group.add_argument("--video", help="Path to a video file")
    p.add_argument("--output", help="Write JSON results to this file")
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
    )

    pipeline = ContainerIDPipeline(args.config)
    all_results = []

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(args.image)
        result = pipeline.run(img, source_label=args.image)
        all_results.append(result.to_dict())
        print(json.dumps(result.to_dict(), indent=2))

    elif args.video:
        for result in pipeline.run_video(args.video):
            d = result.to_dict()
            all_results.append(d)
            print(json.dumps(d))

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
