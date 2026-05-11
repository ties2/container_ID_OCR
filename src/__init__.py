"""
Container ID OCR Pipeline
=========================
Five-stage pipeline:
  1. Detection     – locate the ID plate (YOLOv8 / RT-DETR)
  2. Preprocessing – deskew, CLAHE, sharpen
  3. Recognition   – CRNN or VLM → raw text
  4. Post-process  – ISO 6346 checksum validation
  5. Confidence gate – commit / flag / discard
"""

__version__ = "1.0.0"
