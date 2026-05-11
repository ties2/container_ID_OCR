"""
src/visualization – Pipeline diagnostics and reporting
=======================================================
Functions
---------
plot_confidence_histogram   : distribution of recognition confidence scores
plot_gate_breakdown         : commit / review / discard pie chart
plot_preprocessing_steps    : side-by-side of each transform stage
save_annotated_frame        : draw bboxes + ID text on the original image
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Confidence histogram
# ──────────────────────────────────────────────────────────────────────

def plot_confidence_histogram(
    scores: List[float],
    commit_thr: float = 0.90,
    flag_thr: float   = 0.70,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a histogram of recognition confidence scores with gate thresholds.

    Parameters
    ----------
    scores      : list of float confidence values in [0, 1]
    commit_thr  : green vertical line (commit threshold)
    flag_thr    : orange vertical line (flag threshold)
    save_path   : if given, save figure to this path instead of showing
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores, bins=40, range=(0, 1), color="#4A90D9", edgecolor="white", alpha=0.85)
    ax.axvline(commit_thr, color="#27AE60", lw=2, linestyle="--", label=f"Commit ≥{commit_thr}")
    ax.axvline(flag_thr,   color="#E67E22", lw=2, linestyle="--", label=f"Review ≥{flag_thr}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Recognition Confidence Distribution")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────
# Gate breakdown
# ──────────────────────────────────────────────────────────────────────

def plot_gate_breakdown(
    n_commit: int,
    n_review: int,
    n_discard: int,
    save_path: Optional[str] = None,
) -> None:
    """Pie chart of gate decisions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    labels  = ["Commit", "Review", "Discard"]
    sizes   = [n_commit, n_review, n_discard]
    colors  = ["#27AE60", "#E67E22", "#E74C3C"]
    explode = (0.05, 0, 0)

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        explode=explode, autopct="%1.1f%%", startangle=140,
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Confidence Gate Breakdown")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────
# Preprocessing step visualisation
# ──────────────────────────────────────────────────────────────────────

def plot_preprocessing_steps(
    original: np.ndarray,
    after_deskew: np.ndarray,
    after_clahe: np.ndarray,
    after_sharpen: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side comparison of each preprocessing stage.
    All inputs should be single-channel (grayscale) uint8 images.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    stages = [
        ("Original",     original),
        ("Deskewed",     after_deskew),
        ("CLAHE",        after_clahe),
        ("Sharpened",    after_sharpen),
    ]

    fig, axes = plt.subplots(1, len(stages), figsize=(16, 3))
    for ax, (title, img) in zip(axes, stages):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle("Preprocessing Pipeline", fontsize=11, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────
# Annotated frame helper
# ──────────────────────────────────────────────────────────────────────

def save_annotated_frame(
    image: np.ndarray,
    detections,        # List[Detection]
    gated_reads,       # List[GatedRead]
    save_path: str,
) -> None:
    """
    Draw bounding boxes and committed container IDs on the original frame
    and save as an image file.

    Colour scheme
    -------------
    Green  : COMMIT
    Orange : REVIEW
    Red    : DISCARD
    """
    from src.models.postprocess import GateDecision

    colour_map = {
        GateDecision.COMMIT:  (0, 200, 0),
        GateDecision.REVIEW:  (0, 150, 255),
        GateDecision.DISCARD: (0, 0, 220),
    }

    vis = image.copy()

    for det, read in zip(detections, gated_reads):
        x1, y1, x2, y2 = det.bbox
        colour = colour_map[read.decision]
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        label = read.container_id or read.validation.raw_text or "?"
        label += f"  {read.confidence:.2f}"
        cv2.putText(
            vis, label, (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
        )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, vis)
