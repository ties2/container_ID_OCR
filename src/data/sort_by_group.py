"""
src/data/sort_by_group.py
=========================
Sort the images of each camera folder into sub-folders by variation group (G1..G8),
using the selection sheets in data/metadata/Label-camera-*.csv. Images with
status 'rejected' go to a `rejected` folder; images not found in any sheet go to
`unlisted`. This only helps browsing and choosing what to annotate next; the
pipeline itself reads the groups from the sheets, not from these folders.

    by_camera/AH/1-...jpg   ->   by_camera/AH/G1/1-...jpg
                                 by_camera/AH/rejected/1-...jpg

Files are COPIED by default (safe to run again: existing copies are skipped).
Use --move to move them instead; after an earlier copy run, --move also deletes the
loose originals that already have a complete copy, which frees the disk space. A table Camera,Group,Count (usable images) is
written to <root>/group_counts.csv; it matches data/metadata/images-camera-info.csv.

Usage (from the project root):
    python -m src.data.sort_by_group
    python -m src.data.sort_by_group --root data/01_raw/archive/by_camera --move
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from src.data.prepare_dataset import image_key, load_metadata

IMAGE_TYPES = {".jpg", ".jpeg", ".png"}


def target_folder(meta: dict[str, dict[str, str]], key: str) -> str:
    """G1..G8, 'rejected' or 'unlisted' for one image."""
    info = meta.get(key)
    if info is None:
        return "unlisted"
    if info["status"] == "rejected":
        return "rejected"
    return info["group"] if info["group"] not in ("", "?") else "unlisted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/01_raw/archive/by_camera"),
                    help="folder with one sub-folder per camera (AH, AS, LB, LF, RF)")
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata"))
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    args = ap.parse_args()

    meta = load_metadata(sorted(args.metadata.glob("Label-camera-*.csv")))
    if not meta:
        raise SystemExit(f"No Label-camera-*.csv found in {args.metadata}")

    table = []
    for cam_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        counts: Counter[str] = Counter()
        # only the images directly in the camera folder, not the ones already sorted
        for img in sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in IMAGE_TYPES):
            folder = target_folder(meta, image_key(img.name))
            dest = cam_dir / folder / img.name
            if dest.exists():
                # Already copied by an earlier run: with --move, remove the loose original
                # (only when the copy is complete, i.e. has the same size).
                if args.move and dest.stat().st_size == img.stat().st_size:
                    img.unlink()
                continue
            dest.parent.mkdir(exist_ok=True)
            (shutil.move if args.move else shutil.copy2)(img, dest)
        # count what is in the sub-folders now (also correct when run again)
        for sub in (p for p in cam_dir.iterdir() if p.is_dir()):
            counts[sub.name] = sum(1 for f in sub.iterdir() if f.suffix.lower() in IMAGE_TYPES)
        summary = "  ".join(f"{k}:{counts[k]}" for k in sorted(counts))
        print(f"{cam_dir.name}: {sum(counts.values())} images  ->  {summary}")
        table += [(cam_dir.name, g, n) for g, n in sorted(counts.items()) if g.startswith("G")]

    with open(args.root / "group_counts.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["Camera", "Group", "Count"])
        wr.writerows(table)
    print(f"Counts written to {args.root / 'group_counts.csv'}")


if __name__ == "__main__":
    main()
