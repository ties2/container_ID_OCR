# from pathlib import Path
# from collections import defaultdict
# groups, splits = defaultdict(list), defaultdict(set)
# for split in ["train", "valid", "test"]:
#     for f in Path(split, "images").glob("*.jpg"):
#         gid = f.name.split("-")[1]          # e.g. 144815001
#         groups[gid].append(f.name); splits[gid].add(split)
# sizes = [len(v) for v in groups.values()]
# print("groups:", len(groups), "| max images/group:", max(sizes))
# print("groups in >1 split (leakage):", sum(len(s) > 1 for s in splits.values()))
# print("example:", next(v for v in groups.values() if len(v) > 1))
#
# ######
# train_g = {g for g, s in splits.items() if "train" in s}
# for sp in ["valid", "test"]:
#     imgs = [(g, n) for g, ns in groups.items() for n in ns if sp in splits[g] and Path(sp, "images", n).exists()]
#     leak = sum(g in train_g for g, _ in imgs)
#     print(f"{sp}: {leak}/{len(imgs)} images ({100*leak/len(imgs):.0f}%) have a same-event image in train")
import shutil
from pathlib import Path
for f in Path("train/images").glob("*.jpg"):
    cam = f.name.split("-OCR-")[1][:2]          # AH, AS, LB, LF, RF
    out = Path("by_camera", cam); out.mkdir(parents=True, exist_ok=True)
    shutil.copy(f, out / f.name)