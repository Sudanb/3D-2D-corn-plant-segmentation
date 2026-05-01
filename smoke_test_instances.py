"""
Smoke test: scan all segmented PLY files and report instance counts per plant.
Prints stem presence, raw color count, filtered leaf count, and any plants
that would overflow the expected <=16 leaf limit.
"""

import numpy as np
from pathlib import Path
from plyfile import PlyData

SEG_DIR       = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_SegmentedPCD_100k\FielGrwon_ZeaMays_SegmentedPCD_100k"
STEM_CODE     = 0      # black (0,0,0)
MIN_LEAF_POINTS = 50   # must match dataset_build.py

def encode(colors):
    return (colors[:, 0].astype(np.int32) * 65536 +
            colors[:, 1].astype(np.int32) * 256  +
            colors[:, 2].astype(np.int32))

files = sorted(Path(SEG_DIR).glob("*.ply"))
print(f"Found {len(files)} segmented PLY files\n")
print(f"{'File':<12} {'Raw colors':>10} {'Stem':>6} {'Leaves(raw)':>12} {'Leaves(≥50pts)':>15} {'OK?':>5}")
print("-" * 65)

issues = []
for f in files:
    ply    = PlyData.read(str(f))
    v      = ply['vertex']
    colors = np.vstack([v['red'], v['green'], v['blue']]).T.astype(np.uint8)
    enc    = encode(colors)

    unique, counts = np.unique(enc, return_counts=True)
    has_stem   = STEM_CODE in unique
    raw_leaves = int((unique != STEM_CODE).sum())
    filt_leaves = int(((unique != STEM_CODE) & (counts >= MIN_LEAF_POINTS)).sum())
    ok = "OK" if filt_leaves <= 16 else "WARN"

    print(f"{f.name:<12} {len(unique):>10} {str(has_stem):>6} {raw_leaves:>12} {filt_leaves:>15} {ok:>5}")

    if filt_leaves > 16:
        issues.append((f.name, filt_leaves))

print()
if issues:
    print(f"PLANTS WITH >16 LEAVES (raise MIN_LEAF_POINTS or check data):")
    for name, n in issues:
        print(f"  {name}: {n} leaves")
else:
    print("All plants within expected <=16 leaf range.")
