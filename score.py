"""Split evaluation results into held-out layouts (0-9) and training layouts (10-49)."""
import glob
import json
import sys

import numpy as np

for name in sys.argv[1:]:
    path = glob.glob(f"outputs/eval/{name}/*.json")[0]
    d = json.load(open(path))
    s = np.array(d["per_task"][0]["metrics"]["successes"], bool)
    print(f"{name:12} held-out 0-9: {s[:10].sum():2d}/10 | "
          f"layouts 10-{len(s) - 1}: {s[10:].sum():2d}/{len(s) - 10} | "
          f"all: {s.sum():2d}/{len(s)} ({100 * s.mean():.0f}%)")