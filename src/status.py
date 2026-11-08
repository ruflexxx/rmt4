"""Shows what is done and what is still missing under the CURRENT plan. Run: python src/status.py"""
import collections
from common import *
from run_experiment import build_items, load_cache, key
items = build_items(); seen = load_cache()
print(f"plan: {N_TRANS} translation sentences x2 directions, 204 classification items x2 languages, runs {N_RUNS}\n")
for m in MODELS:
    for task in ("cls", "trans"):
        its = [x for x in items if x[0] == task]
        todo = sum(1 for run in range(1, N_RUNS[task] + 1) for t, c, i, _ in its if key(m, run, t, c, i) not in seen)
        tot = len(its) * N_RUNS[task]
        print(f"{m:24s} {task:5s} done {tot - todo:4d} / {tot:4d}   remaining {todo:4d}")
