"""Merge raw_outputs.jsonl files produced on different machines/keys.
Usage: python src/merge_outputs.py a.jsonl b.jsonl ...   (include results/raw_outputs.jsonl itself)
Writes results/raw_outputs.jsonl (successful calls only, deduplicated; the previous file is backed up)
and results/failed_calls.jsonl (every failed call is kept for the failure log; nothing is silently dropped)."""
import sys, json, shutil, pathlib
from common import RES
files = [pathlib.Path(f) for f in sys.argv[1:]]
assert files, "give at least one input file"
good, bad, n_in = {}, [], 0
for f in files:
    for l in f.open(encoding="utf-8"):
        if not l.strip(): continue
        r = json.loads(l); n_in += 1
        if r.get("finish_reason") == "error": bad.append(r); continue
        good[(r["model"], r["run"], r["task"], r["cond"], r["item_id"])] = r      # later file wins on duplicates
out = RES / "raw_outputs.jsonl"
if out.exists(): shutil.copy(out, RES / "raw_outputs.backup.jsonl")
with out.open("w", encoding="utf-8") as f:
    for r in good.values(): f.write(json.dumps(r, ensure_ascii=False) + "\n")
with (RES / "failed_calls.jsonl").open("w", encoding="utf-8") as f:
    for r in bad: f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(f"read {n_in} records -> {len(good)} successful, {len(bad)} failed (logged)")
print("successful per model:", dict(Counter(k[0] for k in good)))
print("failed per model:", dict(Counter(r["model"] for r in bad)))
