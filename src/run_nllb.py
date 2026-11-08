"""Strong translation baseline: NLLB-200-distilled-600M, beam search (4), deterministic, 1 run.
Batched, resumable (skips items already in results/raw_outputs.jsonl), saves after every batch."""
import json, datetime
import torch
from common import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NAME, TAG = "facebook/nllb-200-distilled-600M", "nllb-200-distilled-600M"
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "(if this says cpu: Runtime > Change runtime type > GPU, then rerun)", flush=True)
tok = AutoTokenizer.from_pretrained(NAME)
mod = AutoModelForSeq2SeqLM.from_pretrained(NAME).to(dev).eval()
eng, kaz = load_flores("devtest"); ids = trans_sample_ids(len(eng))
path = RES / "raw_outputs.jsonl"
done = set()
if path.exists():
    for l in path.open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("model") == TAG: done.add((r["cond"], r["item_id"]))
BS = 16 if dev == "cuda" else 8

def translate(texts, s, t):
    tok.src_lang = s
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
    with torch.no_grad():
        out = mod.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(t), num_beams=4, max_new_tokens=512, max_length=None)
    return tok.batch_decode(out, skip_special_tokens=True)

total = 0
for cond, src, s, t in (("eng2kaz", eng, "eng_Latn", "kaz_Cyrl"), ("kaz2eng", kaz, "kaz_Cyrl", "eng_Latn")):
    todo = sorted([i for i in ids if (cond, i) not in done], key=lambda i: len(src[i]))
    for b in range(0, len(todo), BS):
        batch = todo[b:b + BS]
        outs = translate([src[i] for i in batch], s, t)
        with path.open("a", encoding="utf-8") as f:
            for i, o in zip(batch, outs):
                f.write(json.dumps(dict(model=TAG, model_id=NAME, run=1, task="trans", cond=cond, item_id=i, prompt="", output=o,
                                        finish_reason="stop", temperature=0,
                                        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()), ensure_ascii=False) + "\n")
        total += len(batch)
        print(f"{cond}: {b + len(batch)}/{len(todo)} (written {total})", flush=True)
print("NLLB done. Expected 200 records in total (100 per direction).")