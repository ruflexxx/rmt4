"""Main run: all models x tasks x directions/languages x N_RUNS. Resumable (cache = results/raw_outputs.jsonl)."""
import os, sys, json, time, datetime, argparse
from common import *

class QuotaExhausted(Exception): pass

def key(m, run, task, cond, i): return f"{m}|{run}|{task}|{cond}|{i}"

def load_cache():
    f = RES / "raw_outputs.jsonl"; seen = set()
    if f.exists():
        for l in f.open(encoding="utf-8"):
            r = json.loads(l)
            if r.get("finish_reason") == "error": continue
            seen.add(key(r["model"], r["run"], r["task"], r["cond"], r["item_id"]))
    return seen

def call(client, cfg, prompt):
    last = ""
    for attempt in range(6):
        try:
            t0 = time.time()
            r = client.chat.completions.create(model=cfg["model_id"], temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                                               messages=[{"role": "user", "content": prompt}])
            ch = r.choices[0]; u = r.usage
            return dict(output=ch.message.content or "", finish_reason=ch.finish_reason, latency_s=round(time.time()-t0, 3),
                        prompt_tokens=getattr(u, "prompt_tokens", None), completion_tokens=getattr(u, "completion_tokens", None))
        except Exception as e:
            last = repr(e)[:400]
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                if "per day" in msg or "daily" in msg or "exceeded your current quota" in msg:
                    raise QuotaExhausted(str(e)[:200])
                time.sleep(min(60, 5 * 2 ** attempt)); continue
            time.sleep(2 * (attempt + 1))
    return dict(output="", finish_reason="error", error=last, latency_s=None, prompt_tokens=None, completion_tokens=None)

def build_items():
    eng, kaz = load_flores("devtest"); ids = trans_sample_ids(len(eng))
    items = []
    for i in ids:
        items.append(("trans", "eng2kaz", i, TRANS_PROMPT.format(src="English", tgt="Kazakh", text=eng[i])))
        items.append(("trans", "kaz2eng", i, TRANS_PROMPT.format(src="Kazakh", tgt="English", text=kaz[i])))
    sib = load_sib("test")                       # classification set: SIB-200 test split (untouched)
    for r in sib:
        for L in ("eng", "kaz"):
            items.append(("cls", L, r["id"], CLS_PROMPT.format(text=r[L])))
    json.dump(dict(seed=SEED, flores_devtest_ids=ids, sib_test_ids=[r["id"] for r in sib]), (RES/"sample_ids.json").open("w"))
    return items

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=list(MODELS)); ap.add_argument("--dry", action="store_true"); ap.add_argument("--tasks", nargs="+", default=["cls", "trans"], choices=["cls", "trans"])
    a = ap.parse_args()
    items = build_items(); seen = load_cache()
    items = [x for x in items if x[0] in a.tasks]   # lets you run the cheap, primary classification task first
    print(f"{len(items)} calls per run per model; {sum(N_RUNS[x[0]] for x in items)} per model in total")
    json.dump(dict(flores_devtest_eng_sha256=sha256(DATA/"flores200_dataset/devtest/eng_Latn.devtest"),
                   flores_devtest_kaz_sha256=sha256(DATA/"flores200_dataset/devtest/kaz_Cyrl.devtest"),
                   sample_ids_sha256=sha256(RES/"sample_ids.json")), (RES/"data_hashes.json").open("w"), indent=2)
    if a.dry: sys.exit()
    from openai import OpenAI
    for m in a.models:
        cfg = MODELS[m]; k = os.environ.get(cfg["key_env"])
        if not k: print(f"[skip] {m}: {cfg['key_env']} not set"); continue
        client = OpenAI(base_url=cfg["base_url"], api_key=k); gap = 60.0 / cfg["rpm"]
        bad = 0
        try:
            for run in range(1, max(N_RUNS.values()) + 1):
                for task, cond, i, prompt in items:
                    if run > N_RUNS[task]: continue
                    if key(m, run, task, cond, i) in seen: continue
                    t0 = time.time(); out = call(client, cfg, prompt)
                    bad = bad + 1 if out["finish_reason"] == "error" else 0
                    if bad == 1 or bad == 5: print(f"[ERROR] {m}: {out.get('error')}", flush=True)
                    if bad >= 5: raise QuotaExhausted("5 consecutive failed calls; stopping this model. Fix the cause and rerun (progress is kept).")
                    rec = dict(model=m, model_id=cfg["model_id"], run=run, task=task, cond=cond, item_id=i, prompt=prompt,
                               temperature=TEMPERATURE, max_tokens=MAX_TOKENS, timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), **out)
                    with (RES/"raw_outputs.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    time.sleep(max(0, gap - (time.time() - t0)))
                print(f"[done] {m} run {run}", flush=True)
        except QuotaExhausted as e:
            print(f"[QUOTA EXHAUSTED] {m}: {e}. Progress is cached; rerun later. Do NOT discard: report it in the failure log.")
    json.dump(dict(date=datetime.date.today().isoformat(), seed=SEED, n_trans=N_TRANS, n_runs=N_RUNS, temperature=TEMPERATURE,
               max_tokens=MAX_TOKENS, models={m: {k: v for k, v in c.items() if k != "key_env"} for m, c in MODELS.items()}),
              (RES/"environment.json").open("w"), indent=2)
