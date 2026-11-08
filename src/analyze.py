"""Computes every number in the report from results/raw_outputs.jsonl. Run: python src/analyze.py"""
import os, json, re
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sacrebleu
from common import *

B = int(os.environ.get("N_BOOT", N_BOOT))
rng = np.random.default_rng(SEED)

# ---------- load ----------
raw = pd.DataFrame(json.loads(l) for l in (RES / "raw_outputs.jsonl").open(encoding="utf-8"))
if "finish_reason" in raw: raw = raw[raw.finish_reason != "error"].copy()   # failed API calls are in the failure log, not in the analysis
sid = json.load((RES / "sample_ids.json").open())
T_IDS, C_IDS = sid["flores_devtest_ids"], sid["sib_test_ids"]
eng, kaz = load_flores("devtest")
SRC = {"eng2kaz": [eng[i] for i in T_IDS], "kaz2eng": [kaz[i] for i in T_IDS]}
REF = {"eng2kaz": [kaz[i] for i in T_IDS], "kaz2eng": [eng[i] for i in T_IDS]}
sib_test = load_sib("test"); sib_train = load_sib("train")
GOLD = [r["label"] for r in sib_test]
TEXT = {L: [r[L] for r in sib_test] for L in ("eng", "kaz")}

def chrf(h, r): return sacrebleu.corpus_chrf(list(h), [list(r)]).score
def bleu(h, r): return sacrebleu.corpus_bleu(list(h), [list(r)]).score
def acc(p, g): return float(np.mean(np.array(p) == np.array(g)))
def mf1(p, g): return float(f1_score(g, p, labels=LABELS, average="macro", zero_division=0))

# ---------- assemble arrays: [runs x items] ----------
HYP, PRED, META = {}, {}, {}   # HYP[model][dir], PRED[model][lang]
for m, d in raw.groupby("model"):
    nruns = 0; HYP[m] = {}; PRED[m] = {}
    for c in ("eng2kaz", "kaz2eng"):
        runs = []
        for r, x in d[(d.task == "trans") & (d.cond == c)].groupby("run"):
            if r > N_RUNS["trans"]: continue
            mp = dict(zip(x.item_id, x.output.fillna("")))
            if all(i in mp for i in T_IDS): runs.append([mp[i] for i in T_IDS])
        HYP[m][c] = np.array(runs, dtype=object)
    for L in ("eng", "kaz"):
        runs = []
        for r, x in d[(d.task == "cls") & (d.cond == L)].groupby("run"):
            if r > N_RUNS["cls"]: continue
            mp = dict(zip(x.item_id, x.output.fillna("")))
            if all(i in mp for i in C_IDS): runs.append([parse_label(mp[i]) for i in C_IDS])
        PRED[m][L] = np.array(runs, dtype=object)
    META[m] = dict(n_runs_trans=min(len(HYP[m]["eng2kaz"]), len(HYP[m]["kaz2eng"])),
                   n_runs_cls=min(len(PRED[m]["eng"]), len(PRED[m]["kaz"])))

# ---------- extra classification runs beyond the planned ones: reported separately as a stability check ----------
EXTRA = []
for m, d in raw.groupby("model"):
    for L in ("eng", "kaz"):
        runs = {}
        for r, x in d[(d.task == "cls") & (d.cond == L)].groupby("run"):
            mp = dict(zip(x.item_id, x.output.fillna("")))
            if all(i in mp for i in C_IDS): runs[r] = [parse_label(mp[i]) for i in C_IDS]
        ks = sorted(runs)
        for k in ks[1:]:
            EXTRA.append(dict(model=m, lang=L, run_a=ks[0], run_b=k, identical_predictions=float(np.mean(np.array(runs[ks[0]]) == np.array(runs[k]))),
                              macro_F1_a=mf1(runs[ks[0]], GOLD), macro_F1_b=mf1(runs[k], GOLD)))
pd.DataFrame(EXTRA).to_csv(RES / "extra_runs_stability.csv", index=False)

# ---------- baselines (pseudo-models) ----------
maj = pd.Series([r["label"] for r in sib_train]).mode()[0]
HYP["baseline-copy-source"] = {c: np.array([SRC[c]], dtype=object) for c in SRC}
PRED["baseline-majority"] = {L: np.array([[maj] * len(C_IDS)], dtype=object) for L in ("eng", "kaz")}
rr = np.random.default_rng(SEED)
PRED["baseline-random"] = {L: np.array([rr.choice(LABELS, len(C_IDS))], dtype=object) for L in ("eng", "kaz")}
PRED["baseline-tfidf-lr"] = {}
for L in ("eng", "kaz"):   # trained on SIB-200 train split only, evaluated on the untouched test split
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    X = vec.fit_transform([r[L] for r in sib_train]); y = [r["label"] for r in sib_train]
    clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(X, y)
    PRED["baseline-tfidf-lr"][L] = np.array([clf.predict(vec.transform(TEXT[L]))], dtype=object)
for b in ("baseline-copy-source",): PRED.setdefault(b, {}); META[b] = dict(n_runs_trans=1, n_runs_cls=0)
for b in ("baseline-majority", "baseline-random", "baseline-tfidf-lr"):
    HYP.setdefault(b, {}); META[b] = dict(n_runs_trans=0, n_runs_cls=1)

# ---------- main table: mean +/- SD over runs ----------
rows = []
for m in HYP:
    for c in ("eng2kaz", "kaz2eng"):
        H = HYP[m].get(c)
        if H is None or len(H) == 0: continue
        for name, f in (("chrF", chrf), ("BLEU", bleu)):
            v = [f(h, REF[c]) for h in H]
            rows.append(dict(model=m, task="translation", cond=c, metric=name, mean=np.mean(v), sd=np.std(v, ddof=1) if len(v) > 1 else np.nan, n_runs=len(v), n_items=len(T_IDS),
                             empty_outputs=int(sum(1 for h in H for s in h if not str(s).strip()))))
for m in PRED:
    for L in ("eng", "kaz"):
        P = PRED[m].get(L)
        if P is None or len(P) == 0: continue
        for name, f in (("accuracy", acc), ("macro_F1", mf1)):
            v = [f(p, GOLD) for p in P]
            rows.append(dict(model=m, task="classification", cond=L, metric=name, mean=np.mean(v), sd=np.std(v, ddof=1) if len(v) > 1 else np.nan, n_runs=len(v), n_items=len(C_IDS),
                             empty_outputs=int(sum(1 for p in P for s in p if s == "INVALID"))))
main = pd.DataFrame(rows); main.to_csv(RES / "main_results.csv", index=False)

# ---------- paired bootstrap of the gap (English - Kazakh) ----------
def boot_gap(fe, fk, n):
    pt = fe(np.arange(n)) - fk(np.arange(n)); g = np.empty(B)
    for b in range(B):
        ix = rng.integers(0, n, n); g[b] = fe(ix) - fk(ix)
    return pt, np.percentile(g, 2.5), np.percentile(g, 97.5), (np.sum(g <= 0) + 1) / (B + 1)

def avg_over_runs(P, f, gold_or_ref):
    return lambda ix: float(np.mean([f([p[i] for i in ix], [gold_or_ref[i] for i in ix]) for p in P]))

gaps = []
for m in PRED:
    if len(PRED[m].get("eng", [])) and len(PRED[m].get("kaz", [])):
        for name, f in (("macro_F1", mf1), ("accuracy", acc)):
            fe = avg_over_runs(PRED[m]["eng"], f, GOLD); fk = avg_over_runs(PRED[m]["kaz"], f, GOLD)
            pt, lo, hi, p = boot_gap(fe, fk, len(C_IDS))
            eng_score = fe(np.arange(len(C_IDS)))
            gaps.append(dict(model=m, task="classification", metric=name, eng_side=eng_score, kaz_side=fk(np.arange(len(C_IDS))), gap=pt, ci_lo=lo, ci_hi=hi,
                             rel_gap_pct=100 * pt / eng_score if eng_score else np.nan, p_one_sided=p))
for m in HYP:
    if len(HYP[m].get("kaz2eng", [])) and len(HYP[m].get("eng2kaz", [])):
        for name, f in (("chrF", chrf), ("BLEU", bleu)):
            fe = avg_over_runs(HYP[m]["kaz2eng"], f, REF["kaz2eng"]); fk = avg_over_runs(HYP[m]["eng2kaz"], f, REF["eng2kaz"])
            pt, lo, hi, p = boot_gap(fe, fk, len(T_IDS))
            es = fe(np.arange(len(T_IDS)))
            gaps.append(dict(model=m, task="translation", metric=name, eng_side=es, kaz_side=fk(np.arange(len(T_IDS))), gap=pt, ci_lo=lo, ci_hi=hi,
                             rel_gap_pct=100 * pt / es if es else np.nan, p_one_sided=p))
G = pd.DataFrame(gaps)
# Holm correction over all LLM tests (models x metrics); baselines are reference rows and are not in the family
fam = ~G.model.str.startswith("baseline-"); pv = G.p_one_sided.values[fam.values]
order = np.argsort(pv); k = len(pv); adj_f = np.empty(k); run_max = 0
for rank, ix in enumerate(order):
    run_max = max(run_max, (k - rank) * pv[ix]); adj_f[ix] = min(1.0, run_max)
adj = np.full(len(G), np.nan); adj[fam.values] = adj_f
G["p_holm"] = adj; G.to_csv(RES / "gap_analysis.csv", index=False)

# ---------- McNemar (exact) on parallel items: English-correct vs Kazakh-correct, majority vote over runs ----------
mc = []
for m in PRED:
    if len(PRED[m].get("eng", [])) and len(PRED[m].get("kaz", [])):
        ce = (np.mean([np.array(p) == np.array(GOLD) for p in PRED[m]["eng"]], axis=0) > 0.5)
        ck = (np.mean([np.array(p) == np.array(GOLD) for p in PRED[m]["kaz"]], axis=0) > 0.5)
        b, c = int(np.sum(ce & ~ck)), int(np.sum(~ce & ck))
        mc.append(dict(model=m, eng_only_correct=b, kaz_only_correct=c, p_two_sided=binomtest(b, b + c, 0.5).pvalue if b + c else 1.0))
pd.DataFrame(mc).to_csv(RES / "mcnemar.csv", index=False)

# ---------- subgroup: per-class F1 + confusion matrices (pooled over runs) ----------
pc = []
for m in PRED:
    for L in ("eng", "kaz"):
        P = PRED[m].get(L)
        if P is None or len(P) == 0: continue
        pool_p = [x for p in P for x in p]; pool_g = GOLD * len(P)
        f = f1_score(pool_g, pool_p, labels=LABELS, average=None, zero_division=0)
        for l, v in zip(LABELS, f): pc.append(dict(model=m, lang=L, label=l, f1=v, support=GOLD.count(l)))
        pd.DataFrame(confusion_matrix(pool_g, pool_p, labels=LABELS + ["INVALID"] if "INVALID" in pool_p else LABELS)).to_csv(RES / f"confusion_{m}_{L}.csv")
pd.DataFrame(pc).to_csv(RES / "per_class_f1.csv", index=False)

# ---------- subgroup / sensitivity: source sentence length (terciles of English word count) ----------
wc = np.array([len(eng[i].split()) for i in T_IDS]); q = np.quantile(wc, [1/3, 2/3])
grp = np.where(wc <= q[0], "short", np.where(wc <= q[1], "medium", "long")); sg = []
for m in HYP:
    for c in ("eng2kaz", "kaz2eng"):
        H = HYP[m].get(c)
        if H is None or len(H) == 0: continue
        for gname in ("short", "medium", "long"):
            ix = np.where(grp == gname)[0]
            sg.append(dict(model=m, cond=c, length_group=gname, n=len(ix), chrF=np.mean([chrf([h[i] for i in ix], [REF[c][i] for i in ix]) for h in H])))
pd.DataFrame(sg).to_csv(RES / "length_subgroup.csv", index=False)

# ---------- efficiency / cost ----------
eff = raw.groupby(["model", "task", "cond"]).agg(prompt_tokens=("prompt_tokens", "mean"), completion_tokens=("completion_tokens", "mean"),
                                                 latency_s=("latency_s", "mean"), n_calls=("item_id", "size")).reset_index()
eff.to_csv(RES / "efficiency.csv", index=False)

# ---------- error candidates for MANUAL categorisation (>= 30 failed cases; Appendix B) ----------
ec = []
for m in HYP:
    for c in ("eng2kaz", "kaz2eng"):
        H = HYP[m].get(c)
        if H is None or len(H) == 0 or m.startswith("baseline"): continue
        sc = [sacrebleu.sentence_chrf(H[0][j], [REF[c][j]]).score for j in range(len(T_IDS))]
        for j in np.argsort(sc)[:30]:
            h = str(H[0][j]); flags = []
            if not h.strip(): flags.append("empty")
            if c == "eng2kaz" and not re.search("[А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]", h): flags.append("no_cyrillic")
            if len(h) < 0.5 * len(REF[c][j]) or len(h) > 2 * len(REF[c][j]): flags.append("length_mismatch")
            ec.append(dict(model=m, task="translation", cond=c, item_id=T_IDS[j], sentence_chrF=round(sc[j], 1), source=SRC[c][j], reference=REF[c][j], output=h,
                           auto_flags=";".join(flags), manual_error_type=""))
for m in PRED:
    if m.startswith("baseline"): continue
    for L in ("eng", "kaz"):
        P = PRED[m].get(L)
        if P is None or len(P) == 0: continue
        for j in range(len(C_IDS)):
            if P[0][j] != GOLD[j]:
                ec.append(dict(model=m, task="classification", cond=L, item_id=C_IDS[j], sentence_chrF="", source=TEXT[L][j], reference=GOLD[j], output=P[0][j],
                               auto_flags="INVALID" if P[0][j] == "INVALID" else f"{GOLD[j]}->{P[0][j]}", manual_error_type=""))
pd.DataFrame(ec).to_csv(RES / "error_candidates.csv", index=False)

print(main.round(3).to_string()); print(G.round(4).to_string()); print("incomplete/partial runs:", META)
