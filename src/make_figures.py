"""Figures from results/*.csv. Run after analyze.py."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES, LABELS
G = pd.read_csv(RES / "gap_analysis.csv")
# Figure 1: gap with 95% CI per model, primary + secondary metrics
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, task, ms in ((axes[0], "classification", ["macro_F1", "accuracy"]), (axes[1], "translation", ["chrF", "BLEU"])):
    d = G[(G.task == task) & G.model.isin([m for m in G.model.unique() if not m.startswith("baseline-")])]
    models = list(d.model.unique()); w = 0.38
    for k, met in enumerate(ms):
        x = d[d.metric == met].set_index("model").reindex(models)
        scale = 100 if task == "classification" else 1
        ax.bar(np.arange(len(models)) + (k - .5) * w, x.gap * scale, w, yerr=[(x.gap - x.ci_lo) * scale, (x.ci_hi - x.gap) * scale], capsize=3, label=met)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Gap (English - Kazakh), " + ("percentage points" if task == "classification" else "metric points")); ax.set_title(task.capitalize() + ": performance gap, 95% paired-bootstrap CI"); ax.legend()
plt.tight_layout(); plt.savefig(RES / "fig1_gap.png", dpi=200)
# Figure 2: confusion matrices for each LLM (English vs Kazakh)
import glob, os
files = sorted(glob.glob(str(RES / "confusion_*_eng.csv")))
files = [f for f in files if "baseline" not in f]
if files:
    fig, axes = plt.subplots(len(files), 2, figsize=(10, 4.2 * len(files)), squeeze=False)
    for r, f in enumerate(files):
        m = os.path.basename(f)[len("confusion_"):-len("_eng.csv")]
        for c, L in enumerate(("eng", "kaz")):
            cm = pd.read_csv(RES / f"confusion_{m}_{L}.csv", index_col=0).values; lab = LABELS + (["INVALID"] if cm.shape[0] > len(LABELS) else [])
            ax = axes[r][c]; ax.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=7); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=7)
            ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(f"{m} - {'English' if L=='eng' else 'Kazakh'} (pooled runs)")
    plt.tight_layout(); plt.savefig(RES / "fig2_confusion.png", dpi=200)
# Figure 3: token inflation (Kazakh vs English prompt tokens)
E = pd.read_csv(RES / "efficiency.csv"); E = E[E.task == "classification"].pivot(index="model", columns="cond", values="prompt_tokens").dropna()
if len(E):
    (E.kaz / E.eng).plot.bar(figsize=(6, 3.5)); plt.axhline(1, color="k", lw=.8); plt.ylabel("prompt tokens, Kazakh / English"); plt.title("Token inflation (classification prompts)")
    plt.tight_layout(); plt.savefig(RES / "fig3_tokens.png", dpi=200)
