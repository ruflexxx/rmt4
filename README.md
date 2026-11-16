# Assignment 4 - Kazakh vs English performance gap of LLMs (translation + topic classification)

Research methods and tools, SE-2417. Final project release. Tagged release: `final`.

Authors: Aisultan Kalibek, Radmir Toizhanov, Torekhan Kalin (Astana IT University).

## Structure

- `src/common.py` - frozen settings (seed, N, runs, temperature), prompts, data loaders, label parser
- `src/run_experiment.py` - calls all LLMs (resumable, caches every raw output)
- `src/run_nllb.py` - strong translation baseline (NLLB-200-distilled-600M, local, deterministic)
- `src/analyze.py` - computes ALL reported numbers into `results/*.csv`
- `src/make_figures.py` - figures into `results/*.png`
- `results/` - raw outputs and result files (every number in the manuscript traces to a file here)
- `log_*.txt` - run logs, including quota errors and failed calls of the replaced models

## Installation

```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data (not redistributed; CC BY-SA 4.0)

1. Download the FLORES-200 archive from the official source (NLLB Team, 2022) and unpack it so that `data/flores200_dataset/devtest/eng_Latn.devtest` and `data/flores200_dataset/devtest/kaz_Cyrl.devtest` exist.
2. SIB-200 is downloaded automatically from Hugging Face (`Davlan/sib200`, configs `eng_Latn`, `kaz_Cyrl`).

## Reproduce the tables and figures

Without API keys, `analyze.py` and `make_figures.py` reproduce every table and figure from the committed `results/raw_outputs.jsonl`:

```
python src/analyze.py                       # N_BOOT=1000 by default
python src/make_figures.py
```

| Manuscript item | File |
| --- | --- |
| Table 3 (classification), Table 4 (translation) | `results/main_results.csv` |
| Table 5 (paired gaps, bootstrap CI, Holm p) | `results/gap_analysis.csv` |
| McNemar tests (Section 4.3) | `results/mcnemar.csv` |
| Fig. 1 (paired gaps) | `results/fig1_gap.png` |
| Fig. 2 (confusion matrices) | `results/fig2_confusion.png`, `results/confusion_*.csv` |
| Per-class F1, length subgroups, efficiency (Section 4.4) | `results/per_class_f1.csv`, `results/length_subgroup.csv`, `results/efficiency.csv` |
| qwen stability check (Section 4.4) | `results/extra_runs_stability.csv` |
| Error candidates for the manual audit | `results/error_candidates.csv` |
| Sample ids, data hashes, environment | `results/sample_ids.json`, `results/data_hashes.json`, `results/environment.json` |

## Re-run the model calls (optional, needs API keys)

```
export GROQ_API_KEY=...      # never commit keys
export GEMINI_API_KEY=...
python src/run_experiment.py --dry          # prints number of calls; writes sample_ids.json + data_hashes.json
python src/run_experiment.py --tasks cls --models gpt-oss-120b      # one process per model
python src/run_experiment.py --tasks trans --models gpt-oss-120b
python src/run_nllb.py                      # translation baseline (GPU optional)
python src/analyze.py
python src/make_figures.py
```

## Final executed settings

seed 42 - 100 random FLORES-200 devtest sentences (ids in `results/sample_ids.json`) - SIB-200 test split (204 items per language) - 1 run per model and condition - temperature 0 - max 1,024 tokens - 1,000 paired-bootstrap resamples - identical English prompts (see `src/common.py`).

Models: gpt-oss-120b and qwen3.8-27b (via Groq), gemini-3.5-flash-lite (Gemini API). Baselines: random, majority class, TF-IDF + logistic regression (classification); copy-source and NLLB-200-distilled-600M (translation).

Run date: 8 November 2026. Environment: see `results/environment.json`.

## Deviations from the frozen protocol

- llama-3.3-70b and gemini-2.5-flash-lite were replaced by qwen3.8-27b and gemini-3.5-flash-lite after API errors (see `log_llama.txt` and the quota-error logs).
- Translation was reduced from 500 to 100 sentences because of API quotas (exploratory).
- Classification was run once per model instead of three times (temperature 0).

## License

Code: MIT (see `LICENSE`). FLORES-200 and SIB-200 keep their own licenses and are not redistributed here.