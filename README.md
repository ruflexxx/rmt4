# Kazakh–English LLM Performance in Translation and Topic Classification: A Paired Benchmark

Reproducibility package for the research manuscript by **Aisultan Kalibek, Radmir Toizhanov, and Torekhan Kalin** (Astana IT University, group SE-2417).

Final project release. Tagged release: `final`.

## Structure

- `src/common.py` — frozen settings (`seed=42`, sample sizes, runs, `temperature=0`), English prompts, data loaders, and label parser
- `src/run_experiment.py` — calls all hosted LLMs (resumable, caches every raw output)
- `src/run_nllb.py` — strong translation baseline (`NLLB-200-distilled-600M`, local, beam width 4, deterministic)
- `src/analyze.py` — computes all reported metrics, paired bootstrap CIs, and Holm corrections into `results/*.csv`
- `src/make_figures.py` — generates manuscript figures into `results/*.png`
- `results/` — raw outputs and derived result tables/figures
- `log_*.txt` — available execution logs, including quota errors and failed API calls for replaced models

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
```

Recorded execution environment: **Python 3.10**, `sacrebleu==2.6.0`, `scikit-learn==1.6.1` (full details in `results/environment.json`).

## Data

The datasets are not redistributed in this repository and remain under their original licenses (CC BY-SA 4.0).

1. Download the official FLORES-200 archive (NLLB Team, 2022) and unpack it so that the following files exist:

   - `data/flores200_dataset/devtest/eng_Latn.devtest`
   - `data/flores200_dataset/devtest/kaz_Cyrl.devtest`

   **Important:** `src/analyze.py` requires these local FLORES-200 `devtest` files. It loads the English and Kazakh source sentences and reference translations from disk when calculating BLEU, chrF, and length-subgroup results. The reference translations are **not** embedded in `results/raw_outputs.jsonl`. Therefore, download and unpack FLORES-200 before running the analysis script.

2. SIB-200 is downloaded automatically via Hugging Face `datasets` from `Davlan/sib200`, configurations `eng_Latn` and `kaz_Cyrl`. An internet connection is required on the first run.

## Reproduce the tables and figures

After installing the requirements and placing the FLORES-200 files in the paths listed above, run:

```bash
python src/analyze.py       # N_BOOT=1000 by default
python src/make_figures.py
```

No API keys are required for this analysis-only reproduction. The scripts use the committed `results/raw_outputs.jsonl`, but `analyze.py` also needs the local FLORES-200 devtest files and downloads SIB-200 through Hugging Face if it is not cached.

| Manuscript item | File |
|---|---|
| Table 3 (classification), Table 4 (translation) | `results/main_results.csv` |
| Table 5 (paired gaps, 95% bootstrap CI, Holm p) | `results/gap_analysis.csv` |
| McNemar exact tests (Section 4.3) | `results/mcnemar.csv` |
| Fig. 1 (paired performance gaps) | `results/fig1_gap.png` |
| Fig. 2 (confusion matrices) | `results/fig2_confusion.png`, `results/confusion_*.csv` |
| Per-class F1, length subgroups, token efficiency (Section 4.4) | `results/per_class_f1.csv`, `results/length_subgroup.csv`, `results/efficiency.csv` |
| qwen3.8-27b stability check (Section 4.4) | `results/extra_runs_stability.csv` |
| Error candidates for manual audit (Section 4.4) | `results/error_candidates.csv` |
| Sample IDs, data hashes, environment | `results/sample_ids.json`, `results/data_hashes.json`, `results/environment.json` |

`results/error_candidates.csv` contains candidate cases for manual review and includes a `manual_error_type` column. The presence of this file does not mean that a complete manual annotation draft or all 36 manually reviewed examples are included; consult the manuscript's annotation statement and the actual contents of the CSV when interpreting the manual-audit results.

## Re-run the model calls (optional; requires API keys)

Set the relevant API keys in your local environment. Never commit API keys or credentials.

```bash
# macOS / Linux:
export GROQ_API_KEY=...
export GEMINI_API_KEY=...

# Windows PowerShell:
# $env:GROQ_API_KEY="..."
# $env:GEMINI_API_KEY="..."

python src/run_experiment.py --dry
python src/run_experiment.py --tasks cls --models gpt-oss-120b
python src/run_experiment.py --tasks trans --models gpt-oss-120b
python src/run_nllb.py
python src/analyze.py
python src/make_figures.py
```

The `--dry` command prints the expected call counts and prepares sample-ID and data-hash files. Hosted model calls require valid provider API keys. The NLLB translation baseline can run locally; GPU availability affects performance, not the documented model configuration.

## Final executed settings

- **Seed and sampling:** `seed=42`; 100 randomly selected FLORES-200 devtest sentence pairs; full SIB-200 test split (204 aligned items per language).
- **Sample checksum:** `results/sample_ids.json` SHA-256: `7c559f083c80ed476b67e0079aeec1e00c233e5a758517413d7390e756e44328`.
- **Inference settings:** 1 main run per model and condition; `temperature=0`; maximum 1,024 output tokens; identical English zero-shot prompt templates in `src/common.py`.
- **Statistical analysis:** 1,000 paired-bootstrap resamples with Holm adjustment across 14 planned tests.
- **Evaluated systems:** `gpt-oss-120b` and `qwen3.8-27b` (via Groq), `gemini-3.5-flash-lite` (via Gemini API).
- **Baselines:** random (seed=42), majority class, and character n-gram TF-IDF + logistic regression (classification); copy-source and `NLLB-200-distilled-600M` (translation).
- **Recorded environment:** Python 3.10, `sacrebleu==2.6.0`, `scikit-learn==1.6.1`.
- **Run date:** 8 November 2026.

## Deviations from the frozen protocol

- `llama-3.3-70b` and `gemini-2.5-flash-lite` were replaced by `qwen3.8-27b` and `gemini-3.5-flash-lite` after API endpoint or quota failures (see the available `log_*.txt` files, including `log_llama.txt`).
- The translation sample size was reduced from 500 to 100 sentence pairs due to API rate and token quotas; translation results are treated as exploratory.
- Classification was executed once per model instead of three times (`temperature=0`); an additional `qwen3.8-27b` run is retained solely as a stability check.

## License, Privacy, and Data Statement

- **Code:** MIT License (see `LICENSE`).
- **Datasets:** FLORES-200 and SIB-200 remain under their original CC BY-SA 4.0 licenses and are not redistributed in this repository.
- **Privacy and credentials:** No personal data, API keys, credentials, or confidential material are included in this repository.