# Assignment 4 — Kazakh vs English performance gap of LLMs (translation + topic classification)

Research methods and tools, SE-2417. Tagged release: `assignment-4`.

## Structure
- `src/common.py` — frozen settings (seed, N, runs, temperature), prompts, data loaders, label parser
- `src/run_experiment.py` — calls all LLMs (resumable, caches every raw output)
- `src/run_nllb.py` — strong translation baseline (NLLB-200-distilled-600M, local, deterministic)
- `src/analyze.py` — computes ALL reported numbers → `results/*.csv`
- `src/make_figures.py` — Figures 1–3 → `results/*.png`
- `results/` — raw outputs and result files (every number in the report traces to a file here)

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
## Data (not redistributed; CC BY-SA 4.0)
1. Download the FLORES-200 archive from the official source (NLLB Team, 2022) and unpack so that
   `data/flores200_dataset/devtest/eng_Latn.devtest` and `kaz_Cyrl.devtest` exist (also `dev/` for pilots).
2. SIB-200 is downloaded automatically from Hugging Face (`Davlan/sib200`, configs `eng_Latn`, `kaz_Cyrl`).

## Reproduce the main results table
```bash
export GROQ_API_KEY=...      # never commit keys
export GEMINI_API_KEY=...
python src/run_experiment.py --dry          # prints number of calls; writes sample_ids.json + data_hashes.json (use for Appendix A BEFORE the final run)
python src/run_experiment.py --tasks cls --models gpt-oss-120b   # one process per model, run them in parallel    # classification first (primary outcome, cheapest); then: --tasks trans
python src/run_nllb.py                      # strong translation baseline (GPU optional)
python src/analyze.py                       # N_BOOT=1000 by default
python src/make_figures.py
```
Without API keys, `analyze.py` + `make_figures.py` reproduce every table from the committed `results/raw_outputs.jsonl`.

## Frozen settings
seed 42 · 500 random FLORES-200 devtest sentences · SIB-200 test split (204 items per language) · classification 3 runs / translation 1 run · temperature 0 ·
max 1,024 tokens · 1,000 paired-bootstrap resamples · identical English prompts (see `common.py`, Appendix D of Assignment 3).
Any change must be listed under "Deviations from the frozen protocol" in the report.

## Manual step
`results/error_candidates.csv` lists failed cases; fill the column `manual_error_type` (≥30 cases) → Appendix B of the report.
