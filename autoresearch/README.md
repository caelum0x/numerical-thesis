# Portfolio AutoResearch

Autonomous experiment engine for the thesis project:

**Macroeconomic Factor-Based Dynamic Portfolio Optimization with Autonomous Research and Swarm Intelligence**

This subsystem adapts the autoresearch idea to portfolio research. Instead of training a language model, it runs reproducible machine-learning portfolio experiments against a fixed evaluation harness. Results are consumed by `thesis_portfolio_opt/src/integration/autoresearch_bridge.py` and by the end-to-end orchestrator in `thesis_portfolio_opt/run_all.py`.

## What It Does

- Trains per-asset return prediction models for 12 ETFs.
- Uses macro and engineered market features prepared by the main thesis pipeline.
- Evaluates each configuration with strict temporal separation:
  - Train: 2005-2021
  - Out-of-sample: 2022-2024
  - Prediction horizon: 21 trading days
- Writes experiment results to CSV/TSV files used by the integrated pipeline.

## Files

| File | Purpose |
|---|---|
| `prepare.py` | Fixed data loading, feature preparation, benchmark, and OOS evaluation harness. Do not modify during experiments. |
| `train.py` | Experiment configuration and training logic. The `EXPERIMENT CONFIGURATION` block is the normal edit target. |
| `program.md` | Operating instructions for autonomous experiment loops. |
| `batch_results.csv` | Baseline batch results. |
| `advanced_batch_results.csv` | Advanced search results. |
| `extended_results.csv` | Extended experiment set. |
| `round2_results.csv` | Feedback-loop follow-up experiments. |
| `feedback/latest.json` | Suggestions from the integrated feedback loop. |

## Setup

Use `uv` from this directory:

```bash
uv sync
```

The code loads `.env` from the thesis root and from `thesis_portfolio_opt/.env`. A FRED key is only needed when fetching fresh macro data:

```bash
FRED_API_KEY=your_key_here
```

Cached data already lives in `../thesis_portfolio_opt/data`.

## Run

Single configured experiment:

```bash
uv run python train.py
```

Batch experiments:

```bash
uv run python train.py --batch
```

Prepare or refresh data:

```bash
uv run python prepare.py
```

The top-level integrated flow can also call this subsystem:

```bash
cd ../thesis_portfolio_opt
./venv/bin/python run_all.py --autoresearch
```

## Outputs

`train.py` prints a machine-readable line:

```text
RESULT: sharpe=X.XXXX ic=X.XXXX dir_acc=X.XXXX ann_return=X.XXXX max_dd=X.XXXX description="..."
```

Batch mode writes:

- `batch_results.csv`
- appended rows in `results.tsv`

The bridge merges all result files and selects the highest Sharpe experiment. Current thesis findings show LightGBM with macro-only features and higher max-weight limits as the dominant family, with SVR emerging as a strong secondary model by IC.

## Current Thesis Result Snapshot

Across the recorded AutoResearch runs:

- 72 total experiments across baseline, advanced, extended, and feedback rounds.
- Best AutoResearch experiment: `D13_lgbm_maxw60_tc3`, Sharpe about `0.938`.
- Best integrated strategy: ML-only, Sharpe about `1.33` at 10 bps transaction costs and `1.41` at 3 bps.
- Swarm overlay reduces risk but lowers Sharpe, so it is reported as a beneficial risk overlay rather than a return enhancer.

## Experiment Discipline

Keep the evaluation harness stable. During autonomous research:

- Modify `train.py`, primarily the experiment configuration.
- Do not modify `prepare.py` unless fixing infrastructure.
- Keep train/OOS dates fixed.
- Log every completed experiment.
- Prefer simple configurations when Sharpe differences are small.
