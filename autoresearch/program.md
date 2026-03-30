# Portfolio AutoResearch

Adapted from Karpathy's autoresearch for macro-based portfolio optimization.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar28`).
2. **Read the in-scope files**: The repo is small:
   - `README.md` — context
   - `prepare.py` — fixed data prep, feature engineering, evaluation, benchmarks. **Do not modify.**
   - `train.py` — the file you modify. Model, features, optimization parameters.
3. **Verify data exists**: Check that `../thesis_portfolio_opt/data/raw/prices.csv` and `macro.csv` exist. If not, run `python prepare.py`.
4. **Initialize results.tsv**: Create `results.tsv` with the header row if it doesn't exist.
5. **Confirm and go**: Confirm setup looks good.

## Experimentation

Each experiment trains ML models and runs a strict out-of-sample backtest (train ≤2021, test 2022-2024). Total runtime should be under 2 minutes.

**What you CAN do:**
- Modify `train.py` — the EXPERIMENT CONFIGURATION section. Everything is fair game: model type, hyperparameters, feature selection, optimization parameters, shrinkage, rebalancing frequency.

**What you CANNOT do:**
- Modify `prepare.py`. It contains fixed evaluation, data loading, and backtesting.
- Install new packages.
- Access test data during training.

**The goal: get the highest OOS Sharpe ratio.**

**Benchmarks** (these are your targets):
- Equal Weight: Sharpe ≈ 0.20
- 60/40: Sharpe ≈ 0.56
- SPY: Sharpe ≈ 0.67

**Simplicity criterion**: All else equal, simpler is better. A small Sharpe improvement that adds ugly complexity is not worth it.

## Output format

The script prints:

```
RESULT: sharpe=X.XXXX ic=X.XXXX dir_acc=X.XXXX ann_return=X.XXXX max_dd=X.XXXX description="..."
```

Extract with: `grep "^RESULT:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
experiment	sharpe	ic	dir_acc	status	description
```

Status: `keep`, `discard`, or `crash`.

## The experiment loop

LOOP FOREVER:

1. Look at current `train.py` and `results.tsv`
2. Propose a hypothesis
3. Edit the EXPERIMENT section of `train.py`
4. Run: `python train.py > run.log 2>&1`
5. Read: `grep "^RESULT:" run.log`
6. Log to `results.tsv`
7. If Sharpe improved over best-so-far → keep
8. If not → revert `train.py`
9. Repeat

**NEVER STOP**. The user may be sleeping. Run until manually interrupted.

## Ideas to try

1. Different models: Lasso, Ridge, ElasticNet, XGBoost, LightGBM, RF, SVR
2. Macro-only vs momentum-only vs all features
3. Feature selection: top-K by mutual information
4. PCA dimensionality reduction (164 → 20 components)
5. Different prediction horizons: 5d, 21d, 63d
6. Risk aversion sweep: λ = 1, 2, 5, 10, 20
7. Shrinkage toward historical mean: α = 0.1, 0.3, 0.5, 0.7
8. Ensemble: average predictions from 3+ models
9. Regime-conditional: separate models for high/low VIX
10. Walk-forward retraining: retrain every quarter using rolling window
11. Non-linear features: VIX × momentum, log(VIX), momentum²
12. Transaction cost sensitivity: 5, 10, 20, 50 bps
13. Rebalancing frequency: weekly vs monthly vs quarterly
14. Max weight constraint: 0.2, 0.3, 0.4, 0.5
15. Target rank prediction instead of raw returns
