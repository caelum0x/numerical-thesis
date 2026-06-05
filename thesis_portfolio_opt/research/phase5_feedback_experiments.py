"""
Phase 5 feedback-driven experiments.

Runs the three next experiments suggested by feedback/iteration_006.json:
1. Lower risk aversion with MiroFish overlay.
2. Tighter max-weight / shrinkage robustness.
3. IC-weighted ensemble using available OOS return files.

The first two reuse the integrated backtest helper, so they stay aligned with
the thesis methodology. The ensemble is a strategy-level return blend because
the saved artifacts do not include every AutoResearch model bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR, TICKER_LIST
from src.integration._backtest_helpers import (
    compute_metrics,
    load_best_models,
    run_integrated_backtest,
)
from src.integration.autoresearch_bridge import AutoResearchBridge
from src.integration.mirofish_bridge import MiroFishBridge


def _load_returns(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    series = df.iloc[:, 0].astype(float)
    series.name = path.stem
    return series


def _available_return_file(experiment: str) -> Path | None:
    candidates = [
        RESULTS_DIR / f"adv_{experiment}.csv",
        RESULTS_DIR / f"oos_{experiment}.csv",
        RESULTS_DIR / f"{experiment}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_ic_weighted_return_ensemble(top_n: int = 3) -> tuple[pd.Series, dict]:
    """Blend available strategy return files by positive OOS IC."""
    frames = [AutoResearchBridge().get_all_results()]
    all_experiments_path = RESULTS_DIR / "all_experiments.csv"
    if all_experiments_path.exists():
        frames.append(pd.read_csv(all_experiments_path))
    experiments = pd.concat(frames, ignore_index=True)
    candidates = []

    for _, row in experiments.iterrows():
        ic = row.get("ic")
        sharpe = row.get("sharpe", 0)
        if pd.isna(ic) or ic <= 0 or sharpe <= 0:
            continue

        name = str(row.get("experiment", ""))
        path = _available_return_file(name)
        if path is None:
            continue

        candidates.append(
            {
                "experiment": name,
                "ic": float(ic),
                "sharpe": float(sharpe),
                "path": path,
                "returns": _load_returns(path),
            }
        )

    if not candidates:
        raise RuntimeError("No IC-positive experiments with available return files found.")

    selected = sorted(candidates, key=lambda item: item["ic"], reverse=True)[:top_n]
    weights = np.array([item["ic"] for item in selected], dtype=float)
    weights = weights / weights.sum()

    aligned = pd.concat([item["returns"] for item in selected], axis=1).dropna()
    ensemble = aligned.mul(weights, axis=1).sum(axis=1)
    ensemble.name = "phase5_ic_weighted_top3"

    metadata = {
        "ensemble_members": ", ".join(
            f"{item['experiment']} (IC={item['ic']:.3f}, w={weight:.2f})"
            for item, weight in zip(selected, weights)
        )
    }
    return ensemble, metadata


def main() -> None:
    prices = pd.read_csv(RAW_DIR / "prices.csv", index_col=0, parse_dates=True).ffill().bfill()
    features = pd.read_csv(PROCESSED_DIR / "features.csv", index_col=0, parse_dates=True)
    daily_ret = prices[prices.index >= pd.Timestamp("2022-01-01")].pct_change().dropna()
    tickers = list(prices.columns)

    models = load_best_models(RESULTS_DIR, tickers)
    ar_best = AutoResearchBridge().get_best_config()
    mf_bridge = MiroFishBridge()
    features_swarm = mf_bridge.inject_features(features) if mf_bridge.is_available else features

    if ar_best is None:
        raise RuntimeError("No AutoResearch best config available.")

    experiments: list[dict] = []

    configs = [
        {
            "strategy": "baseline_ml_only_current",
            "notes": "Current AutoResearch best config, no swarm overlay.",
            "features": features,
            "risk_scale_fn": None,
            "risk_aversion": ar_best.risk_aversion,
            "max_weight": ar_best.max_weight,
            "rebalance_freq": ar_best.rebalance_freq,
            "tc_bps": ar_best.tc_bps,
            "shrinkage": ar_best.shrinkage,
        },
        {
            "strategy": "phase5_swarm_low_lambda",
            "notes": "MiroFish risk overlay with lower risk aversion to recover return.",
            "features": features_swarm,
            "risk_scale_fn": mf_bridge.get_risk_scale_at if mf_bridge.is_available else None,
            "risk_aversion": 2.0,
            "max_weight": 0.5,
            "rebalance_freq": ar_best.rebalance_freq,
            "tc_bps": ar_best.tc_bps,
            "shrinkage": ar_best.shrinkage,
        },
        {
            "strategy": "phase5_tight_constraints",
            "notes": "Tighter max-weight and shrinkage robustness test.",
            "features": features,
            "risk_scale_fn": None,
            "risk_aversion": ar_best.risk_aversion,
            "max_weight": 0.3,
            "rebalance_freq": ar_best.rebalance_freq,
            "tc_bps": ar_best.tc_bps,
            "shrinkage": 0.2,
        },
    ]

    for cfg in configs:
        returns = run_integrated_backtest(
            prices=prices,
            daily_ret=daily_ret,
            models=models,
            features=cfg["features"],
            tickers=tickers,
            risk_aversion=cfg["risk_aversion"],
            max_weight=cfg["max_weight"],
            rebalance_freq=cfg["rebalance_freq"],
            tc_bps=cfg["tc_bps"],
            shrinkage=cfg["shrinkage"],
            risk_scale_fn=cfg["risk_scale_fn"],
        )
        metrics = compute_metrics(returns)
        metrics.update(
            {
                "strategy": cfg["strategy"],
                "risk_aversion": cfg["risk_aversion"],
                "max_weight": cfg["max_weight"],
                "tc_bps": cfg["tc_bps"],
                "shrinkage": cfg["shrinkage"],
                "notes": cfg["notes"],
            }
        )
        experiments.append(metrics)

    ensemble_returns, ensemble_meta = run_ic_weighted_return_ensemble()
    ensemble_metrics = compute_metrics(ensemble_returns)
    ensemble_metrics.update(
        {
            "strategy": "phase5_ic_weighted_top3",
            "risk_aversion": np.nan,
            "max_weight": np.nan,
            "tc_bps": np.nan,
            "shrinkage": np.nan,
            "notes": ensemble_meta["ensemble_members"],
        }
    )
    experiments.append(ensemble_metrics)

    results = pd.DataFrame(experiments).set_index("strategy")
    results = results[
        [
            "sharpe",
            "sortino",
            "ann_return",
            "ann_vol",
            "max_dd",
            "calmar",
            "total_return",
            "risk_aversion",
            "max_weight",
            "tc_bps",
            "shrinkage",
            "notes",
        ]
    ]

    out_csv = RESULTS_DIR / "phase5_feedback_experiments.csv"
    out_tex = RESULTS_DIR / "table_phase5_feedback.tex"
    results.to_csv(out_csv)
    results.to_latex(
        out_tex,
        float_format="%.3f",
        caption="Phase 5 Feedback-Driven Experiments (OOS 2022--2024)",
        label="tab:phase5_feedback",
    )

    print(results[["sharpe", "ann_return", "ann_vol", "max_dd", "notes"]])
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_tex}")


if __name__ == "__main__":
    main()
