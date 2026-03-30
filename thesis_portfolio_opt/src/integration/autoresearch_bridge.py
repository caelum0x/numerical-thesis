"""
AutoResearch → Pipeline Bridge
================================
Reads autoresearch batch_results.csv, extracts the best model configuration,
and returns parameters that the thesis pipeline can use for walk-forward
backtesting.

Flow:
    autoresearch/batch_results.csv  →  parse best experiment
    autoresearch/train.py config    →  extract model class + hyperparams
                                    →  dict consumed by trainer / backtester
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

AUTORESEARCH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'autoresearch')
)


@dataclass(frozen=True)
class BestModelConfig:
    """Immutable snapshot of the best autoresearch experiment."""
    experiment_name: str
    sharpe: float
    ann_return: float
    max_dd: float
    ic: float
    dir_acc: float
    description: str
    # Parsed hyperparams
    model_type: str          # e.g. "lgbm", "lasso", "ensemble"
    risk_aversion: float
    max_weight: float
    rebalance_freq: int
    tc_bps: int
    shrinkage: float
    feature_set: str         # "macro", "all", "momentum"


def _parse_experiment_name(name: str, description: str) -> dict:
    """Extract hyperparameters from experiment name and description.

    Autoresearch naming convention:
        B2_lgbm_macro         → model=lgbm, features=macro
        B12_lgbm_macro_maxw50 → model=lgbm, features=macro, max_weight=0.5
        A1_lgbm_maxw50_tc5    → model=lgbm, max_weight=0.5, tc=5
    """
    name_lower = str(name).lower()
    desc_lower = str(description).lower()

    # Model type
    model_type = "lgbm"  # default
    for m in ["lasso", "ridge", "xgb", "rf", "elastic", "ensemble", "lgbm"]:
        if m in name_lower or m in desc_lower:
            model_type = m
            break

    # Feature set
    feature_set = "all"
    if "macro" in name_lower or "macro" in desc_lower:
        feature_set = "macro"
    elif "mom" in name_lower or "momentum" in desc_lower:
        feature_set = "momentum"

    # Max weight
    max_weight = 0.35
    mw_match = re.search(r'maxw(\d+)', name_lower)
    if mw_match:
        max_weight = int(mw_match.group(1)) / 100.0

    # Transaction cost
    tc_bps = 10
    tc_match = re.search(r'tc(\d+)', name_lower)
    if tc_match:
        tc_bps = int(tc_match.group(1))

    # Risk aversion from description
    risk_aversion = 5.0
    lam_match = re.search(r'[λlam]=?(\d+\.?\d*)', desc_lower)
    if lam_match:
        risk_aversion = float(lam_match.group(1))

    # Shrinkage
    shrinkage = 0.0
    sh_match = re.search(r'shrink=?(\d+\.?\d*)', desc_lower)
    if sh_match:
        val = float(sh_match.group(1))
        shrinkage = val / 100.0 if val > 1 else val
    sh_match2 = re.search(r'shrink(\d+)', name_lower)
    if sh_match2:
        shrinkage = int(sh_match2.group(1)) / 100.0

    # Rebalance frequency
    rebalance_freq = 21
    if "weekly" in desc_lower:
        rebalance_freq = 5
    elif "quarterly" in desc_lower:
        rebalance_freq = 63

    return {
        "model_type": model_type,
        "risk_aversion": risk_aversion,
        "max_weight": max_weight,
        "rebalance_freq": rebalance_freq,
        "tc_bps": tc_bps,
        "shrinkage": shrinkage,
        "feature_set": feature_set,
    }


class AutoResearchBridge:
    """Reads autoresearch results and provides best config to the pipeline."""

    def __init__(self, autoresearch_dir: str = AUTORESEARCH_DIR):
        self.autoresearch_dir = autoresearch_dir
        self._batch_path = os.path.join(autoresearch_dir, 'batch_results.csv')
        self._advanced_path = os.path.join(autoresearch_dir, 'advanced_batch_results.csv')

    def get_all_results(self) -> pd.DataFrame:
        """Load and merge all experiment results."""
        frames = []
        for path in [self._batch_path, self._advanced_path]:
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['source'] = os.path.basename(path)
                frames.append(df)

        if not frames:
            logger.warning("No autoresearch results found at %s", self.autoresearch_dir)
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values('sharpe', ascending=False).reset_index(drop=True)
        return combined

    def get_best_config(self, min_sharpe: float = 0.0) -> Optional[BestModelConfig]:
        """Return the best experiment config as an immutable dataclass.

        Parameters
        ----------
        min_sharpe : float
            Minimum Sharpe ratio to consider viable.  If best is below this,
            returns None (signals: don't trust autoresearch output).
        """
        results = self.get_all_results()
        if results.empty:
            return None

        best = results.iloc[0]
        if best['sharpe'] < min_sharpe:
            logger.info("Best Sharpe %.3f < threshold %.3f — skipping", best['sharpe'], min_sharpe)
            return None

        parsed = _parse_experiment_name(
            best.get('experiment', ''),
            best.get('description', ''),
        )

        return BestModelConfig(
            experiment_name=str(best.get('experiment', '')),
            sharpe=float(best.get('sharpe', 0)),
            ann_return=float(best.get('ann_return', 0)),
            max_dd=float(best.get('max_dd', 0)),
            ic=float(best.get('ic', 0)),
            dir_acc=float(best.get('dir_acc', 0)),
            description=str(best.get('description', '')),
            **parsed,
        )

    def get_top_n(self, n: int = 5) -> list[BestModelConfig]:
        """Return top N experiment configs for ensemble or comparison."""
        results = self.get_all_results()
        configs = []
        for _, row in results.head(n).iterrows():
            parsed = _parse_experiment_name(
                row.get('experiment', ''),
                row.get('description', ''),
            )
            configs.append(BestModelConfig(
                experiment_name=str(row.get('experiment', '')),
                sharpe=float(row.get('sharpe', 0)),
                ann_return=float(row.get('ann_return', 0)),
                max_dd=float(row.get('max_dd', 0)),
                ic=float(row.get('ic', 0)),
                dir_acc=float(row.get('dir_acc', 0)),
                description=str(row.get('description', '')),
                **parsed,
            ))
        return configs

    def summary(self) -> str:
        """Human-readable summary for logging."""
        results = self.get_all_results()
        if results.empty:
            return "AutoResearch: No results found."

        best = self.get_best_config()
        n_total = len(results)
        n_positive = len(results[results['sharpe'] > 0])
        lines = [
            f"AutoResearch: {n_total} experiments, {n_positive} with positive Sharpe",
            f"  Best: {best.experiment_name} — Sharpe={best.sharpe:.3f}, "
            f"Return={best.ann_return:.1%}, MaxDD={best.max_dd:.1%}",
            f"  Config: model={best.model_type}, features={best.feature_set}, "
            f"λ={best.risk_aversion}, maxW={best.max_weight}, "
            f"shrinkage={best.shrinkage}",
        ] if best else ["AutoResearch: No viable experiments."]
        return "\n".join(lines)
