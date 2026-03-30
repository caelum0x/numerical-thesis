"""
Feedback Loop — closes the autoresearch ↔ pipeline ↔ MiroFish cycle.

After the integrated backtest runs, this module:

1. Writes pipeline results back to autoresearch/ as a feedback file
   so the next autoresearch iteration knows:
   - Which config won the integrated test (not just the standalone OOS)
   - How the MiroFish overlay affected performance
   - Which directions to explore next

2. Updates autoresearch/program.md with data-driven "Ideas to try"
   based on what worked and what didn't

3. Generates a next_iteration.json consumed by train.py --batch
   to seed the next round of autonomous experiments

Flow:
    integrated_comparison.csv  →  analyze what worked
    autoresearch results       →  identify gaps
                               →  feedback.json (for next autoresearch run)
                               →  updated program hints
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AUTORESEARCH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'autoresearch')
)
RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results')
)


def generate_feedback(
    integrated_results: pd.DataFrame,
    autoresearch_results: pd.DataFrame,
    mirofish_summary: dict,
) -> dict:
    """Analyze integrated results and generate feedback for next iteration.

    Returns a dict with:
        - winning_config: what worked best in the full integrated test
        - swarm_impact: how MiroFish overlay affected performance
        - gaps: what hasn't been tried yet
        - next_experiments: concrete suggestions for next autoresearch batch
    """
    feedback = {
        'generated_at': datetime.now().isoformat(),
        'iteration': _get_iteration_count() + 1,
    }

    # --- What won the integrated test ---
    if not integrated_results.empty:
        best_strategy = integrated_results['sharpe'].idxmax()
        best_sharpe = float(integrated_results.loc[best_strategy, 'sharpe'])
        spy_sharpe = float(integrated_results.loc['SPY', 'sharpe']) if 'SPY' in integrated_results.index else 0

        feedback['winning_strategy'] = {
            'name': best_strategy,
            'sharpe': best_sharpe,
            'beats_spy_by': round(best_sharpe - spy_sharpe, 4),
        }

    # --- MiroFish swarm impact ---
    if not integrated_results.empty:
        ml_only = integrated_results.loc['ML_Only'] if 'ML_Only' in integrated_results.index else None
        ml_swarm = integrated_results.loc['ML_Swarm'] if 'ML_Swarm' in integrated_results.index else None

        if ml_only is not None and ml_swarm is not None:
            sharpe_delta = float(ml_swarm['sharpe'] - ml_only['sharpe'])
            dd_improvement = float(ml_swarm['max_dd'] - ml_only['max_dd'])
            vol_reduction = float(ml_swarm['ann_vol'] - ml_only['ann_vol'])

            feedback['swarm_impact'] = {
                'sharpe_delta': round(sharpe_delta, 4),
                'drawdown_improvement': round(dd_improvement, 4),
                'volatility_reduction': round(vol_reduction, 4),
                'verdict': (
                    'beneficial_risk' if dd_improvement > 0 and sharpe_delta < 0
                    else 'beneficial_both' if sharpe_delta > 0
                    else 'not_beneficial' if sharpe_delta < -0.1
                    else 'neutral'
                ),
            }

    # --- What autoresearch hasn't tried ---
    if not autoresearch_results.empty:
        tried_models = set()
        for _, row in autoresearch_results.iterrows():
            name = str(row.get('experiment', '')).lower()
            for m in ['lasso', 'ridge', 'xgb', 'lgbm', 'rf', 'elastic', 'ensemble', 'svr']:
                if m in name:
                    tried_models.add(m)

        all_models = {'lasso', 'ridge', 'xgb', 'lgbm', 'rf', 'elastic', 'ensemble', 'svr'}
        untried = all_models - tried_models

        tried_features = set()
        for _, row in autoresearch_results.iterrows():
            name = str(row.get('experiment', '')).lower()
            desc = str(row.get('description', '')).lower()
            if 'macro' in name or 'macro' in desc:
                tried_features.add('macro')
            if 'all' in name or 'all' in desc:
                tried_features.add('all')
            if 'mom' in name or 'momentum' in desc:
                tried_features.add('momentum')
            if 'pca' in name or 'pca' in desc:
                tried_features.add('pca')

        all_features = {'macro', 'all', 'momentum', 'pca', 'technical'}
        untried_features = all_features - tried_features

        feedback['gaps'] = {
            'untried_models': sorted(untried),
            'untried_features': sorted(untried_features),
        }

    # --- Generate next experiment suggestions ---
    suggestions = _generate_suggestions(feedback, autoresearch_results)
    feedback['next_experiments'] = suggestions

    return feedback


def _generate_suggestions(feedback: dict, ar_results: pd.DataFrame) -> list[dict]:
    """Generate concrete experiment suggestions for the next autoresearch batch."""
    suggestions = []

    # 1. If swarm overlay reduced drawdown, try lower risk aversion
    swarm = feedback.get('swarm_impact', {})
    if swarm.get('verdict') == 'beneficial_risk':
        suggestions.append({
            'name': 'lower_risk_aversion_with_swarm',
            'rationale': 'Swarm overlay reduces drawdown — try lower λ to capture more return',
            'params': {'risk_aversion': 2.0, 'max_weight': 0.5},
        })

    # 2. Try untried models
    gaps = feedback.get('gaps', {})
    for model in gaps.get('untried_models', [])[:3]:
        suggestions.append({
            'name': f'try_{model}_macro',
            'rationale': f'{model} not yet tested with macro features',
            'params': {'model_type': model, 'feature_set': 'macro'},
        })

    # 3. If best sharpe > 0.8, try to push further with tighter constraints
    winning = feedback.get('winning_strategy', {})
    if winning.get('sharpe', 0) > 0.8:
        suggestions.append({
            'name': 'push_sharpe_tight_constraints',
            'rationale': f'Current best Sharpe={winning["sharpe"]:.3f} — try tighter turnover/weight constraints',
            'params': {'max_weight': 0.3, 'shrinkage': 0.2},
        })

    # 4. Explore regime-conditional if not tried
    ar_names = [str(r.get('experiment', '')).lower() for _, r in ar_results.iterrows()] if not ar_results.empty else []
    if not any('regime' in n for n in ar_names):
        suggestions.append({
            'name': 'regime_conditional_model',
            'rationale': 'No regime-conditional experiments found — VIX-based model switching',
            'params': {'model_type': 'lgbm', 'feature_set': 'macro', 'regime': True},
        })

    # 5. Ensemble of top-3 if not tried
    if not any('ensemble' in n and 'top' in n for n in ar_names):
        suggestions.append({
            'name': 'ensemble_top3_weighted',
            'rationale': 'Try IC-weighted ensemble of top 3 performing models',
            'params': {'model_type': 'ensemble', 'ensemble_method': 'ic_weighted'},
        })

    return suggestions


def _get_iteration_count() -> int:
    """Count how many feedback files exist (= number of completed iterations)."""
    feedback_dir = os.path.join(AUTORESEARCH_DIR, 'feedback')
    if not os.path.exists(feedback_dir):
        return 0
    return len([f for f in os.listdir(feedback_dir) if f.endswith('.json')])


def write_feedback(feedback: dict) -> str:
    """Write feedback to autoresearch/feedback/ for next iteration.

    Returns path to the written file.
    """
    feedback_dir = os.path.join(AUTORESEARCH_DIR, 'feedback')
    os.makedirs(feedback_dir, exist_ok=True)

    iteration = feedback.get('iteration', 1)
    filename = f'iteration_{iteration:03d}.json'
    path = os.path.join(feedback_dir, filename)

    with open(path, 'w') as f:
        json.dump(feedback, f, indent=2, default=str)

    # Also write latest.json for easy consumption
    latest_path = os.path.join(feedback_dir, 'latest.json')
    with open(latest_path, 'w') as f:
        json.dump(feedback, f, indent=2, default=str)

    logger.info("Wrote feedback: %s", path)
    return path


def run_feedback_loop(
    integrated_results: Optional[pd.DataFrame] = None,
    autoresearch_results: Optional[pd.DataFrame] = None,
    mirofish_summary: Optional[dict] = None,
) -> dict:
    """Run the full feedback loop: analyze → generate → write.

    Can be called standalone or from the orchestrator.
    """
    # Load defaults if not provided
    if integrated_results is None:
        comp_path = os.path.join(RESULTS_DIR, 'integrated_comparison.csv')
        if os.path.exists(comp_path):
            integrated_results = pd.read_csv(comp_path, index_col=0)
        else:
            integrated_results = pd.DataFrame()

    if autoresearch_results is None:
        from src.integration.autoresearch_bridge import AutoResearchBridge
        autoresearch_results = AutoResearchBridge().get_all_results()

    if mirofish_summary is None:
        mirofish_summary = {}

    feedback = generate_feedback(integrated_results, autoresearch_results, mirofish_summary)
    path = write_feedback(feedback)

    # Print summary
    print(f"\n  Feedback Loop — Iteration {feedback.get('iteration', '?')}")
    print(f"  {'=' * 50}")

    winning = feedback.get('winning_strategy', {})
    if winning:
        print(f"  Winner: {winning['name']} (Sharpe={winning['sharpe']:.3f}, "
              f"+{winning['beats_spy_by']:.3f} vs SPY)")

    swarm = feedback.get('swarm_impact', {})
    if swarm:
        print(f"  Swarm impact: {swarm['verdict']} "
              f"(ΔSharpe={swarm['sharpe_delta']:+.3f}, "
              f"ΔDD={swarm['drawdown_improvement']:+.3f})")

    gaps = feedback.get('gaps', {})
    if gaps.get('untried_models'):
        print(f"  Untried models: {', '.join(gaps['untried_models'])}")

    suggestions = feedback.get('next_experiments', [])
    if suggestions:
        print(f"  Next experiments ({len(suggestions)}):")
        for s in suggestions:
            print(f"    → {s['name']}: {s['rationale']}")

    print(f"  Written to: {path}")
    return feedback
