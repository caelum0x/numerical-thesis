"""
MiroFish → Pipeline Bridge
============================
Reads MiroFish multi-agent simulation outputs and provides:

1. **Risk overlay** — agent agreement → position scaling factor
   Low agreement = high uncertainty = scale down positions
   High agreement = consensus = full allocation

2. **Features** — regime detection + ensemble signal as macro features
   Injected into the preprocessor for ML models to consume

3. **Swarm weights** — direct portfolio weights from agent consensus
   Can be used as a standalone strategy or blended with ML optimizer

Flow:
    mirofish_simulation.json  →  agreement time series + regime
    mirofish_weights.json     →  per-round portfolio weights
                              →  risk_scale_factor(date) for optimizer
                              →  swarm_features(date) for preprocessor
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results')
)
MIROFISH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MiroFish')
)


@dataclass(frozen=True)
class SwarmSnapshot:
    """One round of MiroFish swarm data."""
    date: pd.Timestamp
    agreement: float          # 0-1, how much agents agree
    regime: str               # defensive / balanced / risk_on
    vix: float
    ensemble_signal: list     # per-asset signal from agent consensus
    risk_scale: float         # derived: agreement → position scale


class MiroFishBridge:
    """Reads MiroFish simulation outputs for the thesis pipeline."""

    def __init__(
        self,
        simulation_path: Optional[str] = None,
        weights_path: Optional[str] = None,
    ):
        self._sim_path = simulation_path or os.path.join(RESULTS_DIR, 'mirofish_simulation.json')
        self._weights_path = weights_path or os.path.join(RESULTS_DIR, 'mirofish_weights.json')
        self._simulation: Optional[dict] = None
        self._weights: Optional[list] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_simulation(self) -> dict:
        if self._simulation is not None:
            return self._simulation
        if not os.path.exists(self._sim_path):
            raise FileNotFoundError(f"MiroFish simulation not found: {self._sim_path}")
        with open(self._sim_path) as f:
            self._simulation = json.load(f)
        return self._simulation

    def _load_weights(self) -> list:
        if self._weights is not None:
            return self._weights
        if not os.path.exists(self._weights_path):
            raise FileNotFoundError(f"MiroFish weights not found: {self._weights_path}")
        with open(self._weights_path) as f:
            self._weights = json.load(f)
        return self._weights

    @property
    def is_available(self) -> bool:
        """Check if MiroFish data exists."""
        return os.path.exists(self._sim_path)

    # ------------------------------------------------------------------
    # 1. Risk Overlay — agreement → position scale factor
    # ------------------------------------------------------------------

    def get_agreement_series(self) -> pd.Series:
        """Return agent agreement as a time-indexed Series."""
        sim = self._load_simulation()
        records = {
            pd.Timestamp(r['date']): r['agent_agreement']
            for r in sim['rounds']
        }
        return pd.Series(records, name='agent_agreement').sort_index()

    def get_risk_scale_series(
        self,
        floor: float = 0.3,
        ceiling: float = 1.0,
    ) -> pd.Series:
        """Convert agreement → risk scaling factor.

        Maps agreement ∈ [0, 1] to scale ∈ [floor, ceiling]:
            scale = floor + (ceiling - floor) * agreement

        When agents strongly disagree (agreement≈0) → scale≈0.3 → reduce positions 70%
        When agents agree (agreement≈1) → scale≈1.0 → full allocation

        Parameters
        ----------
        floor : float
            Minimum scale factor (maximum position reduction).
        ceiling : float
            Maximum scale factor (full allocation).
        """
        agreement = self.get_agreement_series()
        scale = floor + (ceiling - floor) * agreement.clip(0, 1)
        scale.name = 'risk_scale'
        return scale

    def get_risk_scale_at(self, date: pd.Timestamp, floor: float = 0.3) -> float:
        """Get risk scale factor for a specific date (nearest available)."""
        scale = self.get_risk_scale_series(floor=floor)
        if date in scale.index:
            return float(scale.loc[date])
        idx = scale.index.get_indexer([date], method='pad')[0]
        if idx < 0:
            return 1.0  # no data yet, full allocation
        return float(scale.iloc[idx])

    # ------------------------------------------------------------------
    # 2. Features — inject swarm intelligence into ML pipeline
    # ------------------------------------------------------------------

    def get_swarm_features(self) -> pd.DataFrame:
        """Build feature DataFrame from MiroFish simulation for ML consumption.

        Features:
            swarm_agreement      : raw agreement [0, 1]
            swarm_risk_scale     : agreement → position scale
            swarm_regime_defensive : 1 if defensive regime
            swarm_regime_risk_on   : 1 if risk-on regime
            swarm_vix              : VIX observed by agents
            swarm_signal_{ticker}  : ensemble signal per asset
        """
        sim = self._load_simulation()
        records = []

        for r in sim['rounds']:
            row = {
                'date': pd.Timestamp(r['date']),
                'swarm_agreement': r['agent_agreement'],
                'swarm_risk_scale': 0.3 + 0.7 * min(max(r['agent_agreement'], 0), 1),
                'swarm_vix': r.get('market_state', {}).get('vix', np.nan),
            }

            # Regime dummies (detect from market state or VIX)
            vix = row['swarm_vix']
            if pd.notna(vix):
                row['swarm_regime_defensive'] = 1.0 if vix > 25 else 0.0
                row['swarm_regime_risk_on'] = 1.0 if vix < 15 else 0.0
            else:
                row['swarm_regime_defensive'] = 0.0
                row['swarm_regime_risk_on'] = 0.0

            # Per-asset ensemble signals
            for i, val in enumerate(r.get('ensemble_signal', [])):
                row[f'swarm_signal_{i}'] = val

            records.append(row)

        df = pd.DataFrame(records).set_index('date').sort_index()
        return df

    def inject_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Merge swarm features into existing feature matrix.

        Uses forward-fill to align monthly MiroFish rounds with daily features.
        """
        if not self.is_available:
            logger.info("MiroFish data not available — skipping feature injection")
            return features

        swarm = self.get_swarm_features()
        # Reindex to daily, forward-fill (MiroFish runs monthly)
        swarm_daily = swarm.reindex(features.index, method='pad')
        # Only fill forward after first available date
        first_date = swarm.index.min()
        swarm_daily.loc[:first_date] = np.nan

        merged = features.join(swarm_daily, how='left')
        n_new = len(swarm_daily.columns)
        logger.info("Injected %d MiroFish features into feature matrix", n_new)
        return merged

    # ------------------------------------------------------------------
    # 3. Swarm Weights — direct portfolio from agent consensus
    # ------------------------------------------------------------------

    def get_swarm_weights(self) -> pd.DataFrame:
        """Return MiroFish portfolio weights as a time-indexed DataFrame.

        Each row is one rebalancing round, columns are tickers.
        """
        sim = self._load_simulation()
        weights = self._load_weights()
        dates = [pd.Timestamp(r['date']) for r in sim['rounds']]

        # Weights list may be shorter than rounds
        n = min(len(dates), len(weights))
        df = pd.DataFrame(weights[:n], index=dates[:n])
        df.index.name = 'date'
        return df

    def get_weights_at(self, date: pd.Timestamp, tickers: list[str]) -> np.ndarray:
        """Get swarm weights for a specific date, aligned to ticker order."""
        weights_df = self.get_swarm_weights()
        if weights_df.empty:
            return np.ones(len(tickers)) / len(tickers)

        # Find nearest prior date
        valid = weights_df.index[weights_df.index <= date]
        if len(valid) == 0:
            return np.ones(len(tickers)) / len(tickers)

        row = weights_df.loc[valid[-1]]
        result = np.array([row.get(t, 0.0) for t in tickers])
        total = result.sum()
        return result / total if total > 0 else np.ones(len(tickers)) / len(tickers)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary for logging."""
        if not self.is_available:
            return "MiroFish: No simulation data found."

        sim = self._load_simulation()
        n_rounds = len(sim.get('rounds', []))
        n_agents = sim.get('n_agents', 0)
        agreement = self.get_agreement_series()

        lines = [
            f"MiroFish: {n_rounds} rounds, {n_agents} agents",
            f"  Agreement: mean={agreement.mean():.3f}, "
            f"min={agreement.min():.3f}, max={agreement.max():.3f}",
            f"  Risk scale: mean={self.get_risk_scale_series().mean():.3f}",
            f"  Period: {agreement.index.min().date()} to {agreement.index.max().date()}",
        ]
        return "\n".join(lines)
