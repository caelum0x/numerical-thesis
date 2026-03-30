"""
Financial Market Simulator — MiroFish Extension
=================================================
Extends MiroFish's multi-agent simulation engine for financial markets.

Instead of simulating social media agents posting opinions, we simulate
market agents generating predictions and making allocation decisions.

Architecture mapping:
  MiroFish Twitter/Reddit agents  ->  Market trader agents
  Social posts & reactions        ->  Predictions & allocation signals
  Knowledge graph (Zep)           ->  Market state (prices, macro, signals)
  Simulation rounds               ->  Trading days / rebalancing periods
  Report agent                    ->  Performance attribution report

This module can be used standalone or integrated into the MiroFish web UI.
"""

import os
import sys
import glob
import json
import time
import pickle
import logging
import warnings
from datetime import datetime
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Add thesis project to path
THESIS_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'thesis_portfolio_opt')
THESIS_ROOT = os.path.abspath(THESIS_ROOT)
sys.path.insert(0, THESIS_ROOT)

logger = logging.getLogger(__name__)

# Paths used by ML model loading
RESULTS_DIR = os.path.join(THESIS_ROOT, 'data', 'results')
FEATURES_PATH = os.path.join(THESIS_ROOT, 'data', 'processed', 'features.csv')


# ============================================================================
# AGENT PROFILES (analogous to MiroFish's oasis_profile_generator)
# ============================================================================

class AgentType(Enum):
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"
    MACRO = "macro"
    VOLATILITY = "volatility"
    VALUE = "value"
    ML_LINEAR = "ml_linear"
    ML_TREE = "ml_tree"
    ML_ENSEMBLE = "ml_ensemble"
    ADAPTIVE = "adaptive"
    REGIME = "regime"
    NOISE = "noise"


@dataclass
class FinancialAgentProfile:
    """Agent personality -- maps to MiroFish's OASIS agent profiles."""
    agent_id: str
    name: str
    agent_type: AgentType
    risk_tolerance: float       # 0 (conservative) to 1 (aggressive)
    lookback_days: int          # how far back agent looks
    confidence: float           # signal strength multiplier (0-1)
    rebalance_freq: int         # days between trades
    max_position: float         # max weight per asset
    description: str = ""
    track_record: list = field(default_factory=list)


@dataclass
class MarketState:
    """Current market state -- analogous to MiroFish's simulation state."""
    date: pd.Timestamp
    prices: dict               # ticker -> price
    returns_1d: dict           # ticker -> 1d return
    macro: dict                # indicator -> value
    vix: float = 20.0
    regime: str = "normal"     # normal, high_vol, crisis


# ============================================================================
# FINANCIAL AGENTS — Base Class
# ============================================================================

class FinancialAgent(ABC):
    """Base market agent -- analogous to MiroFish's OASIS Agent."""

    def __init__(self, profile: FinancialAgentProfile):
        self.profile = profile

    @abstractmethod
    def generate_signal(self, prices: pd.DataFrame, macro: pd.DataFrame,
                        date: pd.Timestamp) -> np.ndarray:
        """Generate allocation signal for each asset. Returns array of signals."""
        pass

    def update_track_record(self, date, predicted, actual):
        """Update agent's memory with prediction outcome."""
        if len(predicted) > 0 and len(actual) > 0:
            ic = np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 1 else 0
            self.profile.track_record.append({
                'date': str(date), 'ic': float(ic),
                'dir_acc': float(np.mean(np.sign(predicted) == np.sign(actual)))
            })

    def get_recent_ic(self, window: int = 10) -> float:
        """Return the average IC over the last `window` track record entries."""
        if not self.profile.track_record:
            return 0.0
        recent = self.profile.track_record[-window:]
        ics = [r['ic'] for r in recent if np.isfinite(r['ic'])]
        return float(np.mean(ics)) if ics else 0.0


# ============================================================================
# FINANCIAL AGENTS — Classic Strategies
# ============================================================================

class MomentumTrader(FinancialAgent):
    """Follows recent price trends."""

    def generate_signal(self, prices, macro, date):
        idx = prices.index.get_indexer([date], method='nearest')[0]
        w = self.profile.lookback_days
        if idx < w:
            return np.zeros(prices.shape[1])
        recent = prices.iloc[idx - w:idx]
        momentum = (recent.iloc[-1] / recent.iloc[0] - 1).values
        return momentum * self.profile.confidence


class ContrarianTrader(FinancialAgent):
    """Bets against recent trends (mean reversion)."""

    def generate_signal(self, prices, macro, date):
        idx = prices.index.get_indexer([date], method='nearest')[0]
        w = self.profile.lookback_days
        if idx < w:
            return np.zeros(prices.shape[1])
        recent = prices.iloc[idx - w:idx]
        momentum = (recent.iloc[-1] / recent.iloc[0] - 1).values
        return -momentum * self.profile.confidence


class MacroTrader(FinancialAgent):
    """Allocates based on macroeconomic regime."""

    ASSET_MAP = {
        'SPY': 'equity', 'IWM': 'equity', 'EFA': 'equity', 'EEM': 'equity',
        'AGG': 'bond', 'TLT': 'bond', 'LQD': 'bond', 'HYG': 'bond', 'TIP': 'bond',
        'GLD': 'alt', 'VNQ': 'alt', 'DBC': 'alt',
    }

    def generate_signal(self, prices, macro, date):
        tickers = list(prices.columns)
        signal = np.zeros(len(tickers))

        if date not in macro.index:
            idx = macro.index.get_indexer([date], method='nearest')[0]
            date = macro.index[idx]

        row = macro.loc[date]

        # VIX regime
        if 'VIXCLS' in row.index and pd.notna(row['VIXCLS']):
            vix = row['VIXCLS']
            for j, t in enumerate(tickers):
                atype = self.ASSET_MAP.get(t, 'equity')
                if vix > 25:  # high fear -> bonds/gold
                    signal[j] += 0.02 if atype in ('bond', 'alt') else -0.02
                else:
                    signal[j] += 0.01 if atype == 'equity' else 0

        # Yield curve
        if 'T10Y2Y' in row.index and pd.notna(row['T10Y2Y']):
            spread = row['T10Y2Y']
            for j, t in enumerate(tickers):
                atype = self.ASSET_MAP.get(t, 'equity')
                if spread < 0:  # inverted -> recession risk
                    signal[j] += 0.02 if atype == 'bond' else -0.01

        return signal * self.profile.confidence


class VolatilityTrader(FinancialAgent):
    """Inverse-volatility allocation."""

    def generate_signal(self, prices, macro, date):
        idx = prices.index.get_indexer([date], method='nearest')[0]
        w = self.profile.lookback_days
        if idx < w:
            return np.zeros(prices.shape[1])
        rets = prices.iloc[idx - w:idx].pct_change().dropna()
        vols = rets.std().values
        inv_vol = 1.0 / np.maximum(vols, 1e-8)
        signal = inv_vol / inv_vol.sum() - 1.0 / prices.shape[1]
        return signal * self.profile.confidence


class NoiseTrader(FinancialAgent):
    """Random trader (represents uninformed market participant)."""

    def generate_signal(self, prices, macro, date):
        np.random.seed(hash(str(date)) % 2**31)
        return np.random.randn(prices.shape[1]) * 0.001 * self.profile.confidence


# ============================================================================
# ML TRADER — loads pre-trained models from thesis project
# ============================================================================

def scan_model_files(results_dir: str = RESULTS_DIR) -> Dict[str, List[dict]]:
    """
    Scan the thesis results directory for trained model pickle files.

    Returns dict keyed by prediction type:
      'ret' -> list of {ticker, model_name, path, target}   (return models)
      'fwd' -> list of {ticker, model_name, path, target}   (forward models)
    """
    model_files = {'ret': [], 'fwd': []}
    pattern = os.path.join(results_dir, 'model_*.pkl')

    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        # Parse: model_{model_name}_{TICKER}_{ret|fwd}_21d.pkl
        parts = fname.replace('.pkl', '').split('_')
        # Find ret or fwd in parts
        pred_type = None
        ticker = None
        for i, p in enumerate(parts):
            if p in ('ret', 'fwd') and i > 0:
                pred_type = p
                ticker = parts[i - 1]
                model_name = '_'.join(parts[1:i - 1])
                break

        if pred_type and ticker:
            model_files[pred_type].append({
                'ticker': ticker,
                'model_name': model_name,
                'path': fpath,
                'target': f'{ticker}_{pred_type}_21d',
            })

    logger.info(f"Found {len(model_files['ret'])} return models, "
                f"{len(model_files['fwd'])} forward models")
    return model_files


def load_model_bundle(fpath: str) -> Optional[dict]:
    """Load a single model pickle file. Returns dict with model, scaler, etc."""
    try:
        with open(fpath, 'rb') as f:
            bundle = pickle.load(f)
        # Validate expected keys
        if 'model' in bundle and 'scaler' in bundle:
            return bundle
        else:
            logger.warning(f"Model file {fpath} missing expected keys")
            return None
    except Exception as e:
        logger.warning(f"Failed to load model {fpath}: {e}")
        return None


def load_all_models(results_dir: str = RESULTS_DIR,
                    pred_type: str = 'ret') -> Dict[str, dict]:
    """
    Load all models of a given prediction type.

    Returns dict: ticker -> {model, scaler, model_name, target, feature_cols}
    """
    model_catalog = scan_model_files(results_dir)
    loaded = {}

    for entry in model_catalog.get(pred_type, []):
        ticker = entry['ticker']
        bundle = load_model_bundle(entry['path'])
        if bundle is not None:
            # Extract feature column names from scaler
            feature_cols = list(bundle['scaler'].feature_names_in_)
            bundle['feature_cols'] = feature_cols
            loaded[ticker] = bundle
            logger.info(f"Loaded {entry['model_name']} model for {ticker} "
                        f"({len(feature_cols)} features)")

    return loaded


class MLTrader(FinancialAgent):
    """
    Uses pre-trained ML model for predictions.

    Loads models from the thesis project's results directory. Each model
    predicts 21-day returns for a specific asset. The scaler stored with
    each model defines which feature columns are needed.
    """

    def __init__(self, profile: FinancialAgentProfile,
                 model_dict: Dict[str, dict],
                 feature_cols: Optional[List[str]] = None,
                 features_cache: Optional[pd.DataFrame] = None):
        super().__init__(profile)
        self.models = model_dict
        # If feature_cols not given, derive from first model's scaler
        if feature_cols is None and model_dict:
            first_model = next(iter(model_dict.values()))
            self.feature_cols = first_model.get('feature_cols',
                                                list(first_model['scaler'].feature_names_in_))
        else:
            self.feature_cols = feature_cols or []
        # Cache features DataFrame to avoid re-reading on every call
        self._features_cache = features_cache

    def _load_features(self) -> Optional[pd.DataFrame]:
        """Load or return cached features DataFrame."""
        if self._features_cache is not None:
            return self._features_cache
        if not os.path.exists(FEATURES_PATH):
            logger.warning(f"Features file not found: {FEATURES_PATH}")
            return None
        try:
            self._features_cache = pd.read_csv(
                FEATURES_PATH, index_col=0, parse_dates=True
            )
            return self._features_cache
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return None

    def generate_signal(self, prices: pd.DataFrame, macro: pd.DataFrame,
                        date: pd.Timestamp) -> np.ndarray:
        """
        Generate signal by running each asset's pre-trained model.

        For each ticker that has a loaded model:
          1. Look up the feature row for this date
          2. Scale features using the model's scaler
          3. Predict expected return
          4. Use prediction as signal
        """
        features = self._load_features()
        if features is None:
            return np.zeros(prices.shape[1])

        tickers = list(prices.columns)
        signal = np.zeros(len(tickers))

        # Find nearest date in features index
        if date not in features.index:
            idx = features.index.get_indexer([date], method='nearest')[0]
            if idx < 0 or idx >= len(features):
                return signal
            feat_date = features.index[idx]
        else:
            feat_date = date

        for j, ticker in enumerate(tickers):
            if ticker not in self.models:
                continue
            bundle = self.models[ticker]
            scaler = bundle['scaler']
            model = bundle['model']
            fcols = bundle.get('feature_cols', self.feature_cols)

            # Get feature row — only the columns this model expects
            available_cols = [c for c in fcols if c in features.columns]
            if len(available_cols) < len(fcols) * 0.8:
                # Too many missing features, skip
                continue

            row = features.loc[[feat_date], available_cols]
            if row.isnull().all(axis=1).iloc[0]:
                continue

            # Fill any remaining NaN with 0
            row = row.fillna(0.0)

            # Pad missing columns with zeros if needed
            if len(available_cols) < len(fcols):
                for mc in fcols:
                    if mc not in available_cols:
                        row[mc] = 0.0
                row = row[fcols]

            try:
                X_scaled = scaler.transform(row)
                pred = model.predict(X_scaled)[0]
                if np.isfinite(pred):
                    signal[j] = pred
            except Exception as e:
                logger.debug(f"Prediction failed for {ticker}: {e}")

        return signal * self.profile.confidence


class MLEnsembleTrader(FinancialAgent):
    """
    Ensemble ML trader that loads BOTH return and forward models for each
    asset and averages their predictions. This gives a more robust signal
    by combining models trained on different targets.
    """

    def __init__(self, profile: FinancialAgentProfile,
                 ret_models: Dict[str, dict],
                 fwd_models: Dict[str, dict],
                 features_cache: Optional[pd.DataFrame] = None):
        super().__init__(profile)
        self.ret_models = ret_models
        self.fwd_models = fwd_models
        self._features_cache = features_cache

    def _load_features(self) -> Optional[pd.DataFrame]:
        """Load or return cached features DataFrame."""
        if self._features_cache is not None:
            return self._features_cache
        if not os.path.exists(FEATURES_PATH):
            return None
        try:
            self._features_cache = pd.read_csv(
                FEATURES_PATH, index_col=0, parse_dates=True
            )
            return self._features_cache
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return None

    def _predict_with_bundle(self, bundle: dict, feat_row: pd.DataFrame,
                             feat_date) -> Optional[float]:
        """Run a single model prediction."""
        scaler = bundle['scaler']
        model = bundle['model']
        fcols = bundle.get('feature_cols', list(scaler.feature_names_in_))

        available = [c for c in fcols if c in feat_row.columns]
        if len(available) < len(fcols) * 0.8:
            return None

        row = feat_row[available].copy()
        row = row.fillna(0.0)

        # Pad missing columns
        if len(available) < len(fcols):
            for mc in fcols:
                if mc not in available:
                    row[mc] = 0.0
            row = row[fcols]

        try:
            X_scaled = scaler.transform(row)
            pred = model.predict(X_scaled)[0]
            return float(pred) if np.isfinite(pred) else None
        except Exception:
            return None

    def generate_signal(self, prices: pd.DataFrame, macro: pd.DataFrame,
                        date: pd.Timestamp) -> np.ndarray:
        """
        Generate ensemble signal by averaging return and forward model predictions.
        """
        features = self._load_features()
        if features is None:
            return np.zeros(prices.shape[1])

        tickers = list(prices.columns)
        signal = np.zeros(len(tickers))

        # Nearest date
        if date not in features.index:
            idx = features.index.get_indexer([date], method='nearest')[0]
            if idx < 0 or idx >= len(features):
                return signal
            feat_date = features.index[idx]
        else:
            feat_date = date

        feat_row = features.loc[[feat_date]]

        for j, ticker in enumerate(tickers):
            preds = []

            # Return model prediction
            if ticker in self.ret_models:
                p = self._predict_with_bundle(self.ret_models[ticker],
                                              feat_row, feat_date)
                if p is not None:
                    preds.append(p)

            # Forward model prediction
            if ticker in self.fwd_models:
                p = self._predict_with_bundle(self.fwd_models[ticker],
                                              feat_row, feat_date)
                if p is not None:
                    preds.append(p)

            if preds:
                signal[j] = np.mean(preds)

        return signal * self.profile.confidence


# ============================================================================
# ADAPTIVE AGENT — adjusts strategy based on track record
# ============================================================================

class AdaptiveAgent(FinancialAgent):
    """
    Agent that adjusts its strategy based on its own performance track record.

    Implements simple online learning / adaptation:
      - Monitors recent Information Coefficient (IC)
      - If recent IC is negative, reduces confidence to dampen signals
      - If recent IC is positive, increases confidence up to a cap
      - Blends momentum and contrarian signals, shifting blend based on
        which strategy has been working recently
    """

    def __init__(self, profile: FinancialAgentProfile,
                 base_confidence: float = 0.6,
                 adaptation_rate: float = 0.1,
                 lookback_window: int = 10,
                 min_confidence: float = 0.1,
                 max_confidence: float = 0.95):
        super().__init__(profile)
        self.base_confidence = base_confidence
        self.adaptation_rate = adaptation_rate
        self.lookback_window = lookback_window
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        # Internal state: blend weight between momentum (1.0) and contrarian (0.0)
        self._momentum_weight = 0.5
        self._momentum_history: List[float] = []
        self._contrarian_history: List[float] = []

    def _compute_momentum_signal(self, prices: pd.DataFrame,
                                 idx: int, w: int) -> np.ndarray:
        """Compute raw momentum signal."""
        if idx < w:
            return np.zeros(prices.shape[1])
        recent = prices.iloc[idx - w:idx]
        return (recent.iloc[-1] / recent.iloc[0] - 1).values

    def _compute_contrarian_signal(self, prices: pd.DataFrame,
                                   idx: int, w: int) -> np.ndarray:
        """Compute raw contrarian (mean-reversion) signal."""
        return -self._compute_momentum_signal(prices, idx, w)

    def _adapt_parameters(self):
        """
        Adjust internal parameters based on recent performance.

        Uses the track record to determine:
          1. Overall confidence level
          2. Blend between momentum and contrarian signals
        """
        recent_ic = self.get_recent_ic(self.lookback_window)

        # Adjust confidence based on recent IC
        if recent_ic > 0.05:
            # Positive IC: agent is adding value, increase confidence
            self.profile.confidence = min(
                self.profile.confidence + self.adaptation_rate * 0.5,
                self.max_confidence
            )
        elif recent_ic < -0.05:
            # Negative IC: agent is destroying value, reduce confidence
            self.profile.confidence = max(
                self.profile.confidence - self.adaptation_rate,
                self.min_confidence
            )
        else:
            # Neutral: slowly drift back to base confidence
            self.profile.confidence += (
                (self.base_confidence - self.profile.confidence) * 0.1
            )

        # Adjust momentum/contrarian blend based on which has been working
        if len(self._momentum_history) >= 5 and len(self._contrarian_history) >= 5:
            recent_mom_perf = np.mean(self._momentum_history[-5:])
            recent_con_perf = np.mean(self._contrarian_history[-5:])

            if recent_mom_perf > recent_con_perf:
                self._momentum_weight = min(self._momentum_weight + 0.05, 0.9)
            elif recent_con_perf > recent_mom_perf:
                self._momentum_weight = max(self._momentum_weight - 0.05, 0.1)

    def generate_signal(self, prices: pd.DataFrame, macro: pd.DataFrame,
                        date: pd.Timestamp) -> np.ndarray:
        """
        Generate an adaptive signal that blends momentum and contrarian
        strategies based on recent performance.
        """
        # Adapt parameters before generating new signal
        self._adapt_parameters()

        idx = prices.index.get_indexer([date], method='nearest')[0]
        w = self.profile.lookback_days

        mom_signal = self._compute_momentum_signal(prices, idx, w)
        con_signal = self._compute_contrarian_signal(prices, idx, w)

        # Blend signals based on adaptive weight
        blended = (self._momentum_weight * mom_signal +
                   (1.0 - self._momentum_weight) * con_signal)

        return blended * self.profile.confidence

    def update_track_record(self, date, predicted, actual):
        """Extended track record update that also tracks per-strategy performance."""
        super().update_track_record(date, predicted, actual)

        if len(predicted) > 0 and len(actual) > 0:
            # Track how momentum component would have performed
            mom_ic = np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 1 else 0.0
            self._momentum_history.append(float(mom_ic) if np.isfinite(mom_ic) else 0.0)

            # Track contrarian component (inverted)
            con_ic = np.corrcoef(-predicted, actual)[0, 1] if len(predicted) > 1 else 0.0
            self._contrarian_history.append(float(con_ic) if np.isfinite(con_ic) else 0.0)


# ============================================================================
# REGIME AGENT — detects market regime and switches strategy
# ============================================================================

class RegimeAgent(FinancialAgent):
    """
    Detects market regime using VIX and switches strategy accordingly.

    Regime classification:
      - High VIX (>25): Defensive -- overweight bonds and gold
      - Low VIX (<15):  Risk-on  -- overweight equities
      - Normal VIX:     Balanced -- equal across asset classes

    Uses actual VIXCLS data from the macro DataFrame to determine the current
    regime. Also considers the yield curve (T10Y2Y) as a secondary signal.
    """

    # Asset class mapping for regime-based allocation
    ASSET_CLASSES = {
        'SPY': 'equity', 'IWM': 'equity', 'EFA': 'equity', 'EEM': 'equity',
        'AGG': 'bond', 'TLT': 'bond', 'LQD': 'bond', 'HYG': 'bond', 'TIP': 'bond',
        'GLD': 'alt', 'VNQ': 'alt', 'DBC': 'alt',
    }

    # Regime-specific allocation weights by asset class
    REGIME_WEIGHTS = {
        'defensive': {'equity': -0.03, 'bond': 0.04, 'alt': 0.02},
        'risk_on':   {'equity': 0.04, 'bond': -0.02, 'alt': 0.01},
        'balanced':  {'equity': 0.01, 'bond': 0.01, 'alt': 0.005},
    }

    def __init__(self, profile: FinancialAgentProfile,
                 high_vix_threshold: float = 25.0,
                 low_vix_threshold: float = 15.0):
        super().__init__(profile)
        self.high_vix_threshold = high_vix_threshold
        self.low_vix_threshold = low_vix_threshold
        self._regime_history: List[dict] = []

    def _detect_regime(self, macro: pd.DataFrame,
                       date: pd.Timestamp) -> Tuple[str, float]:
        """
        Detect the current market regime from macro data.

        Returns (regime_name, vix_value).
        """
        # Find the VIX value for this date
        vix_val = 20.0  # default
        if 'VIXCLS' in macro.columns:
            vix_data = macro.loc[:date, 'VIXCLS'].dropna()
            if len(vix_data) > 0:
                vix_val = float(vix_data.iloc[-1])

        # Primary regime detection via VIX
        if vix_val > self.high_vix_threshold:
            regime = 'defensive'
        elif vix_val < self.low_vix_threshold:
            regime = 'risk_on'
        else:
            regime = 'balanced'

        # Secondary check: inverted yield curve overrides to defensive
        if 'T10Y2Y' in macro.columns:
            spread_data = macro.loc[:date, 'T10Y2Y'].dropna()
            if len(spread_data) > 0 and float(spread_data.iloc[-1]) < -0.5:
                # Deeply inverted yield curve -> even if VIX is low,
                # become more defensive
                if regime == 'risk_on':
                    regime = 'balanced'
                elif regime == 'balanced':
                    regime = 'defensive'

        return regime, vix_val

    def generate_signal(self, prices: pd.DataFrame, macro: pd.DataFrame,
                        date: pd.Timestamp) -> np.ndarray:
        """
        Generate allocation signal based on detected market regime.
        """
        tickers = list(prices.columns)
        signal = np.zeros(len(tickers))

        regime, vix_val = self._detect_regime(macro, date)

        # Record regime for analysis
        self._regime_history.append({
            'date': str(date),
            'regime': regime,
            'vix': vix_val,
        })

        # Apply regime-specific weights
        weights = self.REGIME_WEIGHTS.get(regime, self.REGIME_WEIGHTS['balanced'])

        for j, ticker in enumerate(tickers):
            asset_class = self.ASSET_CLASSES.get(ticker, 'equity')
            signal[j] = weights.get(asset_class, 0.0)

        # In defensive regime, add extra weight to TLT and GLD
        if regime == 'defensive':
            for j, ticker in enumerate(tickers):
                if ticker == 'TLT':
                    signal[j] += 0.02
                elif ticker == 'GLD':
                    signal[j] += 0.015

        # In risk-on regime, add extra weight to small-cap and EM
        if regime == 'risk_on':
            for j, ticker in enumerate(tickers):
                if ticker == 'IWM':
                    signal[j] += 0.015
                elif ticker == 'EEM':
                    signal[j] += 0.01

        return signal * self.profile.confidence

    def get_regime_history(self) -> pd.DataFrame:
        """Return regime history as DataFrame for analysis."""
        if not self._regime_history:
            return pd.DataFrame()
        df = pd.DataFrame(self._regime_history)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')


# ============================================================================
# AGENT FACTORY (analogous to MiroFish's profile generator)
# ============================================================================

AGENT_CLASSES = {
    AgentType.MOMENTUM: MomentumTrader,
    AgentType.CONTRARIAN: ContrarianTrader,
    AgentType.MACRO: MacroTrader,
    AgentType.VOLATILITY: VolatilityTrader,
    AgentType.NOISE: NoiseTrader,
    AgentType.ADAPTIVE: AdaptiveAgent,
    AgentType.REGIME: RegimeAgent,
}


def create_agent(profile: FinancialAgentProfile, **kwargs) -> FinancialAgent:
    """Factory function to create agents from profiles."""
    if profile.agent_type in (AgentType.ML_LINEAR, AgentType.ML_TREE):
        return MLTrader(profile, **kwargs)
    if profile.agent_type == AgentType.ML_ENSEMBLE:
        return MLEnsembleTrader(profile, **kwargs)
    cls = AGENT_CLASSES.get(profile.agent_type, NoiseTrader)
    return cls(profile)


def create_ml_agents(results_dir: str = RESULTS_DIR,
                     features_cache: Optional[pd.DataFrame] = None
                     ) -> List[FinancialAgent]:
    """
    Scan for pre-trained models and create ML-based agents.

    Creates:
      - One MLTrader for return models (ret_21d)
      - One MLTrader for forward models (fwd_21d)
      - One MLEnsembleTrader combining both
    """
    agents = []

    ret_models = load_all_models(results_dir, pred_type='ret')
    fwd_models = load_all_models(results_dir, pred_type='fwd')

    if ret_models:
        profile_ret = FinancialAgentProfile(
            agent_id="ml_ret",
            name="ML_ReturnPredictor",
            agent_type=AgentType.ML_LINEAR,
            risk_tolerance=0.5,
            lookback_days=21,
            confidence=0.9,
            rebalance_freq=21,
            max_position=0.3,
            description="ML models predicting 21-day returns"
        )
        agents.append(MLTrader(profile_ret, ret_models,
                               features_cache=features_cache))
        print(f"  Created ML return predictor ({len(ret_models)} assets)")

    if fwd_models:
        profile_fwd = FinancialAgentProfile(
            agent_id="ml_fwd",
            name="ML_ForwardPredictor",
            agent_type=AgentType.ML_LINEAR,
            risk_tolerance=0.5,
            lookback_days=21,
            confidence=0.85,
            rebalance_freq=21,
            max_position=0.3,
            description="ML models predicting 21-day forward returns"
        )
        agents.append(MLTrader(profile_fwd, fwd_models,
                               features_cache=features_cache))
        print(f"  Created ML forward predictor ({len(fwd_models)} assets)")

    if ret_models and fwd_models:
        profile_ens = FinancialAgentProfile(
            agent_id="ml_ensemble",
            name="ML_EnsemblePredictor",
            agent_type=AgentType.ML_ENSEMBLE,
            risk_tolerance=0.5,
            lookback_days=21,
            confidence=0.95,
            rebalance_freq=21,
            max_position=0.3,
            description="Ensemble of return + forward ML models"
        )
        agents.append(MLEnsembleTrader(profile_ens, ret_models, fwd_models,
                                       features_cache=features_cache))
        print(f"  Created ML ensemble predictor")

    if not agents:
        print("  No ML models found — skipping ML agents")

    return agents


# ============================================================================
# SIMULATION ENGINE (analogous to MiroFish's SimulationRunner)
# ============================================================================

@dataclass
class SimulationRound:
    """One round of the simulation (one rebalancing period)."""
    round_num: int
    date: str
    agent_signals: dict        # agent_id -> signal array
    ensemble_signal: list      # aggregated signal
    agent_agreement: float     # cross-agent correlation
    market_state: dict         # key market metrics at this point


class FinancialSimulationRunner:
    """
    Orchestrates multi-agent market simulation.
    Direct adaptation of MiroFish's SimulationRunner for financial markets.

    MiroFish flow:  Create agents -> Run rounds -> Collect actions -> Build report
    Our flow:       Create agents -> Run prediction rounds -> Aggregate -> Backtest
    """

    def __init__(self):
        self.agents: list[FinancialAgent] = []
        self.rounds: list[SimulationRound] = []
        self.status = "idle"

    def add_agent(self, agent: FinancialAgent):
        self.agents.append(agent)

    def create_default_agents(self):
        """Create a diverse set of market agents."""
        profiles = [
            FinancialAgentProfile("mom_21", "ShortMomentum", AgentType.MOMENTUM, 0.7, 21, 0.8, 21, 0.3),
            FinancialAgentProfile("mom_63", "MedMomentum", AgentType.MOMENTUM, 0.6, 63, 0.7, 21, 0.3),
            FinancialAgentProfile("mom_126", "LongMomentum", AgentType.MOMENTUM, 0.5, 126, 0.6, 21, 0.3),
            FinancialAgentProfile("cont_21", "ShortContrarian", AgentType.CONTRARIAN, 0.5, 21, 0.5, 21, 0.3),
            FinancialAgentProfile("cont_63", "MedContrarian", AgentType.CONTRARIAN, 0.4, 63, 0.4, 21, 0.3),
            FinancialAgentProfile("macro_1", "MacroStrategist", AgentType.MACRO, 0.5, 252, 0.9, 21, 0.4),
            FinancialAgentProfile("vol_1", "VolTrader", AgentType.VOLATILITY, 0.3, 63, 0.7, 21, 0.3),
            FinancialAgentProfile("noise_1", "RetailTrader", AgentType.NOISE, 0.8, 5, 0.2, 5, 0.2),
            FinancialAgentProfile("noise_2", "RetailTrader2", AgentType.NOISE, 0.9, 5, 0.1, 5, 0.2),
        ]

        for p in profiles:
            self.add_agent(create_agent(p))

        # Add adaptive and regime agents
        adaptive_profile = FinancialAgentProfile(
            "adaptive_1", "AdaptiveTrader", AgentType.ADAPTIVE,
            0.6, 42, 0.6, 21, 0.3,
            description="Self-adapting momentum/contrarian blend"
        )
        self.add_agent(AdaptiveAgent(adaptive_profile))

        regime_profile = FinancialAgentProfile(
            "regime_1", "RegimeTrader", AgentType.REGIME,
            0.5, 252, 0.8, 21, 0.4,
            description="VIX-based regime switching agent"
        )
        self.add_agent(RegimeAgent(regime_profile))

        print(f"Created {len(self.agents)} market agents")

    def run_simulation(self, prices: pd.DataFrame, macro: pd.DataFrame,
                       start_date: str = '2022-01-01', rebalance_freq: int = 21,
                       max_rounds: int = None) -> list[SimulationRound]:
        """
        Run the full simulation. Analogous to MiroFish's start_simulation().

        Each round:
        1. All agents observe market state
        2. Each agent generates a prediction/signal
        3. Signals are aggregated into ensemble
        4. Ensemble drives portfolio allocation
        """
        self.status = "running"
        self.rounds = []

        oos_prices = prices[prices.index >= pd.Timestamp(start_date)]
        dates = oos_prices.index
        round_num = 0

        for t in range(rebalance_freq, len(dates), rebalance_freq):
            if max_rounds and round_num >= max_rounds:
                break

            date = dates[t]
            agent_signals = {}
            all_signals = []

            for agent in self.agents:
                try:
                    signal = agent.generate_signal(prices, macro, date)
                    if signal is not None and np.isfinite(signal).all():
                        agent_signals[agent.profile.agent_id] = signal.tolist()
                        all_signals.append(signal)
                except Exception as e:
                    logger.warning(f"Agent {agent.profile.name} failed: {e}")

            # Aggregate: confidence-weighted ensemble
            if all_signals:
                weights = np.array([a.profile.confidence for a in self.agents[:len(all_signals)]])
                weights /= weights.sum()
                ensemble = np.average(all_signals, axis=0, weights=weights)

                # Agent agreement (how much do they agree?)
                if len(all_signals) > 1:
                    corrs = [np.corrcoef(ensemble, s)[0, 1] for s in all_signals if np.std(s) > 0]
                    agreement = np.mean(corrs) if corrs else 0
                else:
                    agreement = 1.0
            else:
                ensemble = np.zeros(prices.shape[1])
                agreement = 0

            # Market state snapshot
            idx = prices.index.get_indexer([date], method='nearest')[0]
            mkt_state = {
                'date': str(date.date()),
                'spy_price': float(prices.iloc[idx].get('SPY', 0)),
                'vix': float(macro.loc[:date, 'VIXCLS'].dropna().iloc[-1]) if 'VIXCLS' in macro.columns else 0,
            }

            sim_round = SimulationRound(
                round_num=round_num,
                date=str(date.date()),
                agent_signals=agent_signals,
                ensemble_signal=ensemble.tolist(),
                agent_agreement=float(agreement),
                market_state=mkt_state,
            )
            self.rounds.append(sim_round)
            round_num += 1

            if round_num % 10 == 0:
                print(f"  Round {round_num}: {date.date()} | Agents: {len(all_signals)} | Agreement: {agreement:.3f}")

        self.status = "completed"
        print(f"Simulation complete: {len(self.rounds)} rounds")
        return self.rounds

    def get_ensemble_predictions(self) -> pd.DataFrame:
        """Extract ensemble signals as a DataFrame for backtesting."""
        records = []
        for r in self.rounds:
            records.append({'date': pd.Timestamp(r.date), **{f'signal_{i}': v for i, v in enumerate(r.ensemble_signal)}})
        return pd.DataFrame(records).set_index('date') if records else pd.DataFrame()

    def get_agreement_series(self) -> pd.Series:
        """How much do agents agree over time?"""
        return pd.Series(
            {pd.Timestamp(r.date): r.agent_agreement for r in self.rounds}
        )

    def generate_report(self) -> str:
        """
        Generate analysis report -- analogous to MiroFish's ReportAgent.
        In MiroFish, this uses ReACT pattern with tool calls.
        Here we generate a structured text report.
        """
        lines = [
            "=" * 70,
            "FINANCIAL SIMULATION REPORT",
            "Powered by MiroFish Multi-Agent Engine",
            "=" * 70, "",
            f"Simulation period: {self.rounds[0].date} to {self.rounds[-1].date}" if self.rounds else "No rounds",
            f"Total rounds: {len(self.rounds)}",
            f"Active agents: {len(self.agents)}", "",
            "--- Agent Roster ---",
        ]

        for agent in self.agents:
            tr = agent.profile.track_record
            if tr:
                avg_ic = np.mean([r['ic'] for r in tr])
                avg_da = np.mean([r['dir_acc'] for r in tr])
                lines.append(f"  {agent.profile.name:20s} ({agent.profile.agent_type.value:12s}) IC={avg_ic:+.3f} DA={avg_da:.1%}")
            else:
                lines.append(f"  {agent.profile.name:20s} ({agent.profile.agent_type.value:12s}) (no track record)")

        if self.rounds:
            agreements = [r.agent_agreement for r in self.rounds]
            lines.extend(["", "--- Collective Intelligence Metrics ---",
                          f"  Mean agreement: {np.mean(agreements):.3f}",
                          f"  Min agreement:  {np.min(agreements):.3f} (high uncertainty)",
                          f"  Max agreement:  {np.max(agreements):.3f} (consensus)",
                          "", "--- Interpretation ---",
                          "  Low agreement = agents disagree = market uncertainty = reduce positions",
                          "  High agreement = consensus = stronger conviction = normal positions"])

        return "\n".join(lines)

    def save_simulation(self, path: str):
        """Save simulation results to JSON (analogous to MiroFish's state persistence)."""
        data = {
            'status': self.status,
            'n_agents': len(self.agents),
            'n_rounds': len(self.rounds),
            'rounds': [asdict(r) for r in self.rounds],
            'agents': [asdict(a.profile) for a in self.agents],
        }
        # Convert Enum to string for JSON serialization
        for a in data['agents']:
            a['agent_type'] = a['agent_type'].value

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Simulation saved to {path}")


# ============================================================================
# SIMULATION BACKTESTER — converts signals to actual portfolio performance
# ============================================================================

class SimulationBacktester:
    """
    Takes simulation results and runs actual portfolio backtest.

    Pipeline:
      1. Convert ensemble signals to expected return vector (mu)
      2. Run mean-variance optimization using cvxpy
      3. Track portfolio returns, turnover, and drawdowns
      4. Compare against benchmarks (SPY, Equal Weight, 60/40)
      5. Compute full performance metrics suite
    """

    BENCHMARK_60_40 = {
        'SPY': 0.36, 'IWM': 0.08, 'EFA': 0.08, 'EEM': 0.08,
        'AGG': 0.20, 'TLT': 0.10, 'LQD': 0.05, 'HYG': 0.05,
    }

    def __init__(self, prices: pd.DataFrame,
                 risk_aversion: float = 1.0,
                 max_weight: float = 0.30,
                 min_weight: float = 0.0,
                 transaction_cost_bps: float = 10.0):
        self.prices = prices
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.tc_bps = transaction_cost_bps / 10000.0
        self.n_assets = prices.shape[1]
        self.tickers = list(prices.columns)

        # Results storage
        self.portfolio_weights: List[dict] = []
        self.portfolio_returns: List[float] = []
        self.benchmark_returns: Dict[str, List[float]] = {
            'spy': [], 'equal_weight': [], 'sixty_forty': []
        }
        self.turnover_history: List[float] = []
        self.dates: List[pd.Timestamp] = []

    def _mean_variance_optimize(self, mu: np.ndarray,
                                Sigma: np.ndarray) -> np.ndarray:
        """
        Solve the mean-variance optimization problem using cvxpy:

            max  mu^T w - (gamma/2) w^T Sigma w
            s.t. sum(w) = 1
                 w >= min_weight
                 w <= max_weight
        """
        try:
            import cvxpy as cp
        except ImportError:
            # Fallback: signal-weighted allocation without optimization
            logger.warning("cvxpy not available; using signal-proportional allocation")
            w = mu - mu.min() + 1e-6
            w = w / w.sum()
            return np.clip(w, self.min_weight, self.max_weight)

        n = len(mu)
        w = cp.Variable(n)

        # Ensure Sigma is positive semi-definite
        Sigma_reg = Sigma + np.eye(n) * 1e-6

        ret = mu @ w
        risk = cp.quad_form(w, Sigma_reg)
        objective = cp.Maximize(ret - (self.risk_aversion / 2.0) * risk)

        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight,
        ]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
            if prob.status in ('optimal', 'optimal_inaccurate'):
                weights = w.value
                # Clip and re-normalize
                weights = np.clip(weights, self.min_weight, self.max_weight)
                weights = weights / weights.sum()
                return weights
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")

        # Fallback: equal weight
        return np.ones(n) / n

    def _compute_covariance(self, prices: pd.DataFrame,
                            end_date: pd.Timestamp,
                            lookback: int = 126) -> np.ndarray:
        """Compute sample covariance matrix using recent returns."""
        idx = prices.index.get_indexer([end_date], method='nearest')[0]
        start_idx = max(0, idx - lookback)
        rets = prices.iloc[start_idx:idx].pct_change().dropna()
        if len(rets) < 10:
            return np.eye(prices.shape[1]) * 0.01
        cov = rets.cov().values
        # Shrink toward diagonal (Ledoit-Wolf style simple shrinkage)
        alpha = 0.3
        diag = np.diag(np.diag(cov))
        return (1 - alpha) * cov + alpha * diag

    def run_backtest(self, simulation_rounds: List[SimulationRound],
                     agreement_risk_manager: Optional['AgreementBasedRiskManager'] = None
                     ) -> pd.DataFrame:
        """
        Convert simulation ensemble signals to portfolio returns.

        For each simulation round:
          1. Extract ensemble signal as expected return (mu)
          2. Compute covariance from recent prices
          3. Solve mean-variance optimization
          4. Apply risk overlay if provided
          5. Track portfolio return over next period
        """
        prev_weights = np.ones(self.n_assets) / self.n_assets
        results = []

        for i, rnd in enumerate(simulation_rounds):
            date = pd.Timestamp(rnd.date)
            self.dates.append(date)

            # Ensemble signal as expected return proxy
            mu = np.array(rnd.ensemble_signal)
            if len(mu) != self.n_assets:
                mu = np.zeros(self.n_assets)

            # Covariance matrix
            Sigma = self._compute_covariance(self.prices, date)

            # Optimize weights
            weights = self._mean_variance_optimize(mu, Sigma)

            # Apply agreement-based risk overlay
            if agreement_risk_manager is not None:
                weights = agreement_risk_manager.adjust_weights(
                    weights, rnd.agent_agreement
                )

            # Compute turnover and transaction costs
            turnover = float(np.sum(np.abs(weights - prev_weights)))
            tc = turnover * self.tc_bps
            self.turnover_history.append(turnover)

            # Compute portfolio return over next period
            idx = self.prices.index.get_indexer([date], method='nearest')[0]
            next_idx = min(idx + 21, len(self.prices) - 1)
            if next_idx > idx:
                period_returns = (self.prices.iloc[next_idx] /
                                  self.prices.iloc[idx] - 1).values
                port_ret = float(np.dot(weights, period_returns)) - tc
            else:
                port_ret = 0.0

            self.portfolio_returns.append(port_ret)

            # Benchmark returns for the same period
            if next_idx > idx:
                pr = self.prices.iloc[next_idx] / self.prices.iloc[idx] - 1

                # SPY benchmark
                spy_ret = float(pr.get('SPY', 0.0))
                self.benchmark_returns['spy'].append(spy_ret)

                # Equal weight benchmark
                ew_ret = float(pr.mean())
                self.benchmark_returns['equal_weight'].append(ew_ret)

                # 60/40 benchmark
                bm_weights = np.array([
                    self.BENCHMARK_60_40.get(t, 1.0 / self.n_assets)
                    for t in self.tickers
                ])
                bm_weights = bm_weights / bm_weights.sum()
                sixty_forty_ret = float(np.dot(bm_weights, pr.values))
                self.benchmark_returns['sixty_forty'].append(sixty_forty_ret)
            else:
                self.benchmark_returns['spy'].append(0.0)
                self.benchmark_returns['equal_weight'].append(0.0)
                self.benchmark_returns['sixty_forty'].append(0.0)

            # Store weights
            self.portfolio_weights.append(
                {t: float(w) for t, w in zip(self.tickers, weights)}
            )
            prev_weights = weights.copy()

            results.append({
                'date': date,
                'portfolio_return': port_ret,
                'spy_return': self.benchmark_returns['spy'][-1],
                'ew_return': self.benchmark_returns['equal_weight'][-1],
                '60_40_return': self.benchmark_returns['sixty_forty'][-1],
                'turnover': turnover,
                'agreement': rnd.agent_agreement,
            })

        return pd.DataFrame(results).set_index('date')

    def compute_metrics(self, returns_series: List[float],
                        label: str = "Portfolio",
                        periods_per_year: float = 12.0) -> dict:
        """
        Compute full performance metrics suite for a return series.

        Metrics: total return, CAGR, volatility, Sharpe, Sortino,
                 max drawdown, Calmar ratio, win rate, average win/loss.
        """
        rets = np.array(returns_series)
        n = len(rets)
        if n == 0:
            return {'label': label, 'n_periods': 0}

        # Cumulative returns
        cum = np.cumprod(1 + rets)
        total_ret = cum[-1] - 1

        # CAGR
        years = n / periods_per_year
        cagr = (cum[-1]) ** (1.0 / max(years, 0.01)) - 1 if cum[-1] > 0 else -1.0

        # Volatility (annualized)
        vol = np.std(rets) * np.sqrt(periods_per_year)

        # Sharpe (assuming 0 risk-free rate for simplicity)
        sharpe = (np.mean(rets) * periods_per_year) / max(vol, 1e-8)

        # Sortino (downside deviation)
        downside = rets[rets < 0]
        downside_vol = np.std(downside) * np.sqrt(periods_per_year) if len(downside) > 0 else 1e-8
        sortino = (np.mean(rets) * periods_per_year) / max(downside_vol, 1e-8)

        # Max drawdown
        peak = np.maximum.accumulate(cum)
        drawdowns = (cum - peak) / peak
        max_dd = float(np.min(drawdowns))

        # Calmar
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

        # Win rate
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        win_rate = len(wins) / n if n > 0 else 0.0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        # Average turnover
        avg_turnover = (float(np.mean(self.turnover_history))
                        if self.turnover_history else 0.0)

        return {
            'label': label,
            'n_periods': n,
            'total_return': float(total_ret),
            'cagr': float(cagr),
            'annualized_vol': float(vol),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_dd),
            'calmar_ratio': float(calmar),
            'win_rate': float(win_rate),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_turnover': avg_turnover,
        }

    def get_all_metrics(self) -> pd.DataFrame:
        """Compute metrics for portfolio and all benchmarks."""
        rows = []
        rows.append(self.compute_metrics(self.portfolio_returns, "MiroFish Ensemble"))
        rows.append(self.compute_metrics(self.benchmark_returns['spy'], "SPY"))
        rows.append(self.compute_metrics(
            self.benchmark_returns['equal_weight'], "Equal Weight"))
        rows.append(self.compute_metrics(
            self.benchmark_returns['sixty_forty'], "60/40"))
        return pd.DataFrame(rows).set_index('label')

    def get_drawdown_series(self) -> pd.DataFrame:
        """Compute drawdown time series for portfolio and benchmarks."""
        result = {}
        for name, rets in [('portfolio', self.portfolio_returns),
                           ('spy', self.benchmark_returns['spy']),
                           ('equal_weight', self.benchmark_returns['equal_weight']),
                           ('sixty_forty', self.benchmark_returns['sixty_forty'])]:
            cum = np.cumprod(1 + np.array(rets))
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            result[name] = dd

        df = pd.DataFrame(result)
        if self.dates and len(self.dates) == len(df):
            df.index = self.dates
        return df


# ============================================================================
# AGREEMENT-BASED RISK MANAGER
# ============================================================================

class AgreementBasedRiskManager:
    """
    Uses agent agreement to dynamically scale position sizes.

    When agents strongly agree (high agreement), the manager allows full
    equity positions. When agents disagree (low agreement), the manager
    reduces equity exposure and moves to cash.

    Agreement thresholds:
      - High (>0.3): Full positions (100% invested)
      - Medium (0.1 - 0.3): Moderate reduction (75% invested, 25% cash proxy)
      - Low (<0.1): Conservative (50% invested, 50% cash proxy)
    """

    def __init__(self,
                 high_agreement: float = 0.3,
                 low_agreement: float = 0.1,
                 cash_proxy_tickers: Optional[List[str]] = None,
                 min_equity_fraction: float = 0.5):
        self.high_agreement = high_agreement
        self.low_agreement = low_agreement
        self.cash_proxy_tickers = cash_proxy_tickers or ['AGG', 'TIP']
        self.min_equity_fraction = min_equity_fraction
        self._overlay_history: List[dict] = []

    def _compute_scale_factor(self, agreement: float) -> float:
        """
        Map agreement level to a position scaling factor.

        Returns a value between min_equity_fraction and 1.0.
        """
        if agreement >= self.high_agreement:
            return 1.0
        elif agreement <= self.low_agreement:
            return self.min_equity_fraction
        else:
            # Linear interpolation between thresholds
            frac = ((agreement - self.low_agreement) /
                    (self.high_agreement - self.low_agreement))
            return self.min_equity_fraction + frac * (1.0 - self.min_equity_fraction)

    def adjust_weights(self, weights: np.ndarray,
                       agreement: float,
                       tickers: Optional[List[str]] = None) -> np.ndarray:
        """
        Adjust portfolio weights based on agent agreement level.

        Low agreement: reduce risky positions, increase cash proxy allocation.
        High agreement: keep positions as-is.
        """
        scale = self._compute_scale_factor(agreement)

        self._overlay_history.append({
            'agreement': float(agreement),
            'scale_factor': float(scale),
        })

        if scale >= 0.99:
            # Full positions, no adjustment needed
            return weights

        # Scale all weights down
        adjusted = weights * scale

        # Redistribute the freed weight to cash proxy assets
        freed = 1.0 - adjusted.sum()
        if tickers is not None and freed > 0:
            cash_indices = [i for i, t in enumerate(tickers)
                           if t in self.cash_proxy_tickers]
            if cash_indices:
                per_cash = freed / len(cash_indices)
                for ci in cash_indices:
                    adjusted[ci] += per_cash
            else:
                # No cash proxy found, distribute equally
                adjusted += freed / len(adjusted)
        elif freed > 0:
            adjusted += freed / len(adjusted)

        # Re-normalize
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total

        return adjusted

    def get_overlay_history(self) -> pd.DataFrame:
        """Return the risk overlay history as a DataFrame."""
        return pd.DataFrame(self._overlay_history)


# ============================================================================
# SIMULATION REPORT GENERATOR
# ============================================================================

class SimulationReportGenerator:
    """
    Generates detailed text report from simulation and backtest results.

    Sections:
      1. Simulation Summary
      2. Per-Agent Performance Attribution
      3. Regime Analysis
      4. Agreement-Return Relationship
      5. Key Findings and Recommendations
    """

    def __init__(self, runner: FinancialSimulationRunner,
                 backtester: Optional[SimulationBacktester] = None,
                 risk_manager: Optional[AgreementBasedRiskManager] = None):
        self.runner = runner
        self.backtester = backtester
        self.risk_manager = risk_manager

    def _section_header(self, title: str) -> str:
        """Format a section header."""
        return f"\n{'=' * 70}\n  {title}\n{'=' * 70}\n"

    def _simulation_summary(self) -> str:
        """Section 1: Overall simulation summary."""
        lines = [self._section_header("SIMULATION SUMMARY")]

        if not self.runner.rounds:
            lines.append("  No simulation rounds available.")
            return "\n".join(lines)

        first = self.runner.rounds[0]
        last = self.runner.rounds[-1]

        lines.extend([
            f"  Period:        {first.date} to {last.date}",
            f"  Total rounds:  {len(self.runner.rounds)}",
            f"  Active agents: {len(self.runner.agents)}",
            "",
            "  Agent Types:",
        ])

        type_counts = {}
        for a in self.runner.agents:
            t = a.profile.agent_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        for atype, count in sorted(type_counts.items()):
            lines.append(f"    {atype:15s}: {count}")

        agreements = [r.agent_agreement for r in self.runner.rounds]
        lines.extend([
            "",
            "  Agreement Statistics:",
            f"    Mean:   {np.mean(agreements):.4f}",
            f"    Std:    {np.std(agreements):.4f}",
            f"    Min:    {np.min(agreements):.4f}",
            f"    Max:    {np.max(agreements):.4f}",
            f"    Median: {np.median(agreements):.4f}",
        ])

        return "\n".join(lines)

    def _agent_performance_attribution(self) -> str:
        """Section 2: Per-agent performance attribution."""
        lines = [self._section_header("PER-AGENT PERFORMANCE ATTRIBUTION")]

        has_track = False
        for agent in self.runner.agents:
            tr = agent.profile.track_record
            if tr:
                has_track = True
                ics = [r['ic'] for r in tr if np.isfinite(r['ic'])]
                das = [r['dir_acc'] for r in tr if np.isfinite(r['dir_acc'])]
                avg_ic = np.mean(ics) if ics else 0.0
                std_ic = np.std(ics) if ics else 0.0
                avg_da = np.mean(das) if das else 0.0

                lines.extend([
                    f"  {agent.profile.name} ({agent.profile.agent_type.value})",
                    f"    Avg IC:           {avg_ic:+.4f} (std: {std_ic:.4f})",
                    f"    Avg Dir Accuracy: {avg_da:.2%}",
                    f"    Track records:    {len(tr)}",
                    f"    Confidence:       {agent.profile.confidence:.2f}",
                    "",
                ])

        if not has_track:
            lines.append("  No track record data available. Run simulation with ")
            lines.append("  track record updates to see per-agent attribution.")

        return "\n".join(lines)

    def _regime_analysis(self) -> str:
        """Section 3: Regime analysis (which agents work in which regime)."""
        lines = [self._section_header("REGIME ANALYSIS")]

        # Find regime agents and their history
        regime_agents = [a for a in self.runner.agents
                         if isinstance(a, RegimeAgent)]

        if not regime_agents:
            lines.append("  No RegimeAgent present in simulation.")
            lines.append("  Add a RegimeAgent to see regime-based analysis.")
            return "\n".join(lines)

        for ra in regime_agents:
            rh = ra.get_regime_history()
            if rh.empty:
                lines.append(f"  {ra.profile.name}: no regime history recorded.")
                continue

            regime_counts = rh['regime'].value_counts()
            lines.append(f"  {ra.profile.name} Regime Distribution:")
            for regime, count in regime_counts.items():
                pct = count / len(rh) * 100
                lines.append(f"    {regime:12s}: {count:4d} periods ({pct:.1f}%)")

            # VIX stats per regime
            lines.append("")
            lines.append("  VIX Statistics by Regime:")
            for regime in rh['regime'].unique():
                subset = rh[rh['regime'] == regime]
                lines.append(
                    f"    {regime:12s}: mean VIX={subset['vix'].mean():.1f}, "
                    f"range=[{subset['vix'].min():.1f}, {subset['vix'].max():.1f}]"
                )

        return "\n".join(lines)

    def _agreement_return_analysis(self) -> str:
        """Section 4: Relationship between agent agreement and returns."""
        lines = [self._section_header("AGREEMENT-RETURN RELATIONSHIP")]

        if self.backtester is None or not self.backtester.portfolio_returns:
            lines.append("  No backtest results available.")
            lines.append("  Run SimulationBacktester to see agreement analysis.")
            return "\n".join(lines)

        agreements = np.array([r.agent_agreement for r in self.runner.rounds])
        returns = np.array(self.backtester.portfolio_returns)

        n = min(len(agreements), len(returns))
        agreements = agreements[:n]
        returns = returns[:n]

        # Correlation between agreement and returns
        if n > 2:
            corr = np.corrcoef(agreements, returns)[0, 1]
            lines.append(f"  Correlation (agreement vs returns): {corr:+.4f}")
        else:
            lines.append("  Insufficient data for correlation.")

        # Split into agreement terciles
        if n >= 9:
            tercile_1 = np.percentile(agreements, 33)
            tercile_2 = np.percentile(agreements, 67)

            low_mask = agreements <= tercile_1
            mid_mask = (agreements > tercile_1) & (agreements <= tercile_2)
            high_mask = agreements > tercile_2

            lines.extend([
                "",
                "  Returns by Agreement Tercile:",
                f"    Low  agreement (<={tercile_1:.3f}): "
                f"mean ret={np.mean(returns[low_mask]):.4f}, "
                f"n={low_mask.sum()}",
                f"    Mid  agreement:                 "
                f"mean ret={np.mean(returns[mid_mask]):.4f}, "
                f"n={mid_mask.sum()}",
                f"    High agreement (>={tercile_2:.3f}): "
                f"mean ret={np.mean(returns[high_mask]):.4f}, "
                f"n={high_mask.sum()}",
            ])

        # Risk overlay impact
        if self.risk_manager is not None:
            overlay = self.risk_manager.get_overlay_history()
            if not overlay.empty:
                lines.extend([
                    "",
                    "  Risk Overlay Statistics:",
                    f"    Mean scale factor: {overlay['scale_factor'].mean():.3f}",
                    f"    Min scale factor:  {overlay['scale_factor'].min():.3f}",
                    f"    Periods at full allocation: "
                    f"{(overlay['scale_factor'] >= 0.99).sum()}/{len(overlay)}",
                ])

        return "\n".join(lines)

    def _key_findings(self) -> str:
        """Section 5: Key findings and recommendations."""
        lines = [self._section_header("KEY FINDINGS AND RECOMMENDATIONS")]

        findings = []

        # Check if backtest beat benchmarks
        if self.backtester is not None and self.backtester.portfolio_returns:
            metrics = self.backtester.get_all_metrics()
            if 'MiroFish Ensemble' in metrics.index and 'SPY' in metrics.index:
                port_sharpe = metrics.loc['MiroFish Ensemble', 'sharpe_ratio']
                spy_sharpe = metrics.loc['SPY', 'sharpe_ratio']

                if port_sharpe > spy_sharpe:
                    findings.append(
                        f"  [+] Ensemble outperformed SPY on risk-adjusted basis "
                        f"(Sharpe: {port_sharpe:.3f} vs {spy_sharpe:.3f})"
                    )
                else:
                    findings.append(
                        f"  [-] Ensemble underperformed SPY on risk-adjusted basis "
                        f"(Sharpe: {port_sharpe:.3f} vs {spy_sharpe:.3f})"
                    )

                port_dd = metrics.loc['MiroFish Ensemble', 'max_drawdown']
                spy_dd = metrics.loc['SPY', 'max_drawdown']
                if abs(port_dd) < abs(spy_dd):
                    findings.append(
                        f"  [+] Lower max drawdown than SPY "
                        f"({port_dd:.2%} vs {spy_dd:.2%})"
                    )
                else:
                    findings.append(
                        f"  [-] Higher max drawdown than SPY "
                        f"({port_dd:.2%} vs {spy_dd:.2%})"
                    )

        # Check agent agreement insights
        if self.runner.rounds:
            agreements = [r.agent_agreement for r in self.runner.rounds]
            if np.std(agreements) > 0.1:
                findings.append(
                    "  [*] High variance in agent agreement suggests "
                    "market regime shifts during the period"
                )
            if np.mean(agreements) < 0.15:
                findings.append(
                    "  [*] Low average agreement indicates persistent uncertainty; "
                    "consider adding more diverse agents"
                )

        # Recommendations
        findings.extend([
            "",
            "  Recommendations:",
            "    1. Monitor agreement level as a real-time risk indicator",
            "    2. Consider regime-conditional agent weighting",
            "    3. Use agreement-based risk overlay to manage drawdowns",
            "    4. Regularly retrain ML models on expanding window",
        ])

        lines.extend(findings)
        return "\n".join(lines)

    def generate_full_report(self) -> str:
        """Generate the complete multi-section report."""
        sections = [
            "",
            "=" * 70,
            "  MIROFISH FINANCIAL SIMULATION -- FULL REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            self._simulation_summary(),
            self._agent_performance_attribution(),
            self._regime_analysis(),
            self._agreement_return_analysis(),
            self._key_findings(),
            "",
            "=" * 70,
            "  END OF REPORT",
            "=" * 70,
        ]
        return "\n".join(sections)


# ============================================================================
# CLI: Run standalone financial simulation
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    from src.config import RAW_DIR as THESIS_RAW

    print("=" * 70)
    print("  MiroFish Financial Market Simulator")
    print("  Multi-Agent Simulation + Backtest + Report")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. Load market data
    # ------------------------------------------------------------------
    print("[1/6] Loading market data...")
    prices = pd.read_csv(
        os.path.join(str(THESIS_RAW), 'prices.csv'),
        index_col=0, parse_dates=True
    ).ffill().bfill()
    macro = pd.read_csv(
        os.path.join(str(THESIS_RAW), 'macro.csv'),
        index_col=0, parse_dates=True
    ).ffill()
    print(f"  Prices: {prices.shape[0]} days, {prices.shape[1]} assets")
    print(f"  Macro:  {macro.shape[0]} days, {macro.shape[1]} indicators")

    # Pre-load features for ML agents
    features_cache = None
    if os.path.exists(FEATURES_PATH):
        features_cache = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
        print(f"  Features: {features_cache.shape[0]} days, "
              f"{features_cache.shape[1]} features")

    # ------------------------------------------------------------------
    # 2. Create agents (including ML agents if models exist)
    # ------------------------------------------------------------------
    print("\n[2/6] Creating market agents...")
    runner = FinancialSimulationRunner()
    runner.create_default_agents()

    # Attempt to load ML models and create ML agents
    print("\n  Scanning for pre-trained ML models...")
    ml_agents = create_ml_agents(RESULTS_DIR, features_cache=features_cache)
    for agent in ml_agents:
        runner.add_agent(agent)

    print(f"\n  Total agents: {len(runner.agents)}")

    # ------------------------------------------------------------------
    # 3. Run simulation
    # ------------------------------------------------------------------
    print("\n[3/6] Running multi-agent simulation...")
    simulation_rounds = runner.run_simulation(
        prices, macro,
        start_date='2022-01-01',
        rebalance_freq=21,
    )

    # ------------------------------------------------------------------
    # 4. Run backtest on simulation output
    # ------------------------------------------------------------------
    print("\n[4/6] Running portfolio backtest...")
    risk_mgr = AgreementBasedRiskManager(
        high_agreement=0.3,
        low_agreement=0.1,
        cash_proxy_tickers=['AGG', 'TIP'],
        min_equity_fraction=0.5,
    )

    backtester = SimulationBacktester(
        prices,
        risk_aversion=1.0,
        max_weight=0.30,
        transaction_cost_bps=10.0,
    )

    backtest_df = backtester.run_backtest(
        simulation_rounds,
        agreement_risk_manager=risk_mgr,
    )

    # Print metrics
    metrics = backtester.get_all_metrics()
    print("\n  Performance Metrics:")
    print(metrics.to_string())

    # ------------------------------------------------------------------
    # 5. Generate report
    # ------------------------------------------------------------------
    print("\n[5/6] Generating simulation report...")
    report_gen = SimulationReportGenerator(runner, backtester, risk_mgr)
    full_report = report_gen.generate_full_report()
    print(full_report)

    # ------------------------------------------------------------------
    # 6. Save all results
    # ------------------------------------------------------------------
    print("\n[6/6] Saving results...")
    results_dir = str(THESIS_RAW).replace('raw', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save simulation JSON
    sim_path = os.path.join(results_dir, 'mirofish_simulation.json')
    runner.save_simulation(sim_path)

    # Save backtest results CSV
    bt_path = os.path.join(results_dir, 'mirofish_backtest.csv')
    backtest_df.to_csv(bt_path)
    print(f"  Backtest results saved to {bt_path}")

    # Save metrics CSV
    metrics_path = os.path.join(results_dir, 'mirofish_metrics.csv')
    metrics.to_csv(metrics_path)
    print(f"  Metrics saved to {metrics_path}")

    # Save drawdown series
    dd_path = os.path.join(results_dir, 'mirofish_drawdowns.csv')
    backtester.get_drawdown_series().to_csv(dd_path)
    print(f"  Drawdowns saved to {dd_path}")

    # Save report as text
    report_path = os.path.join(results_dir, 'mirofish_report.txt')
    with open(report_path, 'w') as f:
        f.write(full_report)
    print(f"  Report saved to {report_path}")

    # Save weights history
    weights_path = os.path.join(results_dir, 'mirofish_weights.json')
    with open(weights_path, 'w') as f:
        json.dump(backtester.portfolio_weights, f, indent=2, default=str)
    print(f"  Weights history saved to {weights_path}")

    print("\n" + "=" * 70)
    print("  Simulation complete. All results saved.")
    print("=" * 70)
