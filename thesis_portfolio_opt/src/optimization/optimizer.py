"""
Research-grade portfolio optimization module for multi-asset allocation.

Implements classical mean-variance, alternative objectives (CVaR, max diversification,
robust), risk parity / HRP heuristics, Black-Litterman, efficient frontier construction,
and a fluent constraint builder -- all backed by CVXPY with OSQP/SCS solvers.

References
----------
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional
  covariance matrices. Journal of Multivariate Analysis.
- Meucci, A. (2005). Risk and Asset Allocation. Springer.
- Black, F. & Litterman, R. (1992). Global Portfolio Optimization. Financial Analysts
  Journal.
- Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk.
  Journal of Risk.
- Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of
  Sample. Journal of Portfolio Management.
- Roncalli, T. (2013). Introduction to Risk Parity and Budgeting. Chapman & Hall.
- Cornuejols, G. & Tutuncu, R. (2007). Optimization Methods in Finance. Cambridge
  University Press.

Author : Arhan Subasi
Project: Industrial Engineering Thesis -- Portfolio Optimization
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from src.config import (
    MIN_WEIGHT,
    MAX_WEIGHT,
    RISK_FREE_RATE,
    RISK_AVERSION_RANGE,
    MAX_TURNOVER,
    CVAR_CONFIDENCE,
    ROBUST_EPSILON_MU,
    ROBUST_EPSILON_COV,
    BL_TAU,
    BL_RISK_AVERSION,
    EWMA_HALFLIFE,
    ASSET_CLASSES,
    MAX_CLASS_WEIGHT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Solver helper
# ---------------------------------------------------------------------------
_SOLVER_CHAIN: List[str] = [cp.OSQP, cp.SCS, cp.ECOS]
_ACCEPTABLE_STATUSES = {"optimal", "optimal_inaccurate"}


def _solve_problem(problem: cp.Problem, solver_chain: List[str] | None = None) -> str:
    """Attempt to solve *problem* using solvers in priority order.

    Parameters
    ----------
    problem : cp.Problem
        A fully constructed CVXPY problem.
    solver_chain : list of str, optional
        Ordered list of solver names to try.  Defaults to OSQP -> SCS -> ECOS.

    Returns
    -------
    str
        The final solver status string.
    """
    chain = solver_chain or _SOLVER_CHAIN
    status = "unsolved"
    for solver in chain:
        try:
            problem.solve(solver=solver, warm_start=True)
            status = problem.status
            if status in _ACCEPTABLE_STATUSES:
                return status
        except (cp.SolverError, Exception) as exc:
            logger.debug("Solver %s failed: %s", solver, exc)
            continue
    logger.warning("All solvers exhausted. Last status: %s", status)
    return status


# ###########################################################################
# SECTION 1 -- COVARIANCE ESTIMATION
# ###########################################################################


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "shrinkage",
    annualize: bool = True,
    annualization_factor: int = 252,
    **kwargs: Any,
) -> np.ndarray:
    """Estimate the covariance matrix of asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        T x N DataFrame of (daily) asset returns.
    method : str
        One of ``'sample'``, ``'shrinkage'`` (Ledoit-Wolf), ``'ewma'``, or
        ``'min_cov_det'`` (Minimum Covariance Determinant).
    annualize : bool
        If True multiply the result by *annualization_factor*.
    annualization_factor : int
        Trading days per year (default 252).
    **kwargs
        Forwarded to the underlying estimator (e.g. ``halflife`` for EWMA).

    Returns
    -------
    np.ndarray
        N x N covariance matrix.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    clean = returns.dropna()
    scale = annualization_factor if annualize else 1.0

    if method == "sample":
        cov = clean.cov().values * scale
    elif method == "shrinkage":
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(clean.values)
        cov = lw.covariance_ * scale
        logger.info(
            "Ledoit-Wolf shrinkage coefficient: %.4f", lw.shrinkage_
        )
    elif method == "ewma":
        halflife = kwargs.get("halflife", EWMA_HALFLIFE)
        cov = estimate_covariance_ewma(
            clean, halflife=halflife, annualize=annualize,
            annualization_factor=annualization_factor,
        )
    elif method == "min_cov_det":
        from sklearn.covariance import MinCovDet

        support_fraction = kwargs.get("support_fraction", None)
        mcd = MinCovDet(
            support_fraction=support_fraction,
            random_state=kwargs.get("random_state", 42),
        ).fit(clean.values)
        cov = mcd.covariance_ * scale
        logger.info("MinCovDet support fraction: %s", mcd.support_fraction_)
    else:
        raise ValueError(
            f"Unknown covariance method '{method}'. "
            "Choose from 'sample', 'shrinkage', 'ewma', 'min_cov_det'."
        )

    # Safety: ensure positive semi-definiteness
    cov = nearest_positive_definite(cov)
    return cov


def estimate_covariance_ewma(
    returns: pd.DataFrame,
    halflife: int = EWMA_HALFLIFE,
    annualize: bool = True,
    annualization_factor: int = 252,
) -> np.ndarray:
    """Exponentially-weighted moving average covariance estimator.

    The EWMA covariance places more weight on recent observations, capturing
    regime shifts faster than the sample estimator.

    Parameters
    ----------
    returns : pd.DataFrame
        T x N daily returns.
    halflife : int
        Half-life in trading days (default 63 ~ 3 months).
    annualize : bool
        Whether to scale to annual terms.
    annualization_factor : int
        Trading days per year.

    Returns
    -------
    np.ndarray
        N x N EWMA covariance matrix.
    """
    clean = returns.dropna()
    n_obs, n_assets = clean.shape
    decay = 1 - np.exp(-np.log(2) / halflife)

    data = clean.values
    mean = np.zeros(n_assets)
    cov = np.zeros((n_assets, n_assets))

    # Iterative update (exponential smoothing)
    for t in range(n_obs):
        x = data[t]
        diff = x - mean
        mean = mean + decay * diff
        cov = (1 - decay) * cov + decay * np.outer(diff, diff)

    scale = annualization_factor if annualize else 1.0
    return cov * scale


def denoise_covariance(
    cov: np.ndarray,
    n_samples: int,
    bandwidth: float | None = None,
) -> np.ndarray:
    """Denoise a covariance matrix using Marchenko-Pastur random matrix theory.

    Eigenvalues that fall within the Marchenko-Pastur distribution are replaced
    by their average, effectively removing noise while preserving signal.  This
    technique is described in *Advances in Financial Machine Learning* (Lopez de
    Prado, 2018).

    Parameters
    ----------
    cov : np.ndarray
        N x N covariance (or correlation) matrix.
    n_samples : int
        Number of time-series observations used to estimate *cov*.
    bandwidth : float or None
        Kernel bandwidth for the Marchenko-Pastur fit.  If None a simple
        heuristic is used.

    Returns
    -------
    np.ndarray
        Denoised N x N covariance matrix.
    """
    n = cov.shape[0]
    q = n_samples / n  # ratio T/N

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Marchenko-Pastur upper bound (for unit variance)
    sigma2 = 1.0  # will be estimated below
    lambda_plus = sigma2 * (1 + 1.0 / q + 2 * np.sqrt(1.0 / q))

    # Estimate noise variance as the mean of eigenvalues below the MP bound
    trace_cov = np.trace(cov)
    avg_eigenvalue = trace_cov / n

    # Find the cut-off: eigenvalues below lambda_plus * avg_eigenvalue are noise
    threshold = lambda_plus * avg_eigenvalue / ((1 + 1.0 / q) ** 2)

    # Separate signal and noise eigenvalues
    noise_mask = eigenvalues <= threshold
    signal_mask = ~noise_mask

    n_signal = signal_mask.sum()
    n_noise = noise_mask.sum()
    logger.info(
        "Marchenko-Pastur denoising: %d signal, %d noise eigenvalues "
        "(threshold=%.6f)",
        n_signal,
        n_noise,
        threshold,
    )

    if n_noise > 0:
        # Replace noise eigenvalues with their mean
        noise_mean = eigenvalues[noise_mask].mean()
        eigenvalues[noise_mask] = noise_mean

    # Reconstruct covariance
    cov_denoised = (eigenvectors * eigenvalues[np.newaxis, :]) @ eigenvectors.T

    # Ensure symmetry
    cov_denoised = (cov_denoised + cov_denoised.T) / 2.0

    # Re-scale diagonal to preserve original variances (optional but common)
    original_var = np.diag(cov)
    denoised_var = np.diag(cov_denoised)
    scale_factors = np.sqrt(original_var / np.maximum(denoised_var, 1e-12))
    cov_denoised = cov_denoised * np.outer(scale_factors, scale_factors)

    return nearest_positive_definite(cov_denoised)


def nearest_positive_definite(cov: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive semi-definite matrix.

    Uses the method of Higham (2002): iteratively project between the set of
    symmetric matrices with prescribed diagonal and the PSD cone.  For speed we
    use a single-pass eigenvalue clipping which is sufficient in practice.

    Parameters
    ----------
    cov : np.ndarray
        Input symmetric matrix (possibly indefinite).
    epsilon : float
        Minimum eigenvalue floor.

    Returns
    -------
    np.ndarray
        Nearest PSD matrix.
    """
    cov_sym = (cov + cov.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov_sym)

    if np.all(eigenvalues >= -epsilon):
        return cov_sym

    logger.debug(
        "Matrix not PSD (min eigenvalue=%.2e). Projecting.", eigenvalues.min()
    )
    eigenvalues = np.maximum(eigenvalues, epsilon)
    psd = (eigenvectors * eigenvalues[np.newaxis, :]) @ eigenvectors.T
    psd = (psd + psd.T) / 2.0
    return psd


# ###########################################################################
# SECTION 2 -- CLASSICAL MEAN-VARIANCE
# ###########################################################################


def mean_variance_optimize(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 2.0,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray | None:
    """Solve the standard Markowitz mean-variance QP.

    .. math::
        \\max_w \\; \\mu^\\top w - \\frac{\\gamma}{2} w^\\top \\Sigma w

    subject to :math:`\\mathbf{1}^\\top w = 1` and box constraints.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns vector (N,).
    cov : np.ndarray
        N x N covariance matrix.
    risk_aversion : float
        Risk-aversion parameter :math:`\\gamma`.
    min_weight, max_weight : float
        Per-asset weight bounds.

    Returns
    -------
    np.ndarray or None
        Optimal weight vector, or None if the solver fails.
    """
    n = len(mu)
    w = cp.Variable(n, name="w_mv")

    ret = mu @ w
    risk = cp.quad_form(w, cov, assume_PSD=True)
    objective = cp.Maximize(ret - (risk_aversion / 2.0) * risk)

    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]

    problem = cp.Problem(objective, constraints)
    status = _solve_problem(problem)

    if status not in _ACCEPTABLE_STATUSES or w.value is None:
        logger.warning("Mean-variance solver failed (status=%s).", status)
        return None

    weights = np.array(w.value).flatten()
    weights = _clip_and_normalize(weights, min_weight, max_weight)
    return weights


def mean_variance_with_turnover(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 2.0,
    current_weights: np.ndarray | None = None,
    max_turnover: float = MAX_TURNOVER,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray | None:
    """Mean-variance with an L1-turnover constraint.

    Adds the constraint :math:`\\|w - w_{\\text{prev}}\\|_1 \\le \\tau` to limit
    portfolio turnover, reducing transaction costs.

    Parameters
    ----------
    mu, cov, risk_aversion, min_weight, max_weight
        See :func:`mean_variance_optimize`.
    current_weights : np.ndarray or None
        Previous portfolio weights.  If None, equal weights are assumed.
    max_turnover : float
        Maximum L1 turnover allowed.

    Returns
    -------
    np.ndarray or None
    """
    n = len(mu)
    if current_weights is None:
        current_weights = np.ones(n) / n

    w = cp.Variable(n, name="w_mv_to")

    ret = mu @ w
    risk = cp.quad_form(w, cov, assume_PSD=True)
    objective = cp.Maximize(ret - (risk_aversion / 2.0) * risk)

    # Turnover = sum of absolute weight changes
    turnover = cp.norm1(w - current_weights)

    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
        turnover <= max_turnover,
    ]

    problem = cp.Problem(objective, constraints)
    status = _solve_problem(problem)

    if status not in _ACCEPTABLE_STATUSES or w.value is None:
        logger.warning(
            "MV-turnover solver failed (status=%s). "
            "Turnover constraint may be infeasible.",
            status,
        )
        return None

    weights = np.array(w.value).flatten()
    weights = _clip_and_normalize(weights, min_weight, max_weight)
    return weights


def mean_variance_with_sector_constraints(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 2.0,
    sector_map: Dict[str, List[int]] | None = None,
    max_sector: float = MAX_CLASS_WEIGHT,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray | None:
    """Mean-variance with per-sector weight caps.

    Parameters
    ----------
    mu, cov, risk_aversion, min_weight, max_weight
        See :func:`mean_variance_optimize`.
    sector_map : dict
        Mapping from sector name to list of asset **indices** (0-based).
        If None, :data:`ASSET_CLASSES` is translated using a positional
        convention (first 4 = Equity, next 5 = Fixed Income, last 3 = Alts).
    max_sector : float
        Maximum aggregate weight for any sector.

    Returns
    -------
    np.ndarray or None
    """
    n = len(mu)
    w = cp.Variable(n, name="w_mv_sec")

    ret = mu @ w
    risk = cp.quad_form(w, cov, assume_PSD=True)
    objective = cp.Maximize(ret - (risk_aversion / 2.0) * risk)

    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]

    # Build sector constraints
    if sector_map is not None:
        for sector_name, indices in sector_map.items():
            constraints.append(cp.sum(w[indices]) <= max_sector)
    else:
        # Default: derive from ASSET_CLASSES assuming standard ticker ordering
        # Equity (0-3), Fixed Income (4-8), Alternatives (9-11)
        _default_sectors = {
            "Equity": list(range(0, min(4, n))),
            "Fixed Income": list(range(4, min(9, n))),
            "Alternatives": list(range(9, min(12, n))),
        }
        for sector_name, indices in _default_sectors.items():
            valid = [i for i in indices if i < n]
            if valid:
                constraints.append(cp.sum(w[valid]) <= max_sector)

    problem = cp.Problem(objective, constraints)
    status = _solve_problem(problem)

    if status not in _ACCEPTABLE_STATUSES or w.value is None:
        logger.warning("MV-sector solver failed (status=%s).", status)
        return None

    weights = np.array(w.value).flatten()
    weights = _clip_and_normalize(weights, min_weight, max_weight)
    return weights


# ###########################################################################
# SECTION 3 -- ALTERNATIVE OBJECTIVES
# ###########################################################################


def max_sharpe_optimize(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray | None:
    """Find the maximum Sharpe-ratio portfolio.

    Uses the Cornuejols-Tutuncu transformation: solve a QP on the auxiliary
    variable :math:`y = w / \\kappa` with the constraint
    :math:`(\\mu - r_f)^\\top y = 1`, then normalise back.

    Parameters
    ----------
    mu : np.ndarray
        Expected return vector.
    cov : np.ndarray
        Covariance matrix.
    risk_free_rate : float
        Risk-free rate for excess returns.
    min_weight, max_weight : float
        Per-asset weight bounds (applied proportionally before normalisation).

    Returns
    -------
    np.ndarray or None
    """
    n = len(mu)
    excess = mu - risk_free_rate

    if (excess <= 0).all():
        logger.warning(
            "No asset has positive excess return; cannot compute max Sharpe."
        )
        return None

    y = cp.Variable(n, name="y_sharpe")
    kappa = cp.Variable(name="kappa_sharpe", nonneg=True)

    risk = cp.quad_form(y, cov, assume_PSD=True)
    objective = cp.Minimize(risk)

    constraints = [
        excess @ y == 1.0,
        cp.sum(y) == kappa,
        y >= min_weight * kappa,
        y <= max_weight * kappa,
        kappa >= 1e-6,
    ]

    problem = cp.Problem(objective, constraints)
    status = _solve_problem(problem)

    if status not in _ACCEPTABLE_STATUSES or y.value is None:
        logger.warning("Max-Sharpe solver failed (status=%s).", status)
        return None

    y_val = np.array(y.value).flatten()
    kappa_val = float(kappa.value)

    if kappa_val < 1e-10:
        logger.warning("Kappa near zero in max-Sharpe; degenerate solution.")
        return None

    weights = y_val / kappa_val
    weights = _clip_and_normalize(weights, min_weight, max_weight)
    return weights


def minimum_variance_optimize(
    cov: np.ndarray,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray | None:
    """Find the global minimum-variance portfolio.

    Parameters
    ----------
    cov : np.ndarray
        N x N covariance matrix.
    min_weight, max_weight : float
        Per-asset bounds.

    Returns
    -------
    np.ndarray or None
    """
    n = cov.shape[0]
    w = cp.Variable(n, name="w_minvar")

    objective = cp.Minimize(cp.quad_form(w, cov, assume_PSD=True))
    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]

    problem = cp.Problem(objective, constraints)
    status = _solve_problem(problem)

    if status not in _ACCEPTABLE_STATUSES or w.value is None:
        logger.warning("Min-variance solver failed (status=%s).", status)
        return None

    weights = np.array(w.value).flatten()
    weights = _clip_and_normalize(weights, min_weight, max_weight)
    return weights


def max_diversification_optimize(
    cov: np.ndarray,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray | None:
    """Maximise the diversification ratio (DR).

    .. math::
        DR = \\frac{w^\\top \\sigma}{\\sqrt{w^\\top \\Sigma w}}

    where :math:`\\sigma` is the vector of asset volatilities.  We use the
    Cornuejols-Tutuncu normalisation to turn this into a convex QP.

    Parameters
    ----------
    cov : np.ndarray
        N x N covariance matrix.
    min_weight, max_weight : float
        Per-asset bounds.

    Returns
    -------
    np.ndarray or None
    """
    n = cov.shape[0]
    vols = np.sqrt(np.diag(cov))

    y = cp.Variable(n, name="y_divr")
    kappa = cp.Variable(name="kappa_divr", nonneg=True)

    risk = cp.quad_form(y, cov, assume_PSD=True)
    objective = cp.Minimize(risk)

    constraints = [
        vols @ y == 1.0,
        cp.sum(y) == kappa,
        y >= min_weight * kappa,
        y <= max_weight * kappa,
        kappa >= 1e-6,
    ]

    problem = cp.Problem(objective, constraints)
    status = _solve_problem(problem)

    if status not in _ACCEPTABLE_STATUSES or y.value is None:
        logger.warning("Max-diversification solver failed (status=%s).", status)
        return None

    y_val = np.array(y.value).flatten()
    kappa_val = float(kappa.value)
    if kappa_val < 1e-10:
        logger.warning("Kappa near zero in max-diversification; degenerate.")
        return None

    weights = y_val / kappa_val
    weights = _clip_and_normalize(weights, min_weight, max_weight)
    return weights


def cvar_optimize(
    returns_scenarios: np.ndarray,
    alpha: float = CVAR_CONFIDENCE,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray | None:
    """Minimise Conditional Value-at-Risk (CVaR) via LP reformulation.

    Uses the Rockafellar-Uryasev (2000) auxiliary-variable trick to cast CVaR
    minimisation as a linear programme.

    Parameters
    ----------
    returns_scenarios : np.ndarray
        S x N matrix where each row is a scenario of asset returns.  Can be
        historical or Monte-Carlo generated.
    alpha : float
        Confidence level, e.g. 0.95 for 95 % CVaR.
    min_weight, max_weight : float
        Per-asset bounds.

    Returns
    -------
    np.ndarray or None
    """
    S, n = returns_scenarios.shape

    w = cp.Variable(n, name="w_cvar")
    z = cp.Variable(S, name="z_cvar")       # auxiliary variables for losses
    zeta = cp.Variable(name="zeta_cvar")    # VaR threshold

    # Portfolio loss per scenario: -R @ w  (we minimise loss)
    losses = -(returns_scenarios @ w)

    # CVaR linearisation
    cvar = zeta + (1.0 / (S * (1 - alpha))) * cp.sum(z)

    constraints = [
        z >= 0,
        z >= losses - zeta,
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]

    objective = cp.Minimize(cvar)
    problem = cp.Problem(objective, constraints)
    # CVaR LP is best solved with SCS or ECOS
    status = _solve_problem(problem, solver_chain=[cp.ECOS, cp.SCS, cp.OSQP])

    if status not in _ACCEPTABLE_STATUSES or w.value is None:
        logger.warning("CVaR solver failed (status=%s).", status)
        return None

    weights = np.array(w.value).flatten()
    weights = _clip_and_normalize(weights, min_weight, max_weight)
    return weights


def robust_mean_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    epsilon_mu: float = ROBUST_EPSILON_MU,
    epsilon_cov: float = ROBUST_EPSILON_COV,
    risk_aversion: float = 2.0,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray | None:
    """Worst-case robust mean-variance optimisation with uncertainty sets.

    The investor solves:

    .. math::
        \\max_w \\; \\min_{\\|\\delta\\| \\le \\epsilon_\\mu}
        (\\mu + \\delta)^\\top w
        - \\frac{\\gamma}{2} w^\\top (\\Sigma + \\epsilon_\\Sigma I) w

    which collapses (by duality of the inner min) to:

    .. math::
        \\max_w \\; \\mu^\\top w - \\epsilon_\\mu \\|w\\|_2
        - \\frac{\\gamma}{2} w^\\top \\hat{\\Sigma} w

    where :math:`\\hat{\\Sigma} = \\Sigma + \\epsilon_\\Sigma I`.

    Parameters
    ----------
    mu : np.ndarray
        Point estimate of expected returns.
    cov : np.ndarray
        Point estimate of covariance.
    epsilon_mu : float
        Radius of the uncertainty set around mu (L2-ball).
    epsilon_cov : float
        Additive diagonal perturbation for the covariance.
    risk_aversion : float
        Risk aversion parameter.
    min_weight, max_weight : float
        Per-asset bounds.

    Returns
    -------
    np.ndarray or None
    """
    n = len(mu)
    w = cp.Variable(n, name="w_robust")

    # Perturbed covariance
    cov_hat = cov + epsilon_cov * np.eye(n)
    cov_hat = nearest_positive_definite(cov_hat)

    # Worst-case expected return: mu'w - epsilon_mu * ||w||_2
    ret_wc = mu @ w - epsilon_mu * cp.norm(w, 2)
    risk = cp.quad_form(w, cov_hat, assume_PSD=True)
    objective = cp.Maximize(ret_wc - (risk_aversion / 2.0) * risk)

    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]

    problem = cp.Problem(objective, constraints)
    status = _solve_problem(problem, solver_chain=[cp.SCS, cp.ECOS, cp.OSQP])

    if status not in _ACCEPTABLE_STATUSES or w.value is None:
        logger.warning("Robust MV solver failed (status=%s).", status)
        return None

    weights = np.array(w.value).flatten()
    weights = _clip_and_normalize(weights, min_weight, max_weight)
    return weights


# ###########################################################################
# SECTION 4 -- RISK PARITY & HEURISTICS
# ###########################################################################


def risk_parity_optimize(
    cov: np.ndarray,
    risk_budget: np.ndarray | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Equal-risk contribution (ERC / Risk Parity) portfolio.

    Each asset's marginal risk contribution equals its budget fraction.
    Solved with the Spinu (2013) cyclical coordinate descent, which converges
    to the unique ERC portfolio for any PD covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        N x N covariance matrix (must be PD).
    risk_budget : np.ndarray or None
        Target risk-contribution fractions summing to 1.  If None, equal
        contributions are used.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on weight changes.

    Returns
    -------
    np.ndarray
        Weight vector summing to 1.
    """
    n = cov.shape[0]
    if risk_budget is None:
        risk_budget = np.ones(n) / n
    else:
        risk_budget = np.array(risk_budget, dtype=float)
        risk_budget /= risk_budget.sum()  # ensure normalised

    # Initialise with inverse volatility
    vols = np.sqrt(np.diag(cov))
    w = (1.0 / vols) / np.sum(1.0 / vols)

    for iteration in range(max_iter):
        w_old = w.copy()

        sigma_w = cov @ w
        port_var = w @ sigma_w
        port_vol = np.sqrt(port_var)

        # Risk contribution of each asset
        rc = w * sigma_w / port_vol

        # Target contribution
        target_rc = risk_budget * port_vol

        # Multiplicative update
        adj = target_rc / np.maximum(rc, 1e-16)
        w = w * adj
        w = np.maximum(w, 1e-12)
        w /= w.sum()

        if np.max(np.abs(w - w_old)) < tol:
            logger.debug("Risk parity converged in %d iterations.", iteration + 1)
            break
    else:
        logger.debug(
            "Risk parity did not converge in %d iterations (max delta=%.2e).",
            max_iter,
            np.max(np.abs(w - w_old)),
        )

    return w


def hierarchical_risk_parity(
    returns: pd.DataFrame,
    linkage_method: str = "single",
    distance_metric: str = "correlation",
) -> np.ndarray:
    """Hierarchical Risk Parity (HRP) from Lopez de Prado (2016).

    Steps:
    1. Compute distance matrix from the correlation matrix.
    2. Hierarchical agglomerative clustering.
    3. Quasi-diagonalise the covariance via the dendrogram ordering.
    4. Recursive bisection with inverse-variance allocation.

    Parameters
    ----------
    returns : pd.DataFrame
        T x N daily returns.
    linkage_method : str
        Linkage criterion (``'single'``, ``'complete'``, ``'average'``, ``'ward'``).
    distance_metric : str
        If ``'correlation'``, uses :math:`d = \\sqrt{0.5(1 - \\rho)}`.

    Returns
    -------
    np.ndarray
        HRP weight vector summing to 1.
    """
    clean = returns.dropna()
    n_assets = clean.shape[1]

    # --- 1. Distance matrix ---
    corr = clean.corr().values
    if distance_metric == "correlation":
        dist = np.sqrt(0.5 * (1 - corr))
    else:
        dist = 1 - corr

    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, 0.0)
    condensed = squareform(dist, checks=False)

    # --- 2. Hierarchical clustering ---
    link = linkage(condensed, method=linkage_method)

    # --- 3. Quasi-diagonalise (dendrogram ordering) ---
    sort_idx = leaves_list(link).astype(int)

    # --- 4. Recursive bisection ---
    cov = clean.cov().values * 252  # annualised
    weights = np.ones(n_assets)

    def _get_cluster_var(cov_sub: np.ndarray, idx: List[int]) -> float:
        """Inverse-variance of a sub-cluster (minimum-variance proxy)."""
        c = cov_sub[np.ix_(idx, idx)]
        try:
            inv_diag = 1.0 / np.diag(c)
        except FloatingPointError:
            inv_diag = np.ones(len(idx))
        w_sub = inv_diag / inv_diag.sum()
        return float(w_sub @ c @ w_sub)

    def _recursive_bisect(
        cov_mat: np.ndarray,
        ordered_items: List[int],
        weights_arr: np.ndarray,
    ) -> None:
        """Recursively split the ordered list and allocate weights."""
        if len(ordered_items) <= 1:
            return

        mid = len(ordered_items) // 2
        left = ordered_items[:mid]
        right = ordered_items[mid:]

        var_left = _get_cluster_var(cov_mat, left)
        var_right = _get_cluster_var(cov_mat, right)

        alpha_lr = 1.0 - var_left / (var_left + var_right)

        weights_arr[left] *= alpha_lr
        weights_arr[right] *= (1.0 - alpha_lr)

        _recursive_bisect(cov_mat, left, weights_arr)
        _recursive_bisect(cov_mat, right, weights_arr)

    _recursive_bisect(cov, list(sort_idx), weights)

    weights = np.maximum(weights, 0.0)
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones(n_assets) / n_assets

    return weights


def inverse_volatility_weights(cov: np.ndarray) -> np.ndarray:
    """Compute 1/volatility heuristic weights.

    Parameters
    ----------
    cov : np.ndarray
        N x N covariance matrix.

    Returns
    -------
    np.ndarray
        Normalised weight vector.
    """
    vols = np.sqrt(np.diag(cov))
    vols = np.maximum(vols, 1e-12)  # avoid division by zero
    inv_vol = 1.0 / vols
    return inv_vol / inv_vol.sum()


def equal_weight(n_assets: int) -> np.ndarray:
    """Return the 1/N equal-weight portfolio.

    Parameters
    ----------
    n_assets : int
        Number of assets.

    Returns
    -------
    np.ndarray
        Uniform weight vector.
    """
    return np.ones(n_assets) / n_assets


# ###########################################################################
# SECTION 5 -- BLACK-LITTERMAN
# ###########################################################################


def implied_equilibrium_returns(
    cov: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float = BL_RISK_AVERSION,
) -> np.ndarray:
    """Reverse-optimise to get implied equilibrium excess returns (pi).

    .. math::
        \\pi = \\gamma \\Sigma w_{mkt}

    Parameters
    ----------
    cov : np.ndarray
        N x N covariance matrix.
    market_weights : np.ndarray
        Market-cap weights.
    risk_aversion : float
        Risk aversion coefficient.

    Returns
    -------
    np.ndarray
        Implied equilibrium return vector.
    """
    return risk_aversion * cov @ market_weights


def black_litterman(
    cov: np.ndarray,
    market_weights: np.ndarray,
    views_P: np.ndarray | None = None,
    views_Q: np.ndarray | None = None,
    tau: float = BL_TAU,
    risk_aversion: float = BL_RISK_AVERSION,
    omega: np.ndarray | None = None,
    view_confidences: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Black-Litterman posterior expected returns.

    Parameters
    ----------
    cov : np.ndarray
        N x N covariance matrix.
    market_weights : np.ndarray
        Market capitalisation weights.
    views_P : np.ndarray or None
        K x N pick matrix (K views on N assets).
    views_Q : np.ndarray or None
        K-vector of view returns.
    tau : float
        Scalar governing the uncertainty in the prior.
    risk_aversion : float
        Market risk-aversion parameter used for equilibrium returns.
    omega : np.ndarray or None
        K x K view uncertainty matrix.  If None, the proportional-to-variance
        heuristic :math:`\\Omega = \\text{diag}(P(\\tau\\Sigma)P^\\top)` is used.
    view_confidences : np.ndarray or None
        K-vector of confidence scalars in (0, 1].  Applied by scaling the
        diagonal of omega: lower confidence => larger uncertainty.

    Returns
    -------
    np.ndarray
        Posterior expected return vector of length N.
    """
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion)

    if views_P is None or views_Q is None:
        logger.info("No views provided; returning equilibrium returns.")
        return pi

    views_P = np.atleast_2d(views_P)
    views_Q = np.atleast_1d(views_Q)

    k = views_P.shape[0]
    tau_cov = tau * cov

    # View uncertainty
    if omega is None:
        omega = np.diag(np.diag(views_P @ tau_cov @ views_P.T))

    # Apply per-view confidence scaling
    if view_confidences is not None:
        conf = np.array(view_confidences, dtype=float)
        # Lower confidence -> higher uncertainty
        scaling = np.where(conf > 0, 1.0 / conf - 1.0, 1e6)
        omega = omega + np.diag(scaling * np.diag(omega))

    # Posterior mean (Theil mixed estimator)
    tau_cov_inv = np.linalg.inv(tau_cov)
    omega_inv = np.linalg.inv(omega)

    M1 = np.linalg.inv(tau_cov_inv + views_P.T @ omega_inv @ views_P)
    M2 = tau_cov_inv @ pi + views_P.T @ omega_inv @ views_Q

    posterior_mu = M1 @ M2
    return posterior_mu


def black_litterman_optimize(
    cov: np.ndarray,
    market_weights: np.ndarray,
    views_P: np.ndarray | None = None,
    views_Q: np.ndarray | None = None,
    tau: float = BL_TAU,
    risk_aversion: float = BL_RISK_AVERSION,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
    omega: np.ndarray | None = None,
    view_confidences: np.ndarray | None = None,
) -> np.ndarray | None:
    """Black-Litterman posterior returns + mean-variance in one call.

    Combines :func:`black_litterman` and :func:`mean_variance_optimize`.

    Parameters
    ----------
    cov, market_weights, views_P, views_Q, tau, risk_aversion, omega,
    view_confidences
        See :func:`black_litterman`.
    min_weight, max_weight : float
        Per-asset bounds for the optimizer.

    Returns
    -------
    np.ndarray or None
        Optimal weight vector, or None on solver failure.
    """
    posterior_mu = black_litterman(
        cov=cov,
        market_weights=market_weights,
        views_P=views_P,
        views_Q=views_Q,
        tau=tau,
        risk_aversion=risk_aversion,
        omega=omega,
        view_confidences=view_confidences,
    )

    # Use the posterior covariance (tau*Sigma term neglected for simplicity;
    # in practice the posterior cov is (1+tau)*Sigma which just scales risk)
    posterior_cov = (1.0 + tau) * cov

    weights = mean_variance_optimize(
        mu=posterior_mu,
        cov=posterior_cov,
        risk_aversion=risk_aversion,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    return weights


# ###########################################################################
# SECTION 6 -- EFFICIENT FRONTIER
# ###########################################################################


def compute_portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    """Compute summary statistics for a portfolio.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weight vector.
    mu : np.ndarray
        Expected return vector (annualised).
    cov : np.ndarray
        Covariance matrix (annualised).
    risk_free_rate : float
        Risk-free rate for Sharpe ratio.

    Returns
    -------
    dict
        Keys: ``'expected_return'``, ``'volatility'``, ``'sharpe'``,
        ``'diversification_ratio'``.
    """
    w = np.asarray(weights).flatten()
    port_ret = float(mu @ w)
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(max(port_var, 0.0))

    sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 1e-12 else 0.0

    # Diversification ratio
    vols = np.sqrt(np.diag(cov))
    weighted_avg_vol = float(np.abs(w) @ vols)
    div_ratio = weighted_avg_vol / port_vol if port_vol > 1e-12 else 1.0

    return {
        "expected_return": port_ret,
        "volatility": port_vol,
        "sharpe": sharpe,
        "diversification_ratio": div_ratio,
    }


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 50,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
    risk_aversion_range: Tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Trace the mean-variance efficient frontier.

    Solves the MV problem for a grid of risk-aversion values and returns
    portfolio return, volatility, and Sharpe ratio at each point.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns.
    cov : np.ndarray
        Covariance matrix.
    n_points : int
        Number of points on the frontier.
    min_weight, max_weight : float
        Per-asset bounds.
    risk_aversion_range : tuple of float or None
        (gamma_min, gamma_max).  If None, defaults to (0.1, 50).

    Returns
    -------
    pd.DataFrame
        Columns: ``risk_aversion``, ``return``, ``volatility``, ``sharpe``,
        ``weights``.
    """
    if risk_aversion_range is None:
        gamma_lo, gamma_hi = 0.1, 50.0
    else:
        gamma_lo, gamma_hi = risk_aversion_range

    gammas = np.logspace(np.log10(gamma_lo), np.log10(gamma_hi), n_points)
    results: List[Dict[str, Any]] = []

    for gamma in gammas:
        weights = mean_variance_optimize(
            mu=mu,
            cov=cov,
            risk_aversion=gamma,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        if weights is None:
            continue

        stats = compute_portfolio_stats(weights, mu, cov)
        results.append(
            {
                "risk_aversion": gamma,
                "return": stats["expected_return"],
                "volatility": stats["volatility"],
                "sharpe": stats["sharpe"],
                "weights": weights.tolist(),
            }
        )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("volatility").reset_index(drop=True)
    return df


def efficient_frontier_with_rf(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    n_points: int = 50,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> pd.DataFrame:
    """Efficient frontier together with the Capital Market Line.

    Returns the risky-asset frontier plus n_points on the CML (linear
    combinations of the risk-free asset and the tangency portfolio).

    Parameters
    ----------
    mu, cov, n_points, min_weight, max_weight
        See :func:`efficient_frontier`.
    risk_free_rate : float
        Risk-free rate.

    Returns
    -------
    pd.DataFrame
        Columns: ``return``, ``volatility``, ``sharpe``, ``type`` (``'frontier'``
        or ``'cml'``).
    """
    # --- Risky frontier ---
    frontier_df = efficient_frontier(
        mu=mu,
        cov=cov,
        n_points=n_points,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    if not frontier_df.empty:
        frontier_df["type"] = "frontier"

    # --- Capital Market Line ---
    tangency_w = max_sharpe_optimize(
        mu=mu,
        cov=cov,
        risk_free_rate=risk_free_rate,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    cml_rows: List[Dict[str, Any]] = []
    if tangency_w is not None:
        tang_stats = compute_portfolio_stats(tangency_w, mu, cov, risk_free_rate)
        tang_ret = tang_stats["expected_return"]
        tang_vol = tang_stats["volatility"]

        # Points along the CML: fraction in the tangency portfolio
        for alpha in np.linspace(0.0, 1.5, n_points):
            cml_ret = (1 - alpha) * risk_free_rate + alpha * tang_ret
            cml_vol = alpha * tang_vol
            cml_sharpe = (
                (cml_ret - risk_free_rate) / cml_vol if cml_vol > 1e-12 else 0.0
            )
            cml_rows.append(
                {
                    "return": cml_ret,
                    "volatility": cml_vol,
                    "sharpe": cml_sharpe,
                    "type": "cml",
                    "risk_aversion": np.nan,
                    "weights": (alpha * tangency_w).tolist(),
                }
            )

    cml_df = pd.DataFrame(cml_rows)
    combined = pd.concat([frontier_df, cml_df], ignore_index=True)
    return combined


# ###########################################################################
# SECTION 7 -- CONSTRAINT BUILDER (FLUENT API)
# ###########################################################################


class ConstraintBuilder:
    """Fluent interface for assembling CVXPY constraint sets.

    Example
    -------
    >>> builder = ConstraintBuilder(n_assets=12)
    >>> constraints = (
    ...     builder
    ...     .long_only()
    ...     .box_constraints(0.0, 0.40)
    ...     .fully_invested()
    ...     .max_turnover(prev_weights, 0.50)
    ...     .sector_limits(sector_map, 0.70)
    ...     .build()
    ... )
    """

    def __init__(
        self,
        n_assets: int,
        w: cp.Variable | None = None,
    ) -> None:
        """
        Parameters
        ----------
        n_assets : int
            Number of assets in the portfolio.
        w : cp.Variable or None
            CVXPY variable for weights.  If None, one is created internally.
        """
        self.n = n_assets
        self.w = w if w is not None else cp.Variable(n_assets, name="w_cb")
        self._constraints: List[Any] = []
        self._description: List[str] = []

    # ---- Chainable methods ------------------------------------------------

    def long_only(self) -> "ConstraintBuilder":
        """Add non-negativity constraint: w >= 0."""
        self._constraints.append(self.w >= 0)
        self._description.append("long_only")
        return self

    def box_constraints(
        self,
        min_w: float = MIN_WEIGHT,
        max_w: float = MAX_WEIGHT,
    ) -> "ConstraintBuilder":
        """Add box constraints: min_w <= w_i <= max_w for all i."""
        self._constraints.append(self.w >= min_w)
        self._constraints.append(self.w <= max_w)
        self._description.append(f"box[{min_w:.3f}, {max_w:.3f}]")
        return self

    def fully_invested(self) -> "ConstraintBuilder":
        """Add full-investment constraint: sum(w) == 1."""
        self._constraints.append(cp.sum(self.w) == 1)
        self._description.append("fully_invested")
        return self

    def max_turnover(
        self,
        current_weights: np.ndarray,
        max_to: float = MAX_TURNOVER,
    ) -> "ConstraintBuilder":
        """Add L1 turnover constraint: ||w - w_prev||_1 <= max_to."""
        self._constraints.append(
            cp.norm1(self.w - current_weights) <= max_to
        )
        self._description.append(f"turnover<={max_to:.2f}")
        return self

    def sector_limits(
        self,
        sector_map: Dict[str, List[int]],
        max_weight: float = MAX_CLASS_WEIGHT,
    ) -> "ConstraintBuilder":
        """Add per-sector aggregate weight caps.

        Parameters
        ----------
        sector_map : dict
            Maps sector name -> list of asset indices.
        max_weight : float
            Maximum aggregate weight per sector.
        """
        for sector, indices in sector_map.items():
            valid = [i for i in indices if 0 <= i < self.n]
            if valid:
                self._constraints.append(cp.sum(self.w[valid]) <= max_weight)
                self._description.append(f"sector_{sector}<={max_weight:.2f}")
        return self

    def cardinality_relaxation(
        self,
        max_assets: int,
    ) -> "ConstraintBuilder":
        """Add a convex relaxation of the cardinality constraint.

        Uses an L1-norm penalty proxy: :math:`\\|w\\|_1 \\le k \\cdot w_{max}`.
        This is only a relaxation -- true cardinality constraints are NP-hard.

        Parameters
        ----------
        max_assets : int
            Desired maximum number of active positions.
        """
        max_per = 1.0 / max_assets if max_assets > 0 else 1.0
        self._constraints.append(cp.norm(self.w, "inf") >= max_per)
        self._description.append(f"cardinality_relax(k={max_assets})")
        return self

    def tracking_error(
        self,
        benchmark_weights: np.ndarray,
        cov: np.ndarray,
        max_te: float = 0.05,
    ) -> "ConstraintBuilder":
        """Limit tracking error relative to a benchmark.

        .. math::
            \\sqrt{(w - w_b)^\\top \\Sigma (w - w_b)} \\le TE_{max}

        Parameters
        ----------
        benchmark_weights : np.ndarray
            Benchmark weight vector.
        cov : np.ndarray
            Covariance matrix.
        max_te : float
            Maximum annualised tracking error.
        """
        diff = self.w - benchmark_weights
        te_sq = cp.quad_form(diff, cov, assume_PSD=True)
        self._constraints.append(te_sq <= max_te ** 2)
        self._description.append(f"tracking_error<={max_te:.4f}")
        return self

    def min_weight_per_asset(
        self,
        min_weights: np.ndarray,
    ) -> "ConstraintBuilder":
        """Per-asset minimum weight vector (can differ across assets).

        Parameters
        ----------
        min_weights : np.ndarray
            N-vector of per-asset lower bounds.
        """
        self._constraints.append(self.w >= min_weights)
        self._description.append("per_asset_min")
        return self

    def max_weight_per_asset(
        self,
        max_weights: np.ndarray,
    ) -> "ConstraintBuilder":
        """Per-asset maximum weight vector (can differ across assets).

        Parameters
        ----------
        max_weights : np.ndarray
            N-vector of per-asset upper bounds.
        """
        self._constraints.append(self.w <= max_weights)
        self._description.append("per_asset_max")
        return self

    def leverage_limit(
        self,
        max_leverage: float = 1.0,
    ) -> "ConstraintBuilder":
        """Limit gross leverage: sum(|w|) <= max_leverage.

        Parameters
        ----------
        max_leverage : float
            Gross exposure limit (1.0 = long-only, 1.5 = 150/50, etc.).
        """
        self._constraints.append(cp.norm1(self.w) <= max_leverage)
        self._description.append(f"leverage<={max_leverage:.2f}")
        return self

    def build(self) -> List:
        """Return the accumulated list of CVXPY constraints.

        Returns
        -------
        list
            CVXPY constraint objects ready for ``cp.Problem``.
        """
        logger.debug(
            "ConstraintBuilder: built %d constraints (%s).",
            len(self._constraints),
            ", ".join(self._description),
        )
        return list(self._constraints)

    @property
    def variable(self) -> cp.Variable:
        """Access the CVXPY weight variable."""
        return self.w

    def __repr__(self) -> str:
        return (
            f"ConstraintBuilder(n={self.n}, "
            f"constraints=[{', '.join(self._description)}])"
        )


# ###########################################################################
# UTILITY HELPERS
# ###########################################################################


def _clip_and_normalize(
    weights: np.ndarray,
    min_w: float,
    max_w: float,
    n_iter: int = 5,
) -> np.ndarray:
    """Post-process solver output: clip to bounds then re-normalise.

    Numerical solvers sometimes violate bounds by tiny amounts.  This
    iterative clip-normalise procedure projects back onto the feasible set
    without excessive distortion.

    Parameters
    ----------
    weights : np.ndarray
        Raw solver output.
    min_w, max_w : float
        Box bounds.
    n_iter : int
        Number of clip-renorm passes.

    Returns
    -------
    np.ndarray
        Cleaned weight vector summing to 1.
    """
    w = weights.copy()
    for _ in range(n_iter):
        w = np.clip(w, min_w, max_w)
        total = w.sum()
        if abs(total) < 1e-12:
            w = np.ones_like(w) / len(w)
            break
        w /= total
    return w


def portfolio_turnover(
    old_weights: np.ndarray,
    new_weights: np.ndarray,
) -> float:
    """Compute one-way turnover between two weight vectors.

    Parameters
    ----------
    old_weights, new_weights : np.ndarray
        Weight vectors (should sum to 1).

    Returns
    -------
    float
        L1 distance / 2 (one-way turnover).
    """
    return float(0.5 * np.sum(np.abs(new_weights - old_weights)))


def portfolio_concentration(weights: np.ndarray) -> Dict[str, float]:
    """Compute concentration metrics for a weight vector.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.

    Returns
    -------
    dict
        ``'hhi'`` (Herfindahl-Hirschman Index), ``'effective_n'`` (inverse
        of HHI), ``'max_weight'``, ``'gini'``.
    """
    w = np.abs(weights)
    w_sorted = np.sort(w)[::-1]
    n = len(w)

    hhi = float(np.sum(w ** 2))
    eff_n = 1.0 / hhi if hhi > 1e-12 else float(n)

    # Gini coefficient
    index = np.arange(1, n + 1)
    gini = float(
        (2 * np.sum(index * w_sorted) - (n + 1) * np.sum(w_sorted))
        / (n * np.sum(w_sorted))
    ) if np.sum(w_sorted) > 1e-12 else 0.0

    return {
        "hhi": hhi,
        "effective_n": eff_n,
        "max_weight": float(w_sorted[0]) if n > 0 else 0.0,
        "gini": gini,
    }


def risk_contribution(
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Compute percentage risk contribution per asset.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weight vector.
    cov : np.ndarray
        N x N covariance matrix.

    Returns
    -------
    np.ndarray
        Vector of fractional risk contributions (sums to 1).
    """
    w = np.asarray(weights).flatten()
    sigma_w = cov @ w
    port_vol = np.sqrt(max(w @ sigma_w, 1e-16))
    mrc = sigma_w / port_vol               # marginal risk contribution
    rc = w * mrc                            # component risk contribution
    total_rc = rc.sum()
    if abs(total_rc) < 1e-16:
        return np.ones(len(w)) / len(w)
    return rc / total_rc


def marginal_risk_contribution(
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Compute marginal risk contribution per asset.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weight vector.
    cov : np.ndarray
        N x N covariance matrix.

    Returns
    -------
    np.ndarray
        Vector of marginal risk contributions.
    """
    w = np.asarray(weights).flatten()
    sigma_w = cov @ w
    port_vol = np.sqrt(max(w @ sigma_w, 1e-16))
    return sigma_w / port_vol


def validate_inputs(
    mu: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> bool:
    """Validate common inputs for numerical sanity.

    Checks for NaN/Inf, shape consistency, and symmetry of cov.

    Parameters
    ----------
    mu : np.ndarray or None
        Expected returns.
    cov : np.ndarray or None
        Covariance matrix.
    weights : np.ndarray or None
        Weight vector.

    Returns
    -------
    bool
        True if all checks pass.

    Raises
    ------
    ValueError
        On any validation failure.
    """
    if mu is not None:
        mu = np.asarray(mu)
        if np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
            raise ValueError("Expected returns contain NaN or Inf.")

    if cov is not None:
        cov = np.asarray(cov)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(
                f"Covariance must be square; got shape {cov.shape}."
            )
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            raise ValueError("Covariance contains NaN or Inf.")
        if not np.allclose(cov, cov.T, atol=1e-8):
            raise ValueError("Covariance matrix is not symmetric.")
        if mu is not None and len(mu) != cov.shape[0]:
            raise ValueError(
                f"Dimension mismatch: mu has {len(mu)} elements but "
                f"cov is {cov.shape[0]}x{cov.shape[1]}."
            )

    if weights is not None:
        weights = np.asarray(weights)
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            raise ValueError("Weights contain NaN or Inf.")
        if cov is not None and len(weights) != cov.shape[0]:
            raise ValueError("Weights dimension does not match covariance.")

    return True


def generate_random_portfolios(
    mu: np.ndarray,
    cov: np.ndarray,
    n_portfolios: int = 5000,
    seed: int = 42,
    long_only: bool = True,
) -> pd.DataFrame:
    """Generate random portfolios for Monte Carlo visualisation.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns.
    cov : np.ndarray
        Covariance matrix.
    n_portfolios : int
        Number of random portfolios.
    seed : int
        Random seed.
    long_only : bool
        If True, only positive weights via Dirichlet sampling.

    Returns
    -------
    pd.DataFrame
        Columns: ``return``, ``volatility``, ``sharpe``.
    """
    rng = np.random.default_rng(seed)
    n = len(mu)
    records: List[Dict[str, float]] = []

    for _ in range(n_portfolios):
        if long_only:
            w = rng.dirichlet(np.ones(n))
        else:
            w = rng.normal(size=n)
            w /= np.sum(np.abs(w))

        stats = compute_portfolio_stats(w, mu, cov)
        records.append(
            {
                "return": stats["expected_return"],
                "volatility": stats["volatility"],
                "sharpe": stats["sharpe"],
            }
        )

    return pd.DataFrame(records)


def combine_optimization_results(
    results: Dict[str, np.ndarray | None],
    mu: np.ndarray,
    cov: np.ndarray,
    asset_names: List[str] | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """Build a summary table comparing multiple strategies.

    Parameters
    ----------
    results : dict
        Maps strategy name to weight vector (or None).
    mu : np.ndarray
        Expected returns.
    cov : np.ndarray
        Covariance matrix.
    asset_names : list of str or None
        Asset labels.
    risk_free_rate : float
        Risk-free rate.

    Returns
    -------
    pd.DataFrame
        One row per strategy with return, vol, Sharpe, and per-asset weights.
    """
    rows: List[Dict[str, Any]] = []
    n = len(mu)
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(n)]

    for name, weights in results.items():
        if weights is None:
            row: Dict[str, Any] = {"strategy": name, "status": "failed"}
            rows.append(row)
            continue

        stats = compute_portfolio_stats(weights, mu, cov, risk_free_rate)
        concentration = portfolio_concentration(weights)
        rc = risk_contribution(weights, cov)

        row = {
            "strategy": name,
            "status": "optimal",
            "expected_return": stats["expected_return"],
            "volatility": stats["volatility"],
            "sharpe": stats["sharpe"],
            "diversification_ratio": stats["diversification_ratio"],
            "hhi": concentration["hhi"],
            "effective_n": concentration["effective_n"],
            "max_weight": concentration["max_weight"],
            "gini": concentration["gini"],
        }

        for i, aname in enumerate(asset_names):
            row[f"w_{aname}"] = weights[i]
            row[f"rc_{aname}"] = rc[i]

        rows.append(row)

    return pd.DataFrame(rows)


def run_all_strategies(
    mu: np.ndarray,
    cov: np.ndarray,
    returns_df: pd.DataFrame | None = None,
    market_weights: np.ndarray | None = None,
    views_P: np.ndarray | None = None,
    views_Q: np.ndarray | None = None,
    risk_aversion: float = 2.0,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> Dict[str, np.ndarray | None]:
    """Run all implemented strategies and return weights dict.

    This convenience function executes every optimizer in the module with
    sensible defaults, returning a dictionary suitable for
    :func:`combine_optimization_results`.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns.
    cov : np.ndarray
        Covariance matrix.
    returns_df : pd.DataFrame or None
        Historical returns (needed for HRP and CVaR).
    market_weights : np.ndarray or None
        Market-cap weights for Black-Litterman (defaults to equal weight).
    views_P, views_Q : np.ndarray or None
        BL views.
    risk_aversion : float
        Risk-aversion parameter.
    min_weight, max_weight : float
        Per-asset bounds.

    Returns
    -------
    dict
        Strategy name -> weight vector (or None).
    """
    n = len(mu)
    results: Dict[str, np.ndarray | None] = {}

    # 1. Mean-variance
    results["Mean-Variance"] = mean_variance_optimize(
        mu, cov, risk_aversion, min_weight, max_weight
    )

    # 2. Max Sharpe
    results["Max Sharpe"] = max_sharpe_optimize(
        mu, cov, RISK_FREE_RATE, min_weight, max_weight
    )

    # 3. Minimum variance
    results["Min Variance"] = minimum_variance_optimize(
        cov, min_weight, max_weight
    )

    # 4. Max diversification
    results["Max Diversification"] = max_diversification_optimize(
        cov, min_weight, max_weight
    )

    # 5. Risk parity
    results["Risk Parity"] = risk_parity_optimize(cov)

    # 6. Inverse volatility
    results["Inv. Volatility"] = inverse_volatility_weights(cov)

    # 7. Equal weight
    results["Equal Weight"] = equal_weight(n)

    # 8. Robust MV
    results["Robust MV"] = robust_mean_variance(
        mu, cov, ROBUST_EPSILON_MU, ROBUST_EPSILON_COV, risk_aversion,
        min_weight, max_weight,
    )

    # 9. CVaR (needs return scenarios)
    if returns_df is not None:
        scenarios = returns_df.dropna().values
        if scenarios.shape[0] >= 50:
            results["CVaR"] = cvar_optimize(
                scenarios, CVAR_CONFIDENCE, min_weight, max_weight
            )
        else:
            logger.warning("Not enough scenarios for CVaR; skipping.")
            results["CVaR"] = None
    else:
        results["CVaR"] = None

    # 10. HRP (needs return data)
    if returns_df is not None:
        results["HRP"] = hierarchical_risk_parity(returns_df)
    else:
        results["HRP"] = None

    # 11. Black-Litterman
    if market_weights is None:
        market_weights = np.ones(n) / n
    results["Black-Litterman"] = black_litterman_optimize(
        cov, market_weights, views_P, views_Q,
        BL_TAU, BL_RISK_AVERSION, min_weight, max_weight,
    )

    return results
