"""Covariance estimation, risk decomposition and factor exposure."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import periods_per_year

SAMPLE = "Sample"
LEDOIT = "Ledoit-Wolf shrinkage"
EWMA = "EWMA (λ = 0.94)"
COV_METHODS = [LEDOIT, SAMPLE, EWMA]


# --------------------------------------------------------------------------- #
# Covariance
# --------------------------------------------------------------------------- #
def _ledoit_wolf(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage toward a constant-correlation target."""
    t, n = x.shape
    x = x - x.mean(axis=0)
    sample = (x.T @ x) / t
    var = np.diag(sample)
    std = np.sqrt(np.maximum(var, 1e-18))
    corr = sample / np.outer(std, std)
    off = corr[~np.eye(n, dtype=bool)]
    r_bar = off.mean() if off.size else 0.0
    target = r_bar * np.outer(std, std)
    np.fill_diagonal(target, var)

    # pi: sum of asymptotic variances of the sample covariance entries
    x2 = x ** 2
    pi_mat = (x2.T @ x2) / t - sample ** 2
    pi_hat = pi_mat.sum()

    # rho: covariance between the target and the sample estimator
    term = ((x ** 3).T @ x) / t - var[:, None] * sample
    rho_hat = np.diag(pi_mat).sum()
    if n > 1 and r_bar != 0:
        ratio = np.outer(std, 1.0 / std)
        contrib = 0.5 * r_bar * (ratio * term + ratio.T * term.T)
        np.fill_diagonal(contrib, 0.0)
        rho_hat += contrib.sum()

    gamma = float(((target - sample) ** 2).sum())
    if gamma <= 0:
        return sample, 0.0
    kappa = (pi_hat - rho_hat) / gamma
    shrink = float(np.clip(kappa / t, 0.0, 1.0))
    return shrink * target + (1.0 - shrink) * sample, shrink


def _ewma_cov(x: np.ndarray, lam: float = 0.94) -> np.ndarray:
    t, n = x.shape
    x = x - x.mean(axis=0)
    weights = lam ** np.arange(t - 1, -1, -1)
    weights /= weights.sum()
    xw = x * weights[:, None]
    return x.T @ xw


def covariance(returns: pd.DataFrame, method: str = LEDOIT,
               annualise: bool = True) -> pd.DataFrame:
    """Covariance matrix of the given return panel."""
    clean = returns.dropna(how="any")
    if clean.empty or clean.shape[0] < 3:
        n = returns.shape[1]
        return pd.DataFrame(np.eye(n) * 1e-6, index=returns.columns, columns=returns.columns)

    x = clean.to_numpy(dtype=float)
    if method == SAMPLE:
        cov = np.cov(x, rowvar=False, ddof=1)
    elif method == EWMA:
        cov = _ewma_cov(x)
    else:
        cov, _ = _ledoit_wolf(x)

    cov = np.atleast_2d(cov)
    if annualise:
        cov = cov * periods_per_year(clean.index)
    # Enforce symmetry and positive semi-definiteness.
    cov = (cov + cov.T) / 2.0
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    cov = vecs @ np.diag(vals) @ vecs.T
    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)


def expected_returns(returns: pd.DataFrame, method: str = "Historical mean",
                     cov: pd.DataFrame | None = None,
                     risk_aversion: float = 2.5,
                     shrink_to: float | None = None,
                     blend: float = 0.5) -> pd.Series:
    """Annualised expected returns under several estimators.

    * ``Historical mean``     — geometric annualised return of the sample.
    * ``Shrunk (James-Stein)``— pulls each fund toward the cross-sectional mean.
    * ``Equilibrium (reverse-optimised)`` — implied returns from equal risk
      budgets, i.e. ``λ · Σ · w_mkt`` with ``w_mkt`` equal-weight. Far more
      stable than sample means and standard practice in Black-Litterman.
    * ``Blend`` — half historical, half equilibrium.
    """
    clean = returns.dropna(how="any")
    ann = periods_per_year(clean.index) if len(clean) > 2 else 252.0
    hist = (1.0 + clean).prod() ** (ann / max(len(clean), 1)) - 1.0
    hist = hist.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if method == "Historical mean":
        return hist
    if method == "Shrunk (James-Stein)":
        grand = hist.mean()
        target = shrink_to if shrink_to is not None else grand
        n = len(hist)
        var = float(hist.var(ddof=1)) if n > 1 else 0.0
        if var <= 0:
            return hist
        factor = float(np.clip(1.0 - (n - 2) * (var / n) / max(((hist - target) ** 2).sum(), 1e-12),
                               0.0, 1.0))
        return target + factor * (hist - target)
    if cov is None:
        cov = covariance(clean)
    w_mkt = np.ones(len(clean.columns)) / len(clean.columns)
    eq = pd.Series(risk_aversion * (cov.to_numpy() @ w_mkt), index=clean.columns)
    if method == "Equilibrium (reverse-optimised)":
        return eq
    return blend * hist + (1.0 - blend) * eq


# --------------------------------------------------------------------------- #
# Risk decomposition
# --------------------------------------------------------------------------- #
def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(weights @ cov @ weights, 0.0)))


def risk_contributions(weights: Mapping[str, float], cov: pd.DataFrame) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility into per-fund contributions.

    Returns weight, marginal contribution (∂σ/∂w), contribution to risk
    (w · MCTR, which sums to σ) and the percentage share of total risk.
    """
    codes = [c for c in cov.columns if c in weights]
    w = np.array([weights[c] for c in codes], dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
    sigma = cov.loc[codes, codes].to_numpy()
    port_vol = portfolio_volatility(w, sigma)
    if port_vol <= 0:
        mctr = np.zeros_like(w)
    else:
        mctr = (sigma @ w) / port_vol
    ctr = w * mctr
    total = ctr.sum()
    return pd.DataFrame(
        {
            "weight": w,
            "marginal_risk": mctr,
            "risk_contribution": ctr,
            "risk_share": ctr / total if total > 0 else np.zeros_like(ctr),
        },
        index=codes,
    )


def group_risk_contributions(weights: Mapping[str, float], cov: pd.DataFrame,
                             group_of: Mapping[str, str]) -> pd.DataFrame:
    rc = risk_contributions(weights, cov)
    rc["group"] = [group_of.get(c, "อื่น ๆ") for c in rc.index]
    out = rc.groupby("group")[["weight", "risk_contribution", "risk_share"]].sum()
    return out.sort_values("risk_share", ascending=False)


def diversification_ratio(weights: Mapping[str, float], cov: pd.DataFrame) -> float:
    """Weighted average stand-alone vol ÷ portfolio vol. 1.0 = no diversification."""
    codes = [c for c in cov.columns if c in weights]
    w = np.array([weights[c] for c in codes], dtype=float)
    if w.sum() <= 0:
        return 1.0
    w = w / w.sum()
    sigma = cov.loc[codes, codes].to_numpy()
    stand_alone = float(w @ np.sqrt(np.diag(sigma)))
    port = portfolio_volatility(w, sigma)
    return stand_alone / port if port > 0 else 1.0


def effective_bets(weights: Mapping[str, float]) -> float:
    """Inverse Herfindahl of the weights — the effective number of positions."""
    w = np.array([v for v in weights.values() if v > 0], dtype=float)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    return float(1.0 / (w ** 2).sum())


def concentration_index(weights: Mapping[str, float]) -> float:
    """Herfindahl-Hirschman index of the weights, 0 (diffuse) to 1 (one fund)."""
    w = np.array([v for v in weights.values() if v > 0], dtype=float)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    return float((w ** 2).sum())


# --------------------------------------------------------------------------- #
# Factor exposure
# --------------------------------------------------------------------------- #
def factor_betas(target: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """OLS betas of a return series on factor return series (with intercept)."""
    joined = pd.concat([target.rename("y"), factors], axis=1).dropna()
    if len(joined) < 20:
        return pd.Series(0.0, index=factors.columns)
    y = joined["y"].to_numpy()
    x = joined[factors.columns].to_numpy()
    x = np.column_stack([np.ones(len(x)), x])
    try:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(0.0, index=factors.columns)
    return pd.Series(coef[1:], index=factors.columns)


def factor_r2(target: pd.Series, factors: pd.DataFrame) -> float:
    joined = pd.concat([target.rename("y"), factors], axis=1).dropna()
    if len(joined) < 20:
        return 0.0
    y = joined["y"].to_numpy()
    x = np.column_stack([np.ones(len(joined)), joined[factors.columns].to_numpy()])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - resid.var() * len(y) / ss_tot) if ss_tot > 0 else 0.0


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.dropna(how="any").corr()


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 126) -> pd.Series:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < window:
        return pd.Series(dtype=float)
    return joined.iloc[:, 0].rolling(window).corr(joined.iloc[:, 1])
