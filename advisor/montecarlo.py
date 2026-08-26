"""Forward-looking Monte Carlo simulation of a portfolio.

Simulation happens at the *fund* level with a full covariance matrix, then the
paths are aggregated through the portfolio weights with periodic rebalancing.
That matters: simulating the portfolio's own return series directly would bake
in today's diversification and understate the risk of a concentrated book.

Five generators are available, each answering a different objection:

``Geometric Brownian Motion``
    The textbook baseline — multivariate normal log-returns. Fast, smooth, and
    known to understate tail risk.
``Student-t (fat tails)``
    Same covariance, but shocks drawn from a multivariate t. Degrees of freedom
    are estimated from the sample kurtosis unless overridden.
``Historical bootstrap``
    Resamples actual observed periods i.i.d. Makes no distributional
    assumption and preserves cross-sectional correlation exactly.
``Block bootstrap``
    Resamples contiguous blocks, preserving volatility clustering and
    momentum/mean-reversion that i.i.d. sampling destroys.
``Regime-switching``
    A two-state Gaussian model (calm / stressed) fitted by volatility
    thresholding, with an empirical transition matrix. Produces realistic
    clustered drawdowns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .compat import resample_alias
from .data import periods_per_year

GBM = "Geometric Brownian Motion"
STUDENT_T = "Student-t (หางหนา)"
BOOTSTRAP = "Historical Bootstrap"
BLOCK = "Block Bootstrap"
REGIME = "Regime-Switching (2 สถานะ)"

# Every generator implemented, still exercised by the test suite.
ALL_METHODS: List[str] = [GBM, STUDENT_T, BOOTSTRAP, BLOCK, REGIME]

# The app runs block bootstrap only. It is the one generator that keeps
# volatility clustering and trend intact, which is what makes its drawdown
# numbers usable in a client conversation; offering the alternatives mostly
# invites picking whichever produces the nicest fan chart.
METHODS: List[str] = [BLOCK]
DEFAULT_METHOD: str = BLOCK

METHOD_NOTES: Dict[str, str] = {
    GBM: "ผลตอบแทนแบบ log สุ่มจากการแจกแจงปกติหลายตัวแปร ใช้ค่าเฉลี่ยและ "
         "ความแปรปรวนร่วมจากข้อมูลจริง เป็นวิธีมาตรฐานของอุตสาหกรรม "
         "แต่จะประเมินโอกาสขาดทุนรุนแรงต่ำกว่าความจริง",
    STUDENT_T: "โมเมนต์สองอันดับแรกเท่ากับ GBM แต่หางของการแจกแจงหนากว่า "
               "ทำให้เดือนที่ผันผวนรุนแรงปรากฏในความถี่ที่สมจริง "
               "ค่า degrees of freedom ยิ่งต่ำ หางยิ่งหนา",
    BOOTSTRAP: "สุ่มช่วงเวลาจริงในอดีตแบบใส่คืน ไม่ตั้งสมมติฐานเรื่องรูปแบบ "
               "การแจกแจง และรักษาสหสัมพันธ์ระหว่างกองทุนไว้ครบ "
               "เพราะสุ่มทั้งแถวไปด้วยกัน",
    BLOCK: "สุ่มช่วงเวลาที่ต่อเนื่องกัน ทำให้การกระจุกตัวของความผันผวนและแนวโน้ม "
           "ยังคงอยู่ในผลจำลอง เป็นตัวเลือกที่ตรงไปตรงมาที่สุด "
           "เมื่อต้องการตอบคำถามเรื่อง Max Drawdown",
    REGIME: "สองสถานะ คือช่วงตลาดสงบและช่วงตลาดกดดัน พร้อมเมทริกซ์ความน่าจะเป็น "
            "ในการเปลี่ยนสถานะที่ประมาณจากข้อมูลจริง จำลองช่วงขาดทุนที่ยืดเยื้อ "
            "และเกาะกลุ่มกันซึ่งแบบจำลองสถานะเดียวมองไม่เห็น",
}

PERIOD_LABELS: Dict[str, Tuple[str, float]] = {
    "เดือน": ("M", 12.0),
    "ไตรมาส": ("Q", 4.0),
    "ปี": ("A", 1.0),
}


@dataclass
class SimulationResult:
    method: str
    paths: np.ndarray                 # (n_paths, n_periods + 1) growth of 1.0
    period_label: str
    periods_per_year: float
    n_periods: int
    percentiles: pd.DataFrame         # index = period, columns = "p5" ... "p95"
    terminal: np.ndarray              # growth multiples at the horizon
    max_drawdowns: np.ndarray         # per-path worst drawdown
    diagnostics: Dict[str, object]

    @property
    def horizon_years(self) -> float:
        return self.n_periods / self.periods_per_year

    def terminal_stats(self, initial_value: float = 1.0) -> Dict[str, float]:
        term = self.terminal * initial_value
        years = max(self.horizon_years, 1e-9)
        cagr = self.terminal ** (1.0 / years) - 1.0
        return {
            "median": float(np.median(term)),
            "mean": float(term.mean()),
            "p5": float(np.percentile(term, 5)),
            "p25": float(np.percentile(term, 25)),
            "p75": float(np.percentile(term, 75)),
            "p95": float(np.percentile(term, 95)),
            "prob_loss": float((self.terminal < 1.0).mean()),
            "median_cagr": float(np.median(cagr)),
            "cagr_p5": float(np.percentile(cagr, 5)),
            "cagr_p95": float(np.percentile(cagr, 95)),
            "median_max_dd": float(np.median(self.max_drawdowns)),
            "worst_max_dd": float(np.percentile(self.max_drawdowns, 1)),
            "var95_terminal": float(np.percentile(term, 5)),
            "cvar95_terminal": float(term[term <= np.percentile(term, 5)].mean()),
        }

    def probability_above(self, multiple: float) -> float:
        return float((self.terminal >= multiple).mean())

    def shortfall_probability(self, annual_target: float) -> float:
        """P(realised CAGR < target)."""
        years = max(self.horizon_years, 1e-9)
        cagr = self.terminal ** (1.0 / years) - 1.0
        return float((cagr < annual_target).mean())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resample_panel(returns: pd.DataFrame, freq: str) -> pd.DataFrame:
    return (1.0 + returns).resample(resample_alias(freq)).prod() - 1.0


def _estimate_df(x: np.ndarray, floor: float = 3.0, cap: float = 30.0) -> float:
    """Degrees of freedom implied by excess kurtosis: κ = 6/(ν−4)."""
    flat = x.reshape(-1)
    if len(flat) < 30:
        return 8.0
    kurt = float(pd.Series(flat).kurtosis())
    if kurt <= 0.1:
        return cap
    return float(np.clip(6.0 / kurt + 4.0, floor, cap))


def _chol(cov: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-14)
        return vecs @ np.diag(np.sqrt(vals))


def _drawdowns(paths: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(paths, axis=1)
    return (paths / running_max - 1.0).min(axis=1)


# --------------------------------------------------------------------------- #
# Generators — each returns (n_paths, n_periods, n_assets) simple returns
# --------------------------------------------------------------------------- #
def _draw_gbm(hist: np.ndarray, n_paths: int, n_periods: int,
              rng: np.random.Generator, drift_scale: float) -> np.ndarray:
    log_r = np.log1p(np.clip(hist, -0.99, None))
    mu = log_r.mean(axis=0) * drift_scale
    cov = np.cov(log_r, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    L = _chol(cov)
    z = rng.standard_normal((n_paths, n_periods, hist.shape[1]))
    shocks = z @ L.T + mu
    return np.expm1(shocks)


def _draw_student_t(hist: np.ndarray, n_paths: int, n_periods: int,
                    rng: np.random.Generator, drift_scale: float,
                    df: Optional[float]) -> np.ndarray:
    log_r = np.log1p(np.clip(hist, -0.99, None))
    mu = log_r.mean(axis=0) * drift_scale
    cov = np.atleast_2d(np.cov(log_r, rowvar=False, ddof=1))
    nu = df if df is not None else _estimate_df(log_r)
    # Scale so the simulated covariance matches the sample covariance.
    scale = (nu - 2.0) / nu if nu > 2 else 1.0
    L = _chol(cov * scale)
    z = rng.standard_normal((n_paths, n_periods, hist.shape[1]))
    g = rng.chisquare(nu, size=(n_paths, n_periods, 1)) / nu
    shocks = (z / np.sqrt(g)) @ L.T + mu
    return np.expm1(shocks)


def _draw_bootstrap(hist: np.ndarray, n_paths: int, n_periods: int,
                    rng: np.random.Generator, drift_scale: float) -> np.ndarray:
    t = hist.shape[0]
    idx = rng.integers(0, t, size=(n_paths, n_periods))
    out = hist[idx]
    if drift_scale != 1.0:
        mean = hist.mean(axis=0)
        out = out + (drift_scale - 1.0) * mean
    return out


def _draw_block(hist: np.ndarray, n_paths: int, n_periods: int,
                rng: np.random.Generator, drift_scale: float,
                block: int = 6) -> np.ndarray:
    t, n = hist.shape
    block = int(max(2, min(block, max(t // 4, 2))))
    n_blocks = int(np.ceil(n_periods / block))
    starts = rng.integers(0, t, size=(n_paths, n_blocks))
    offsets = np.arange(block)
    # Circular blocks so every observation can start a block.
    idx = (starts[:, :, None] + offsets[None, None, :]) % t
    idx = idx.reshape(n_paths, -1)[:, :n_periods]
    out = hist[idx]
    if drift_scale != 1.0:
        out = out + (drift_scale - 1.0) * hist.mean(axis=0)
    return out


def _fit_regimes(hist: np.ndarray, quantile: float = 0.75):
    """Split history into calm/stressed by cross-sectional shock magnitude."""
    mag = np.abs(hist).mean(axis=1)
    threshold = np.quantile(mag, quantile)
    stressed = mag > threshold
    if stressed.sum() < 5 or (~stressed).sum() < 5:
        stressed = mag > np.median(mag)

    states = stressed.astype(int)
    trans = np.zeros((2, 2))
    for a, b in zip(states[:-1], states[1:]):
        trans[a, b] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    trans = np.divide(trans, row_sums, out=np.full_like(trans, 0.5), where=row_sums > 0)

    params = []
    for s in (0, 1):
        block = hist[states == s]
        mu = block.mean(axis=0)
        cov = np.atleast_2d(np.cov(block, rowvar=False, ddof=1)) if len(block) > 2 \
            else np.atleast_2d(np.cov(hist, rowvar=False, ddof=1))
        params.append((mu, _chol(cov)))
    return params, trans, float(states.mean())


def _draw_regime(hist: np.ndarray, n_paths: int, n_periods: int,
                 rng: np.random.Generator, drift_scale: float) -> np.ndarray:
    params, trans, stressed_share = _fit_regimes(hist)
    n = hist.shape[1]
    out = np.empty((n_paths, n_periods, n))
    state = (rng.random(n_paths) < stressed_share).astype(int)
    for t in range(n_periods):
        z = rng.standard_normal((n_paths, n))
        for s in (0, 1):
            mask = state == s
            if not mask.any():
                continue
            mu, L = params[s]
            out[mask, t, :] = z[mask] @ L.T + mu * drift_scale
        switch = rng.random(n_paths)
        p_stay = trans[state, state]
        flipped = switch > p_stay
        state = np.where(flipped, 1 - state, state)
    return out


_GENERATORS = {
    GBM: _draw_gbm,
    BOOTSTRAP: _draw_bootstrap,
    BLOCK: _draw_block,
    REGIME: _draw_regime,
}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def simulate(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    method: str = BLOCK,
    n_paths: int = 10_000,
    n_periods: int = 60,
    period: str = "เดือน",
    rebalance: bool = True,
    drift_scale: float = 1.0,
    student_df: Optional[float] = None,
    block_size: int = 6,
    mu_override: Optional[Mapping[str, float]] = None,
    seed: int = 20260818,
) -> SimulationResult:
    """Simulate forward paths for a weighted portfolio.

    ``drift_scale`` haircuts the historical mean (0.0 = zero-drift, a common
    conservatism when past returns are not considered repeatable).
    ``mu_override`` replaces the sample mean entirely with forward-looking
    annualised expected returns per fund — the right answer when five years of
    trailing history is not a defensible forecast. Volatility, correlation and
    tail shape still come from history.
    ``rebalance`` toggles constant-mix versus buy-and-hold weight drift.
    """
    codes = [c for c in weights if c in returns.columns and weights[c] > 0]
    if not codes:
        raise ValueError("no holdings with return history")

    w = np.array([weights[c] for c in codes], dtype=float)
    w = w / w.sum()

    freq, per_year = PERIOD_LABELS.get(period, ("M", 12.0))
    panel = _resample_panel(returns[codes], freq).dropna(how="any")
    if len(panel) < 12:
        # Fall back to the raw frequency if the resample leaves too little.
        panel = returns[codes].dropna(how="any")
        per_year = periods_per_year(panel.index)
    if len(panel) < 6:
        raise ValueError("not enough overlapping history to simulate")

    hist = panel.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    if method == STUDENT_T:
        draws = _draw_student_t(hist, n_paths, n_periods, rng, drift_scale, student_df)
    elif method == BLOCK:
        draws = _draw_block(hist, n_paths, n_periods, rng, drift_scale, block_size)
    else:
        gen = _GENERATORS.get(method, _draw_block)
        draws = gen(hist, n_paths, n_periods, rng, drift_scale)

    # Recentre onto forward-looking expected returns if the RM supplied them.
    # Done after generation so it applies uniformly to all five generators and
    # leaves volatility, correlation and tail shape untouched.
    mu_source = "ค่าเฉลี่ยผลตอบแทนในอดีต"
    if mu_override:
        target = np.array(
            [(1.0 + float(mu_override.get(c, 0.0))) ** (1.0 / per_year) - 1.0 for c in codes]
        )
        draws = draws + (target - draws.mean(axis=(0, 1)))
        mu_source = "Capital Market Assumptions แบบมองไปข้างหน้า"
    elif drift_scale != 1.0:
        mu_source = f"ค่าเฉลี่ยผลตอบแทนในอดีต × {drift_scale:g}"

    draws = np.clip(draws, -0.95, 5.0)

    if rebalance:
        port = draws @ w                                    # constant mix
    else:
        # Buy and hold: let the weights drift with realised performance.
        growth = np.cumprod(1.0 + draws, axis=1)
        value = growth @ w
        port = np.empty((n_paths, n_periods))
        port[:, 0] = value[:, 0] - 1.0
        port[:, 1:] = value[:, 1:] / value[:, :-1] - 1.0

    paths = np.empty((n_paths, n_periods + 1))
    paths[:, 0] = 1.0
    paths[:, 1:] = np.cumprod(1.0 + port, axis=1)

    levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct = np.percentile(paths, levels, axis=0)
    percentiles = pd.DataFrame(
        pct.T, columns=[f"p{p}" for p in levels], index=np.arange(n_periods + 1)
    )
    percentiles.index.name = period

    hist_port = hist @ w
    diagnostics = {
        "hist_mean_ann": float((1.0 + hist_port.mean()) ** per_year - 1.0),
        "hist_vol_ann": float(hist_port.std(ddof=1) * np.sqrt(per_year)),
        "sim_mean_ann": float((1.0 + port.mean()) ** per_year - 1.0),
        "sim_vol_ann": float(port.std(ddof=1) * np.sqrt(per_year)),
        "sim_skew": float(pd.Series(port.reshape(-1)).skew()),
        "sim_kurtosis": float(pd.Series(port.reshape(-1)).kurtosis()),
        "history_periods": float(len(panel)),
        "mu_source": mu_source,
        "student_df": float(student_df if student_df is not None
                            else (_estimate_df(np.log1p(np.clip(hist, -0.99, None)))
                                  if method == STUDENT_T else np.nan)),
    }

    return SimulationResult(
        method=method,
        paths=paths,
        period_label=period,
        periods_per_year=per_year,
        n_periods=n_periods,
        percentiles=percentiles,
        terminal=paths[:, -1],
        max_drawdowns=_drawdowns(paths),
        diagnostics=diagnostics,
    )


def compare_methods(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    methods: Sequence[str] = tuple(METHODS),
    **kwargs,
) -> pd.DataFrame:
    """Terminal-wealth statistics under every generator, side by side."""
    rows = {}
    for m in methods:
        try:
            res = simulate(returns, weights, method=m, **kwargs)
        except Exception:
            continue
        s = res.terminal_stats()
        rows[m] = {
            "มูลค่าปลายทาง (Median)": s["median"],
            "Percentile 5": s["p5"],
            "Percentile 95": s["p95"],
            "CAGR (Median)": s["median_cagr"],
            "โอกาสขาดทุน": s["prob_loss"],
            "Max Drawdown (ทั่วไป)": s["median_max_dd"],
            "Max Drawdown (1 ใน 100)": s["worst_max_dd"],
        }
    return pd.DataFrame(rows).T
