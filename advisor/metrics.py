"""Performance and risk statistics for a return series.

Everything annualises off the *observed* frequency of the series rather than a
hard-coded 252, because the workbook mixes weekly (1990s) and daily (2005+)
observations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .compat import resample_alias
from .data import periods_per_year

# Thai policy-rate proxy used as the risk-free leg of Sharpe/Sortino. Adjustable
# from the UI; 1.75% is roughly the BoT policy rate over the recent sample.
DEFAULT_RF = 0.0175


@dataclass
class Stats:
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp
    years: float
    total_return: float
    cagr: float
    volatility: float
    downside_deviation: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_dd_start: Optional[pd.Timestamp]
    max_dd_trough: Optional[pd.Timestamp]
    max_dd_recovery: Optional[pd.Timestamp]
    dd_duration_days: Optional[int]
    var_95: float
    cvar_95: float
    var_99: float
    skew: float
    kurtosis: float
    best_period: float
    worst_period: float
    hit_rate: float
    ann_factor: float

    def as_dict(self) -> Dict:
        return asdict(self)


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    return curve / curve.cummax() - 1.0


def max_drawdown_detail(returns: pd.Series):
    """Return (depth, peak_date, trough_date, recovery_date)."""
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = curve.cummax()
    dd = curve / running_max - 1.0
    if dd.empty:
        return 0.0, None, None, None
    trough = dd.idxmin()
    depth = float(dd.loc[trough])
    if depth == 0.0:
        return 0.0, None, None, None
    peak = curve.loc[:trough].idxmax()
    after = curve.loc[trough:]
    recovered = after[after >= curve.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    return depth, peak, trough, recovery


def compute(returns: pd.Series, rf: float = DEFAULT_RF) -> Stats:
    r = returns.dropna()
    if len(r) < 2:
        return Stats(len(r), returns.index[0] if len(returns) else pd.NaT,
                     returns.index[-1] if len(returns) else pd.NaT,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     None, None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 252.0)

    ann = periods_per_year(r.index)
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    total = float((1.0 + r).prod() - 1.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 else -1.0

    vol = float(r.std(ddof=1) * np.sqrt(ann))
    rf_per = (1.0 + rf) ** (1.0 / ann) - 1.0
    excess = r - rf_per
    downside = excess[excess < 0]
    dd_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(ann)) if len(downside) else 0.0

    sharpe = float(excess.mean() / r.std(ddof=1) * np.sqrt(ann)) if r.std(ddof=1) > 0 else 0.0
    sortino = float(excess.mean() * ann / dd_dev) if dd_dev > 0 else 0.0

    depth, peak, trough, recovery = max_drawdown_detail(r)
    calmar = float(cagr / abs(depth)) if depth < 0 else 0.0
    duration = int((recovery - peak).days) if (recovery is not None and peak is not None) else None

    var95 = float(np.percentile(r, 5))
    cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else var95
    var99 = float(np.percentile(r, 1))

    return Stats(
        n_obs=len(r),
        start=r.index[0],
        end=r.index[-1],
        years=years,
        total_return=total,
        cagr=cagr,
        volatility=vol,
        downside_deviation=dd_dev,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=depth,
        max_dd_start=peak,
        max_dd_trough=trough,
        max_dd_recovery=recovery,
        dd_duration_days=duration,
        var_95=var95,
        cvar_95=cvar95,
        var_99=var99,
        skew=float(r.skew()),
        kurtosis=float(r.kurtosis()),
        best_period=float(r.max()),
        worst_period=float(r.min()),
        hit_rate=float((r > 0).mean()),
        ann_factor=ann,
    )


def rolling_volatility(returns: pd.Series, window_days: int = 252) -> pd.Series:
    ann = periods_per_year(returns.index)
    window = max(int(round(window_days * ann / 252.0)), 10)
    return returns.rolling(window).std(ddof=1) * np.sqrt(ann)


def rolling_return(returns: pd.Series, window_days: int = 252) -> pd.Series:
    ann = periods_per_year(returns.index)
    window = max(int(round(window_days * ann / 252.0)), 10)
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def calendar_year_returns(returns: pd.Series) -> pd.Series:
    grouped = (1.0 + returns.dropna()).groupby(returns.dropna().index.year).prod() - 1.0
    grouped.index.name = "ปี"
    return grouped


def annual_return_table(returns: pd.Series) -> pd.DataFrame:
    """Year-by-year performance, the way a factsheet states it.

    A single column of calendar returns would only repeat the bar chart beside
    it, so each year also carries the risk it was earned with — a +18% year
    bought with a -30% drawdown is a different conversation from a quiet one.
    """
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()

    rows = []
    for year, chunk in r.groupby(r.index.year):
        chunk = chunk.dropna()
        if chunk.empty:
            continue
        ann = periods_per_year(chunk.index) if len(chunk) > 2 else 252
        curve = (1.0 + chunk).cumprod()
        monthly = (1.0 + chunk).resample(resample_alias("M")).prod() - 1.0
        rows.append({
            "ปี": int(year),
            "ผลตอบแทน": float(curve.iloc[-1] - 1.0),
            "ความผันผวน": (float(chunk.std(ddof=1) * np.sqrt(ann))
                           if len(chunk) > 2 else np.nan),
            "Max Drawdown": float((curve / curve.cummax() - 1.0).min()),
            "เดือนที่ดีที่สุด": float(monthly.max()) if len(monthly) else np.nan,
            "เดือนที่แย่ที่สุด": float(monthly.min()) if len(monthly) else np.nan,
            "วันทำการ": int(len(chunk)),
        })

    table = pd.DataFrame(rows).set_index("ปี")
    return table.sort_index(ascending=False)


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Month-by-year grid of returns, with Thai month abbreviations."""
    from .th import MONTH_ABBR

    m = (1.0 + returns.dropna()).resample(resample_alias("M")).prod() - 1.0
    frame = pd.DataFrame({"year": m.index.year,
                          "month": [MONTH_ABBR[d.strftime("%b")] for d in m.index],
                          "ret": m.values})
    table = frame.pivot(index="year", columns="month", values="ret")
    order = [MONTH_ABBR[k] for k in ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]]
    table = table.reindex(columns=[c for c in order if c in table.columns])
    table["รวมทั้งปี"] = (1.0 + m).groupby(m.index.year).prod() - 1.0
    table.index.name = "ปี"
    return table.sort_index(ascending=False)


def tracking_stats(portfolio: pd.Series, benchmark: pd.Series,
                   rf: float = DEFAULT_RF) -> Dict[str, float]:
    """Active statistics of ``portfolio`` versus ``benchmark``."""
    joined = pd.concat([portfolio, benchmark], axis=1).dropna()
    if len(joined) < 3:
        return {"alpha": 0.0, "beta": 0.0, "tracking_error": 0.0,
                "information_ratio": 0.0, "up_capture": 0.0, "down_capture": 0.0,
                "correlation": 0.0}
    p, b = joined.iloc[:, 0], joined.iloc[:, 1]
    ann = periods_per_year(joined.index)
    var_b = b.var(ddof=1)
    beta = float(p.cov(b) / var_b) if var_b > 0 else 0.0
    rf_per = (1.0 + rf) ** (1.0 / ann) - 1.0
    alpha = float(((p - rf_per).mean() - beta * (b - rf_per).mean()) * ann)
    active = p - b
    te = float(active.std(ddof=1) * np.sqrt(ann))
    ir = float(active.mean() * ann / te) if te > 0 else 0.0

    up, down = b > 0, b < 0
    up_cap = float(p[up].mean() / b[up].mean()) if up.any() and b[up].mean() != 0 else 0.0
    dn_cap = float(p[down].mean() / b[down].mean()) if down.any() and b[down].mean() != 0 else 0.0
    return {"alpha": alpha, "beta": beta, "tracking_error": te, "information_ratio": ir,
            "up_capture": up_cap, "down_capture": dn_cap, "correlation": float(p.corr(b))}


def summary_frame(series_map: Dict[str, pd.Series], rf: float = DEFAULT_RF) -> pd.DataFrame:
    """Side-by-side stats table for several return series."""
    rows = {}
    for label, series in series_map.items():
        s = compute(series, rf)
        rows[label] = {
            "CAGR": s.cagr,
            "Volatility": s.volatility,
            "Sharpe": s.sharpe,
            "Sortino": s.sortino,
            "Max Drawdown": s.max_drawdown,
            "Calmar": s.calmar,
            "VaR 95% (1 วัน)": s.var_95,
            "CVaR 95% (1 วัน)": s.cvar_95,
            "Skew": s.skew,
            "Excess Kurtosis": s.kurtosis,
            "Hit Rate": s.hit_rate,
        }
    return pd.DataFrame(rows)
