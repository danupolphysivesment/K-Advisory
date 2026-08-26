"""One-call analytics bundle for a portfolio.

The three tabs of the app all need the same underlying numbers for whatever
portfolio is in front of them — the current book, a proposed portfolio, or a
policy benchmark. Assembling them once here keeps the UI layer to presentation
and guarantees that Part 1 and Part 2 are computed identically, which is the
only way their comparison means anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import attribution, metrics, risk
from .data import FundData, portfolio_returns
from .universe import SEC_BUCKETS


@dataclass
class Analysis:
    label: str
    weights: Dict[str, float]
    panel: pd.DataFrame               # aligned fund returns over the window
    returns: pd.Series                # portfolio return series
    stats: metrics.Stats
    drawdown: pd.Series
    cov: pd.DataFrame
    risk_frame: pd.DataFrame          # per-fund weight / MCTR / risk share
    contributions: pd.DataFrame       # per-fund return contribution
    class_weights: Dict[str, float]
    region_weights: Dict[str, float]
    bucket_exposure: Dict[str, float]
    role_weights: Dict[str, float]
    effective_bets: float
    diversification_ratio: float
    concentration: float
    window: Tuple[pd.Timestamp, pd.Timestamp]
    coverage_note: str = ""

    @property
    def top_risk(self) -> Optional[Tuple[str, float]]:
        if self.risk_frame.empty:
            return None
        row = self.risk_frame["risk_share"].idxmax()
        return row, float(self.risk_frame.loc[row, "risk_share"])

    def class_risk(self, universe: Mapping) -> pd.DataFrame:
        return risk.group_risk_contributions(
            self.weights, self.cov,
            {c: universe[c].asset_class for c in self.weights if c in universe})

    def fund_labels(self, universe: Mapping) -> Dict[str, str]:
        return {c: (universe[c].name if c in universe else c) for c in self.weights}


def _group(weights: Mapping[str, float], universe: Mapping,
           attr: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for code, w in weights.items():
        fund = universe.get(code)
        if fund is None or w <= 0:
            continue
        key = getattr(fund, attr)
        out[key] = out.get(key, 0.0) + w
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def analyse(
    fund_data: FundData,
    weights: Mapping[str, float],
    universe: Mapping,
    start=None,
    end=None,
    rebalance: str = "Q",
    rf: float = metrics.DEFAULT_RF,
    cov_method: str = risk.LEDOIT,
    label: str = "Portfolio",
) -> Analysis:
    """Everything the UI needs about one set of weights over one window."""
    live = {c: float(w) for c, w in weights.items() if w > 1e-9 and c in fund_data.returns.columns}
    if not live:
        raise ValueError("no holdings with return history")
    total = sum(live.values())
    live = {c: w / total for c, w in live.items()}

    codes = list(live)
    natural_start = fund_data.coverage_start(codes)
    effective_start = max(pd.Timestamp(start), natural_start) if start is not None \
        else natural_start

    panel = fund_data.slice(effective_start, end, codes).dropna(how="any")
    if panel.empty:
        raise ValueError("no overlapping history for these holdings")

    note = ""
    if start is not None and natural_start > pd.Timestamp(start):
        newest = max(codes, key=lambda c: fund_data.first_valid[c])
        fund = universe.get(newest)
        from .th import month as _th_month

        note = (f"ข้อมูลย้อนหลังเริ่มที่ {_th_month(natural_start)} ซึ่งเป็นวันจัดตั้ง"
                f"ของ {fund.name if fund else newest} กองทุนที่ใหม่ที่สุดในพอร์ต "
                f"กองอื่นมีข้อมูลย้อนหลังมากกว่านี้ แต่ไม่สามารถนำมารวมเป็นพอร์ต"
                f"เดียวกันได้หากไม่มีข้อมูลของกองนี้")

    port = portfolio_returns(panel, live, rebalance=rebalance)
    cov = risk.covariance(panel, method=cov_method)

    return Analysis(
        label=label,
        weights=live,
        panel=panel,
        returns=port,
        stats=metrics.compute(port, rf),
        drawdown=metrics.drawdown_series(port),
        cov=cov,
        risk_frame=risk.risk_contributions(live, cov),
        contributions=attribution.contribution(panel, live, rebalance),
        class_weights=_group(live, universe, "asset_class"),
        region_weights=_group(live, universe, "region"),
        bucket_exposure=_bucket_exposure(live, universe),
        role_weights=_group(live, universe, "role"),
        effective_bets=risk.effective_bets(live),
        diversification_ratio=risk.diversification_ratio(live, cov),
        concentration=risk.concentration_index(live),
        window=(panel.index[0], panel.index[-1]),
        coverage_note=note,
    )


def _bucket_exposure(weights: Mapping[str, float],
                     universe: Mapping) -> Dict[str, float]:
    out = {b: 0.0 for b in SEC_BUCKETS}
    for code, w in weights.items():
        fund = universe.get(code)
        if fund is None:
            continue
        for bucket, share in fund.lookthrough.items():
            out[bucket] += w * share
    return out


def align_windows(a: Analysis, b: Analysis) -> Tuple[pd.Series, pd.Series]:
    """Trim two portfolios to their common dates so a comparison is fair."""
    joined = pd.concat([a.returns.rename("a"), b.returns.rename("b")],
                       axis=1).dropna()
    return joined["a"], joined["b"]


def comparison_table(
    current: Analysis,
    proposed: Analysis,
    rf: float = metrics.DEFAULT_RF,
) -> pd.DataFrame:
    """Side-by-side statistics on the window both portfolios actually share."""
    a, b = align_windows(current, proposed)
    if a.empty:
        return pd.DataFrame()
    sa, sb = metrics.compute(a, rf), metrics.compute(b, rf)

    rows = [
        ("ผลตอบแทนต่อปี (CAGR)", sa.cagr, sb.cagr, "pct", "higher"),
        ("Volatility", sa.volatility, sb.volatility, "pct", "lower"),
        ("Sharpe Ratio", sa.sharpe, sb.sharpe, "num", "higher"),
        ("Sortino Ratio", sa.sortino, sb.sortino, "num", "higher"),
        ("Max Drawdown", sa.max_drawdown, sb.max_drawdown, "pct", "higher"),
        ("Calmar Ratio", sa.calmar, sb.calmar, "num", "higher"),
        ("วันที่แย่ที่สุด", sa.worst_period, sb.worst_period, "pct", "higher"),
        ("VaR 95% (1 วัน)", sa.var_95, sb.var_95, "pct", "higher"),
        ("CVaR 95% (1 วัน)", sa.cvar_95, sb.cvar_95, "pct", "higher"),
        ("Hit Rate", sa.hit_rate, sb.hit_rate, "pct", "higher"),
        ("Effective Positions", current.effective_bets, proposed.effective_bets,
         "num", "higher"),
        ("Diversification Ratio", current.diversification_ratio,
         proposed.diversification_ratio, "num", "higher"),
    ]
    frame = pd.DataFrame(
        [{"ตัวชี้วัด": m, "ปัจจุบัน": c, "เสนอใหม่": p, "เปลี่ยนแปลง": p - c,
          "fmt": f, "better": bt} for m, c, p, f, bt in rows]
    ).set_index("ตัวชี้วัด")
    frame.attrs["window"] = (a.index[0], a.index[-1])
    return frame


def improvement_flags(table: pd.DataFrame) -> Dict[str, bool]:
    """True where the proposed portfolio is better on that metric."""
    out: Dict[str, bool] = {}
    for metric, row in table.iterrows():
        delta = row["เปลี่ยนแปลง"]
        out[metric] = delta > 0 if row["better"] == "higher" else delta < 0
    return out
