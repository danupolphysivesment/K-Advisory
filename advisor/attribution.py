"""Return attribution: who actually produced the portfolio's return.

Two views are provided.

*Contribution* answers "how many percentage points of the portfolio's total
return came from each holding?" It is computed by carrying the drifting weight
of each holding through time under the chosen rebalancing rule, so the
contributions add up exactly to the portfolio's compounded return — no residual
fudge factor.

*Brinson allocation vs selection* compares the portfolio to a policy benchmark
built from the client's risk-profile band, splitting active return into the
part that came from over/under-weighting an asset class and the part that came
from picking different funds inside it.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .compat import period_alias, resample_alias, writable
from .data import periods_per_year


def contribution(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    rebalance: str = "Q",
) -> pd.DataFrame:
    """Per-holding contribution to the portfolio's compounded total return.

    The contributions sum exactly to the portfolio total return: each period's
    holding P&L is measured in units of the *portfolio's* value at that moment,
    so compounding is attributed to whichever holding earned it.
    """
    codes = [c for c in weights if c in returns.columns and weights[c] > 0]
    if not codes:
        return pd.DataFrame(columns=["weight", "total_return", "contribution", "share"])

    w0 = np.array([weights[c] for c in codes], dtype=float)
    w0 = w0 / w0.sum()

    panel = returns[codes].dropna(how="all")
    r = panel.fillna(0.0).to_numpy()
    idx = panel.index

    if rebalance in ("D", "none"):
        marks = np.zeros(len(idx), dtype=bool)
        constant_mix = rebalance == "D"
    else:
        period = pd.Series(idx, index=idx).dt.to_period(period_alias(rebalance))
        marks = writable(period.ne(period.shift()).to_numpy())
        marks[0] = False
        constant_mix = False

    value = 1.0
    w = w0.copy()
    contrib = np.zeros(len(codes))

    for t in range(len(idx)):
        if marks[t] or constant_mix:
            w = w0.copy()
        pnl = value * w * r[t]          # THB P&L per holding on a 1.0 base
        contrib += pnl
        grown = w * (1.0 + r[t])
        new_value = value * grown.sum()
        w = grown / grown.sum() if grown.sum() > 0 else w0.copy()
        value = new_value

    total = value - 1.0
    standalone = (1.0 + panel.fillna(0.0)).prod() - 1.0
    out = pd.DataFrame(
        {
            "weight": w0,
            "total_return": standalone.reindex(codes).to_numpy(),
            "contribution": contrib,
            "share": contrib / total if abs(total) > 1e-12 else np.zeros(len(codes)),
        },
        index=codes,
    )
    out.attrs["portfolio_total"] = total
    return out.sort_values("contribution", ascending=False)


def contribution_by_group(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    group_of: Mapping[str, str],
    rebalance: str = "Q",
) -> pd.DataFrame:
    detail = contribution(returns, weights, rebalance)
    if detail.empty:
        return detail
    detail = detail.copy()
    detail["group"] = [group_of.get(c, "Other") for c in detail.index]
    out = detail.groupby("group")[["weight", "contribution", "share"]].sum()
    out.attrs["portfolio_total"] = detail.attrs.get("portfolio_total", 0.0)
    return out.sort_values("contribution", ascending=False)


# --------------------------------------------------------------------------- #
# Brinson-Fachler allocation vs selection
# --------------------------------------------------------------------------- #
def brinson(
    returns: pd.DataFrame,
    portfolio_weights: Mapping[str, float],
    benchmark_weights: Mapping[str, float],
    group_of: Mapping[str, str],
) -> pd.DataFrame:
    """Single-period Brinson-Fachler attribution over the whole window.

    Group returns are the weighted returns of the funds inside each group, so
    "selection" captures fund choice within an asset class and "allocation"
    captures the size of the class bet.
    """
    groups = sorted({group_of.get(c, "Other")
                     for c in set(portfolio_weights) | set(benchmark_weights)})

    def _group_stats(weights: Mapping[str, float]):
        w_out, r_out = {}, {}
        for g in groups:
            members = [c for c in weights
                       if group_of.get(c, "Other") == g and weights[c] > 0
                       and c in returns.columns]
            gw = sum(weights[c] for c in members)
            w_out[g] = gw
            if gw > 0:
                rets = [(weights[c] / gw) * float((1.0 + returns[c].dropna()).prod() - 1.0)
                        for c in members]
                r_out[g] = float(np.sum(rets))
            else:
                r_out[g] = np.nan
        return w_out, r_out

    wp, rp = _group_stats(portfolio_weights)
    wb, rb = _group_stats(benchmark_weights)

    total_b = float(np.nansum([wb[g] * (rb[g] if not np.isnan(rb[g]) else 0.0) for g in groups]))

    rows = []
    for g in groups:
        rb_g = rb[g] if not np.isnan(rb[g]) else total_b
        rp_g = rp[g] if not np.isnan(rp[g]) else rb_g
        allocation = (wp[g] - wb[g]) * (rb_g - total_b)
        selection = wb[g] * (rp_g - rb_g)
        interaction = (wp[g] - wb[g]) * (rp_g - rb_g)
        rows.append({
            "group": g,
            "w_portfolio": wp[g],
            "w_benchmark": wb[g],
            "r_portfolio": rp_g if wp[g] > 0 else np.nan,
            "r_benchmark": rb_g if wb[g] > 0 else np.nan,
            "allocation": allocation,
            "selection": selection,
            "interaction": interaction,
            "total_effect": allocation + selection + interaction,
        })
    frame = pd.DataFrame(rows).set_index("group")
    frame.attrs["benchmark_total"] = total_b
    return frame.sort_values("total_effect", ascending=False)
