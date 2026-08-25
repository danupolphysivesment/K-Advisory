"""Forward-looking capital market assumptions.

Five years of trailing returns is not a forecast. Over the sample in this
workbook Chinese equity compounded at roughly −13% a year and Thai bonds
returned less than the policy rate; feeding those numbers into a mean-variance
optimiser produces a portfolio that shorts the past rather than one that is
positioned for the future.

The expected returns here are built the way an asset manager builds them — from
a cash rate plus a term premium plus an equity risk premium — and are stated
per asset class in THB terms. Volatility and correlation still come entirely
from the workbook, because those *are* reasonably persistent.

Every number is editable from the UI. They are assumptions, and the app says so.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd

from .universe import (
    AC_ALLOC, AC_ALT, AC_ASIA_EQ, AC_GL_EQ, AC_GL_FI, AC_MONEY, AC_SECTOR,
    AC_TH_EQ, AC_TH_FI, ALT, CASH, EQUITY, FIXED,
)

# Building blocks, annualised, in THB.
THB_CASH_RATE = 0.0175          # BoT policy rate proxy
THB_TERM_PREMIUM = 0.0100       # 10y Thai govt over cash
THB_CREDIT_SPREAD = 0.0075
EQUITY_RISK_PREMIUM = 0.0500    # over Thai cash, for developed equity
EM_EXTRA_PREMIUM = 0.0100

# Long-run expected return by reporting asset class.
DEFAULT_CMA: Dict[str, float] = {
    AC_MONEY: THB_CASH_RATE,
    AC_TH_FI: THB_CASH_RATE + THB_TERM_PREMIUM * 0.8 + THB_CREDIT_SPREAD * 0.5,
    AC_GL_FI: THB_CASH_RATE + THB_TERM_PREMIUM * 1.2 + THB_CREDIT_SPREAD * 1.0,
    AC_TH_EQ: THB_CASH_RATE + EQUITY_RISK_PREMIUM + 0.0050,
    AC_GL_EQ: THB_CASH_RATE + EQUITY_RISK_PREMIUM,
    AC_ASIA_EQ: THB_CASH_RATE + EQUITY_RISK_PREMIUM + EM_EXTRA_PREMIUM,
    AC_SECTOR: THB_CASH_RATE + EQUITY_RISK_PREMIUM + 0.0100,
    AC_ALLOC: np.nan,           # derived from the fund's own look-through
    AC_ALT: THB_CASH_RATE + 0.0300,
}

# Expected return of each SEC bucket, used to derive allocation funds.
BUCKET_CMA: Dict[str, float] = {
    CASH: THB_CASH_RATE,
    FIXED: THB_CASH_RATE + THB_TERM_PREMIUM,
    EQUITY: THB_CASH_RATE + EQUITY_RISK_PREMIUM,
    ALT: THB_CASH_RATE + 0.0300,
}

CMA_NOTES: Dict[str, str] = {
    AC_MONEY: "อัตราดอกเบี้ยนโยบาย",
    AC_TH_FI: "ดอกเบี้ยเงินสด + 80% ของ term premium ไทย + ครึ่งหนึ่งของ credit spread",
    AC_GL_FI: "ดอกเบี้ยเงินสด + duration premium ต่างประเทศ + credit spread เต็มจำนวน",
    AC_TH_EQ: "ดอกเบี้ยเงินสด + equity risk premium + ส่วนเพิ่มจากมูลค่าหุ้นไทย",
    AC_GL_EQ: "ดอกเบี้ยเงินสด + equity risk premium",
    AC_ASIA_EQ: "ดอกเบี้ยเงินสด + equity risk premium + ส่วนเพิ่มตลาดเกิดใหม่",
    AC_SECTOR: "ดอกเบี้ยเงินสด + equity risk premium + ส่วนเพิ่มจากการกระจุกตัว/การเติบโต",
    AC_ALLOC: "คำนวณจากสัดส่วน look-through ของกองทุนเองใน 4 กลุ่มสินทรัพย์",
    AC_ALT: "ดอกเบี้ยเงินสด + ส่วนเพิ่มสินทรัพย์จริง ทองคำไม่มีกระแสเงินสด "
            "จึงตั้งไว้ต่ำกว่าหุ้นโดยเจตนา",
}

HISTORICAL = "Historical (ย้อนหลัง)"
CMA = "Kasikorn Asset CMA"
BLEND = "ผสม CMA + Historical"
MU_SOURCES = [CMA, BLEND, HISTORICAL]

# The app blends the two rather than making the RM pick one. Neither is right
# alone: trailing returns extrapolate whatever just happened, and a pure house
# view ignores what these particular funds have actually delivered. The default
# leans forward-looking because five trailing years is a short sample.
DEFAULT_CMA_WEIGHT = 0.70


def blend_label(weight_cma: float) -> str:
    """How the mix reads on screen, e.g. "Kasikorn Asset CMA 70% + Historical 30%"."""
    w = max(0.0, min(float(weight_cma), 1.0))
    if w >= 0.999:
        return f"{CMA} 100%"
    if w <= 0.001:
        return "Historical 100%"
    return f"{CMA} {w:.0%} + Historical {1 - w:.0%}"


def blend_short(weight_cma: float) -> str:
    """The same mix in a form that fits a metric card, e.g. "CMA 70% + Hist 30%"."""
    w = max(0.0, min(float(weight_cma), 1.0))
    if w >= 0.999:
        return "CMA 100%"
    if w <= 0.001:
        return "Historical 100%"
    return f"CMA {w:.0%} + Hist {1 - w:.0%}"


def cma_for_fund(fund, overrides: Mapping[str, float] | None = None) -> float:
    """Expected return for one fund, from its class or its look-through."""
    table = dict(DEFAULT_CMA)
    if overrides:
        table.update(overrides)
    value = table.get(fund.asset_class, np.nan)
    if not np.isnan(value):
        return float(value)
    # Allocation funds: weight the bucket assumptions by the look-through.
    return float(sum(BUCKET_CMA[b] * w for b, w in fund.lookthrough.items()))


def cma_series(codes, universe: Mapping,
               overrides: Mapping[str, float] | None = None) -> pd.Series:
    return pd.Series(
        {c: cma_for_fund(universe[c], overrides) for c in codes if c in universe},
        dtype=float,
    )


def blended_mu(
    codes,
    universe: Mapping,
    historical: pd.Series,
    source: str = CMA,
    overrides: Mapping[str, float] | None = None,
    weight_cma: float = 0.5,
) -> pd.Series:
    """Expected returns under the chosen source, aligned to ``codes``."""
    forward = cma_series(codes, universe, overrides)
    hist = historical.reindex(forward.index).fillna(forward)
    if source == HISTORICAL:
        return hist
    if source == CMA:
        return forward
    w = max(0.0, min(float(weight_cma), 1.0))
    return w * forward + (1.0 - w) * hist


def mixed_mu(codes, universe: Mapping, historical: pd.Series,
             weight_cma: float = DEFAULT_CMA_WEIGHT,
             overrides: Mapping[str, float] | None = None) -> pd.Series:
    """Expected returns as a weighted mix of the house view and history.

    1.0 is the pure Kasikorn Asset CMA, 0.0 is pure trailing history. Volatility
    and correlation are never touched by this — they always come from the
    workbook, because those persist and average returns do not.
    """
    return blended_mu(codes, universe, historical, BLEND, overrides, weight_cma)


def cma_table(overrides: Mapping[str, float] | None = None) -> pd.DataFrame:
    table = dict(DEFAULT_CMA)
    if overrides:
        table.update(overrides)
    rows = []
    for cls, value in table.items():
        rows.append({
            "Asset Class": cls,
            "ผลตอบแทนคาดหวัง": value,
            "ที่มา": CMA_NOTES.get(cls, ""),
        })
    return pd.DataFrame(rows).set_index("Asset Class")
