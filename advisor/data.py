"""Loading and cleaning of the K-Asset fund return workbook.

The workbook (``Fund Return.xlsx``, sheet ``Return``) holds one column per fund
of *simple periodic returns* with a ``Date`` index. Observation frequency is
weekly in the 1990s and daily from the mid-2000s, so anything that annualises
must use the realised observations-per-year of the slice it is looking at
rather than a hard-coded 252.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

SHEET = "Return"

# Return observations larger than this in absolute value are treated as data
# artefacts (dividend/NAV restatements) rather than market moves. The largest
# genuine one-day move in Thai funds during 2008 was about -26%.
_ARTEFACT_THRESHOLD = 0.35

_SEARCH_PATHS: Sequence[Path] = (
    Path(__file__).resolve().parent.parent / "data" / "Fund Return.xlsx",
    Path.home() / "Downloads" / "Fund Return.xlsx",
    Path.cwd() / "Fund Return.xlsx",
)


def resolve_workbook(explicit: Optional[str] = None) -> Path:
    """Find the workbook, honouring ``FUND_RETURN_XLSX`` if set."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("FUND_RETURN_XLSX")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(_SEARCH_PATHS)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find 'Fund Return.xlsx'. Put it in the app's data/ folder or "
        "set the FUND_RETURN_XLSX environment variable."
    )


@dataclass
class FundData:
    """Cleaned return panel plus the provenance needed to explain it."""

    returns: pd.DataFrame          # daily/weekly simple returns, NaN where absent
    source: Path
    artefacts: pd.DataFrame        # rows removed as data artefacts
    first_valid: pd.Series         # per-fund inception in this dataset
    last_valid: pd.Series

    @property
    def codes(self) -> List[str]:
        return list(self.returns.columns)

    @property
    def start(self) -> pd.Timestamp:
        return self.returns.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.returns.index[-1]

    def slice(self, start=None, end=None, codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
        df = self.returns if codes is None else self.returns[list(codes)]
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def common_history(self, codes: Sequence[str], start=None, end=None) -> pd.DataFrame:
        """Rows where *every* requested fund has an observation."""
        return self.slice(start, end, codes).dropna(how="any")

    def coverage_start(self, codes: Sequence[str]) -> pd.Timestamp:
        """Latest inception among ``codes`` — where common history can begin."""
        firsts = [self.first_valid[c] for c in codes if c in self.first_valid.index]
        firsts = [f for f in firsts if pd.notna(f)]
        return max(firsts) if firsts else self.start


def load_workbook(path: Optional[str] = None) -> FundData:
    src = resolve_workbook(path)
    raw = pd.read_excel(src, sheet_name=SHEET)
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.loc[:, ~raw.columns.duplicated()]

    # Drop columns that are entirely empty.
    raw = raw.loc[:, raw.notna().any()]

    # Quarantine implausible observations instead of silently keeping them:
    # K-FIXED-A carries a -54% print on 1997-12-26 that is a NAV restatement,
    # not a bond-market move, and it would dominate every risk estimate.
    mask = raw.abs() > _ARTEFACT_THRESHOLD
    artefacts = (
        raw.where(mask)
        .stack()
        .rename("return")
        .reset_index()
        .rename(columns={"level_1": "fund"})
        if mask.any().any()
        else pd.DataFrame(columns=["Date", "fund", "return"])
    )
    cleaned = raw.mask(mask)

    return FundData(
        returns=cleaned,
        source=src,
        artefacts=artefacts,
        first_valid=cleaned.apply(lambda s: s.first_valid_index()),
        last_valid=cleaned.apply(lambda s: s.last_valid_index()),
    )


@lru_cache(maxsize=4)
def load_cached(path: Optional[str] = None) -> FundData:
    return load_workbook(path)


# --------------------------------------------------------------------------- #
# Frequency helpers
# --------------------------------------------------------------------------- #
def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Realised observation frequency of a date index (obs per calendar year)."""
    if len(index) < 3:
        return 252.0
    span_days = (index[-1] - index[0]).days
    if span_days <= 0:
        return 252.0
    return float(len(index) - 1) * 365.25 / span_days


def to_monthly(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compound periodic returns into calendar-month returns."""
    return (1.0 + returns).resample("M").prod() - 1.0


def cumulative(returns: pd.Series | pd.DataFrame, base: float = 1.0):
    return base * (1.0 + returns.fillna(0.0)).cumprod()


def portfolio_returns(
    returns: pd.DataFrame,
    weights: dict,
    rebalance: str = "Q",
) -> pd.Series:
    """Return series of a weighted basket with periodic rebalancing.

    ``rebalance`` accepts a pandas offset alias (``M``, ``Q``, ``A``), ``"D"``
    for constant-mix (rebalance every observation) or ``"none"`` for buy & hold.
    """
    codes = [c for c in weights if c in returns.columns]
    if not codes:
        return pd.Series(dtype=float)
    w0 = np.array([weights[c] for c in codes], dtype=float)
    total = w0.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    w0 = w0 / total

    r = returns[codes].dropna(how="all")
    r = r.fillna(0.0).to_numpy()
    idx = returns[codes].dropna(how="all").index

    if rebalance == "D":
        return pd.Series(r @ w0, index=idx, name="portfolio")

    if rebalance == "none":
        marks = np.zeros(len(idx), dtype=bool)
    else:
        period = pd.Series(idx, index=idx).dt.to_period(rebalance)
        marks = period.ne(period.shift()).to_numpy()
        marks[0] = False  # already at target on day one

    out = np.empty(len(idx))
    w = w0.copy()
    for t in range(len(idx)):
        if marks[t]:
            w = w0.copy()
        step = w @ r[t]
        out[t] = step
        grown = w * (1.0 + r[t])
        s = grown.sum()
        w = grown / s if s > 0 else w0.copy()
    return pd.Series(out, index=idx, name="portfolio")
