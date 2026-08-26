"""Cross-version helpers so the app runs on the pandas the host happens to ship.

Developed against pandas 1.5 / numpy 1.24; Streamlit Community Cloud installs
whatever is current, which as of pandas 2.2 means two behaviour changes that
break code silently or loudly:

1. **Copy-on-Write.** ``Series.to_numpy()`` may hand back a *read-only view*
   of the Series' buffer rather than a copy, so writing one element of it raises
   ``ValueError: assignment destination is read-only``. Anything this app then
   mutates in place has to go through :func:`writable`.

2. **Renamed offset aliases.** ``"M"``, ``"Q"`` and ``"A"`` became ``"ME"``,
   ``"QE"`` and ``"YE"`` for *resampling* (deprecated in 2.2, removed in 3.0),
   while ``to_period`` still wants the short forms. The two APIs no longer take
   the same string, so every frequency is resolved through here.

Both are probed once at import rather than compared against a version string,
because the version at which each landed differs by API.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

__all__ = ["writable", "resample_alias", "period_alias", "REPORT"]


# --------------------------------------------------------------------------- #
# Copy-on-Write
# --------------------------------------------------------------------------- #
def writable(array) -> np.ndarray:
    """A NumPy array that is safe to assign into.

    Copies only when the array cannot already be written to, so this costs
    nothing on the pandas versions that still return owned buffers.
    """
    arr = np.asarray(array)
    return arr if arr.flags.writeable else arr.copy()


# --------------------------------------------------------------------------- #
# Offset aliases
# --------------------------------------------------------------------------- #
_PROBE = pd.Series(1.0, index=pd.date_range("2024-01-01", periods=4, freq="D"))


def _resample_works(alias: str) -> bool:
    try:
        _PROBE.resample(alias).sum()
        return True
    except Exception:
        return False


def _period_works(alias: str) -> bool:
    try:
        pd.Series(_PROBE.index, index=_PROBE.index).dt.to_period(alias)
        return True
    except Exception:
        return False


def _pick(candidates) -> str:
    for alias in candidates:
        if _resample_works(alias):
            return alias
    return candidates[-1]


def _pick_period(candidates) -> str:
    for alias in candidates:
        if _period_works(alias):
            return alias
    return candidates[-1]


# Resampling: prefer the modern name, fall back to the legacy one.
_RESAMPLE: Dict[str, str] = {
    "D": "D",
    "M": _pick(["ME", "M"]),
    "Q": _pick(["QE", "Q"]),
    "A": _pick(["YE", "A"]),
    "Y": _pick(["YE", "A"]),
}

# Periods: "Y" is accepted by every version we support; "A" is deprecated.
_PERIOD: Dict[str, str] = {
    "D": "D",
    "M": _pick_period(["M"]),
    "Q": _pick_period(["Q"]),
    "A": _pick_period(["Y", "A"]),
    "Y": _pick_period(["Y", "A"]),
}


def resample_alias(freq: str) -> str:
    """Offset alias to pass to ``.resample()`` on this pandas."""
    return _RESAMPLE.get(freq, freq)


def period_alias(freq: str) -> str:
    """Offset alias to pass to ``.dt.to_period()`` on this pandas."""
    return _PERIOD.get(freq, freq)


REPORT = (
    f"pandas {pd.__version__} · numpy {np.__version__} · "
    f"resample M→{_RESAMPLE['M']} Q→{_RESAMPLE['Q']} A→{_RESAMPLE['A']} · "
    f"to_period A→{_PERIOD['A']}"
)
