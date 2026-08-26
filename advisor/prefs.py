"""Sticky settings: session state seeded from, and written back to, the URL.

Two things reset an RM's settings mid-conversation. Widgets rendered without a
``key`` are re-created from their default arguments whenever those arguments
change, and — the one that actually bites on Streamlit Community Cloud — a
dropped websocket starts a brand new session, wiping ``st.session_state``
entirely. Neither is recoverable from inside a normal widget callback.

So settings live in the query string. Each control declares a key here, seeds
its session state from the URL on first render, and the URL is rewritten
afterwards. That makes the settings survive a reconnect and makes any view
shareable as a link, which is the same mechanism already used for ``?client=``.

Only values that differ from the default are written, so the common case stays
a clean ``?client=C06`` rather than a wall of parameters.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import streamlit as st


# --------------------------------------------------------------------------- #
# Query-string access, across Streamlit versions
# --------------------------------------------------------------------------- #
def _read_params() -> Dict[str, str]:
    try:
        return {k: v for k, v in st.query_params.items()}
    except AttributeError:                       # Streamlit < 1.30
        raw = st.experimental_get_query_params()
        return {k: (v[0] if isinstance(v, list) and v else v)
                for k, v in raw.items()}


def _write_params(values: Dict[str, str]) -> None:
    try:
        st.query_params.clear()
        for key, value in values.items():
            st.query_params[key] = value
    except AttributeError:                       # Streamlit < 1.30
        st.experimental_set_query_params(**values)


# --------------------------------------------------------------------------- #
# Declaring a setting
# --------------------------------------------------------------------------- #
_DEFAULTS: Dict[str, Any] = {}


def resolve(raw: Optional[str], default: Any,
            cast: Callable[[str], Any] = str,
            valid: Optional[Iterable[Any]] = None) -> Any:
    """Turn one raw query-string value into a usable setting.

    Pure, so the parsing rules can be tested without a Streamlit session: a
    missing, malformed, or hand-edited-to-nonsense parameter always falls back
    to the default rather than crashing the app or handing a widget an option
    it does not offer.
    """
    if raw is None:
        return default
    try:
        parsed = cast(raw)
    except (TypeError, ValueError):
        return default
    return parsed if (valid is None or parsed in valid) else default


def remember(key: str, default: Any, cast: Callable[[str], Any] = str,
             valid: Optional[Iterable[Any]] = None) -> Any:
    """Seed ``st.session_state[key]`` from the URL, once, and return it.

    After this, the widget owns the value: pass ``key=key`` and *omit*
    ``value=``/``index=``, so Streamlit reads and writes session state directly
    instead of resetting to a literal on every rerun.
    """
    _DEFAULTS[key] = default
    if key in st.session_state:
        return st.session_state[key]

    value = resolve(_read_params().get(key), default, cast, valid)
    st.session_state[key] = value
    return value


def index_of(key: str, options, fallback: int = 0) -> int:
    """Position of the remembered value in ``options``, for a selectbox."""
    try:
        return list(options).index(st.session_state[key])
    except (KeyError, ValueError):
        return fallback


def sync_url(keys: Iterable[str], always: Optional[Dict[str, str]] = None) -> None:
    """Rewrite the query string from session state.

    Writes only when the result differs from what is already there: assigning
    to ``st.query_params`` triggers a rerun, so an unconditional write would
    put the app in a permanent rerun loop.
    """
    wanted: Dict[str, str] = dict(always or {})
    for key in keys:
        if key not in st.session_state:
            continue
        value = st.session_state[key]
        if value == _DEFAULTS.get(key):
            continue                              # keep the URL short
        if isinstance(value, bool):
            wanted[key] = "1" if value else "0"
        else:
            wanted[key] = str(value)

    if _read_params() != wanted:
        _write_params(wanted)


def encode_holdings(weights: Dict[str, float]) -> str:
    """Pack a hand-built portfolio into one query-string value.

    Fund codes carry hyphens and parentheses but never a comma or a colon, so
    ``CODE:12.5,CODE:30`` round-trips without escaping. Weights are percentages
    rounded to one decimal — enough for an allocation, short enough that a
    fifteen-fund book still makes a link an RM can paste into an email.
    """
    parts = [f"{code}:{weight * 100:.4g}"
             for code, weight in sorted(weights.items()) if weight > 1e-9]
    return ",".join(parts)


def decode_holdings(raw: Optional[str]) -> Dict[str, float]:
    """Unpack :func:`encode_holdings`, ignoring anything malformed.

    A hand-edited link is a normal thing to receive, so every failure mode here
    drops the offending pair rather than raising: a broken URL costs the RM the
    positions it could not read, never the session.
    """
    out: Dict[str, float] = {}
    if not raw:
        return out
    for pair in str(raw).split(","):
        code, sep, value = pair.partition(":")
        code = code.strip()
        if not sep or not code:
            continue
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if weight > 0:
            out[code] = out.get(code, 0.0) + weight / 100.0
    return out


def as_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def reset(keys: Iterable[str]) -> None:
    """Drop remembered values so the controls fall back to their defaults."""
    for key in keys:
        st.session_state.pop(key, None)


# --------------------------------------------------------------------------- #
# A slider you can also type into
# --------------------------------------------------------------------------- #
def number_slider(
    label: str,
    key: str,
    min_value,
    max_value,
    step,
    *,
    fmt: Optional[str] = None,
    help: Optional[str] = None,
    disabled: bool = False,
    ratio=(3, 1),
):
    """A slider paired with a number box, both driving one remembered value.

    Streamlit has no widget that is both, and two widgets cannot share a key.
    So each gets its own key and an ``on_change`` callback that writes the
    canonical value plus its twin. Callbacks run before the next render, which
    is the only point at which another widget's state may still be assigned.

    Call :func:`remember` for ``key`` first; this reads the value from there.
    """
    canonical = st.session_state[key]
    slider_key, number_key = f"{key}__slider", f"{key}__num"
    for twin in (slider_key, number_key):
        if twin not in st.session_state:
            st.session_state[twin] = canonical

    def _from_slider() -> None:
        value = st.session_state[slider_key]
        st.session_state[key] = value
        st.session_state[number_key] = value

    def _from_number() -> None:
        # The box is free text, so clamp before it reaches the slider, which
        # raises on an out-of-range value.
        value = st.session_state[number_key]
        value = min(max(value, min_value), max_value)
        st.session_state[key] = value
        st.session_state[number_key] = value
        st.session_state[slider_key] = value

    left, right = st.columns(ratio, gap="small")
    left.slider(label, min_value, max_value, step=step, format=fmt, key=slider_key,
                help=help, disabled=disabled, on_change=_from_slider)
    right.number_input(label, min_value, max_value, step=step, format=fmt,
                       key=number_key, disabled=disabled,
                       on_change=_from_number, label_visibility="hidden")
    return st.session_state[key]
