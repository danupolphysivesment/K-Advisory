"""Visual system for the app: palette, CSS, Plotly template and components.

The look follows the K-Asset Investment Bootcamp KTM2025 theme — deep teal
canvas, mint primary accent, sand secondary. The interface is Thai, so text is
set in Sarabun and only figures keep Georgia's print-like numerals.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from . import universe as _u

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
BG = "#003336"          # canvas
PANEL = "#04282B"       # sidebar / recessed panel
CARD = "#0A3A3E"        # card fill, slightly lifted
CARD_HI = "#0E464A"     # hover / emphasis
LINE = "#155055"        # hairline borders

MINT = "#40F8A9"        # primary accent
MINT_DIM = "#2BC486"
TEAL = "#00868D"        # secondary
SAND = "#EEC295"        # warm accent
TAUPE = "#95887D"
CORAL = "#E5736A"       # negative / breach
AMBER = "#E8B44A"       # warning / watch
SKY = "#6FC8E8"         # informational

TEXT = "#EAF3F1"
MUTED = "#8FB5B2"
DIM = "#5E8B89"

# Thai needs a font with Thai glyphs; Georgia and Trebuchet MS have none, so
# Thai text in them silently falls back to whatever the browser picks and the
# line rhythm falls apart. Sarabun is the Thai government standard face and
# reads as a document font rather than a UI font, which suits this app.
# Loaded from Google Fonts with system Thai faces behind it (Thonburi on macOS,
# Leelawadee UI on Windows) so the app still sets properly offline.
SANS = ("'Sarabun', 'IBM Plex Sans Thai', 'Noto Sans Thai', 'Thonburi', "
        "'Leelawadee UI', 'Trebuchet MS', sans-serif")

# Reserved for figures — percentages, ratios, currency. Latin digits only, so
# Georgia is safe here and keeps the editorial, print-like numerals.
SERIF = "Georgia, 'Times New Roman', serif"

# Thai headings: Sarabun at a heavier weight rather than a serif, because Thai
# serif faces are rare and read as decorative.
DISPLAY = SANS

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Sarabun:wght@300;400;500;600;700&display=swap');"
)

# Asset-class colours, used consistently in every chart and table.
# Keyed off the constants in advisor.universe so a wording change there cannot
# silently drop a colour.
CLASS_COLORS: Dict[str, str] = {
    _u.AC_MONEY: "#7FD4C1",
    _u.AC_TH_FI: TEAL,
    _u.AC_GL_FI: "#2F9BA2",
    _u.AC_TH_EQ: MINT,
    _u.AC_GL_EQ: "#8FE9C4",
    _u.AC_ASIA_EQ: "#B7E86F",
    _u.AC_SECTOR: SAND,
    _u.AC_ALLOC: "#5FA8AE",
    _u.AC_ALT: "#D89A5C",
    "อื่น ๆ": TAUPE,
}

BUCKET_COLORS: Dict[str, str] = {
    _u.CASH: "#7FD4C1",
    _u.FIXED: TEAL,
    _u.EQUITY: MINT,
    _u.ALT: SAND,
}

SEVERITY_COLORS: Dict[str, str] = {
    "Acute": CORAL,
    "Elevated": AMBER,
    "Monitor": SKY,
    "Background": DIM,
    "Breach": CORAL,
    "Watch": AMBER,
    "Compliant": MINT,
}

SEQUENCE: List[str] = [MINT, TEAL, SAND, "#B7E86F", SKY, "#D89A5C", "#5FA8AE",
                       TAUPE, "#8FE9C4", CORAL]


def series_color(index: int) -> str:
    return SEQUENCE[index % len(SEQUENCE)]


def pnl_color(value: float) -> str:
    return MINT if value >= 0 else CORAL


def rgba(hex_color: str, alpha: float) -> str:
    """``#RRGGBB`` plus an alpha, as ``rgba(...)``.

    CSS accepts 8-digit hex but Plotly does not, so any translucent fill handed
    to a figure has to come through here.
    """
    h = hex_color.lstrip("#")
    if len(h) == 8:
        h = h[:6]
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.3f})"


# --------------------------------------------------------------------------- #
# Plotly template
# --------------------------------------------------------------------------- #
def register_template() -> None:
    axis = dict(
        gridcolor=LINE,
        zerolinecolor=LINE,
        linecolor=LINE,
        tickfont=dict(family=SANS, size=11, color=MUTED),
        title=dict(font=dict(family=SANS, size=11, color=DIM)),
        showspikes=False,
    )
    pio.templates["kasset"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=SANS, size=12, color=TEXT),
            title=dict(font=dict(family=SANS, size=14, color=TEXT), x=0, xanchor="left"),
            colorway=SEQUENCE,
            xaxis=axis,
            yaxis=axis,
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(family=SANS, size=11, color=MUTED),
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            ),
            hoverlabel=dict(
                bgcolor=CARD_HI, bordercolor=LINE,
                font=dict(family=SANS, size=12, color=TEXT),
            ),
            margin=dict(l=8, r=8, t=40, b=8),
        )
    )
    pio.templates.default = "kasset"


CSS = f"""
<style>
  {FONT_IMPORT}
  /* ---------- canvas ---------- */
  .stApp {{ background: {BG}; color: {TEXT}; }}
  html, body, [class*="css"], .stMarkdown, input, textarea, button, select,
  .stSelectbox, .stRadio, .stSlider {{ font-family: {SANS}; }}
  .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  [data-testid="stDecoration"] {{ display: none; }}
  [data-testid="stToolbar"] {{ display: none; }}

  h1, h2, h3, h4 {{ font-family: {DISPLAY}; font-weight: 600; color: {TEXT}; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.75rem; }}
  h2 {{ font-size: 1.3rem; margin-top: 0.4rem; }}
  h3 {{ font-size: 1.05rem; }}
  a {{ color: {MINT}; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  hr {{ border-color: {LINE}; }}

  /* ---------- sidebar ---------- */
  section[data-testid="stSidebar"] {{
    background: {PANEL}; border-right: 1px solid {LINE};
  }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
  section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
    font-size: 0.8rem; font-family: {SANS}; letter-spacing: 0.04em;
    color: {DIM}; font-weight: 600;
    margin: 1.3rem 0 0.4rem 0;
  }}
  section[data-testid="stSidebar"] label {{ color: {MUTED} !important; font-size: 0.82rem; }}

  /* ---------- masthead ---------- */
  .masthead {{
    display: flex; align-items: baseline; gap: 0.85rem; flex-wrap: nowrap;
    border-bottom: 1px solid {LINE}; padding-bottom: 0.7rem; margin-bottom: 1.1rem;
  }}
  .masthead .mark {{
    font-family: {SERIF}; font-size: 1.55rem; color: {TEXT}; letter-spacing: -0.02em;
    white-space: nowrap;
  }}
  .masthead .meta, .klabel, .chip {{ font-family: {SANS}; }}
  .masthead .mark b {{ color: {MINT}; font-weight: 400; }}
  .masthead .rule {{ flex: 1; height: 1px; background: {LINE}; }}
  .masthead .meta {{
    font-size: 0.72rem; letter-spacing: 0.04em; color: {DIM};
  }}

  /* ---------- cards ---------- */
  .kcard {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 3px;
    padding: 0.95rem 1.1rem; height: 100%;
  }}
  .kcard.flush {{ padding: 0.75rem 0.9rem; }}
  .klabel {{
    font-size: 0.72rem; letter-spacing: 0.02em;
    color: {DIM}; font-weight: 600; margin-bottom: 0.3rem;
  }}
  .kvalue {{
    font-family: {SERIF}; font-size: 1.65rem; line-height: 1.1; color: {TEXT};
    font-variant-numeric: tabular-nums;
  }}
  .kvalue.sm {{ font-size: 1.25rem; }}
  .kvalue.pos {{ color: {MINT}; }}
  .kvalue.neg {{ color: {CORAL}; }}
  .knote {{ font-size: 0.76rem; color: {MUTED}; margin-top: 0.3rem; line-height: 1.65; }}
  .kdelta {{ font-size: 0.75rem; font-variant-numeric: tabular-nums; }}
  .kdelta.pos {{ color: {MINT}; }}
  .kdelta.neg {{ color: {CORAL}; }}

  /* ---------- chips ---------- */
  .chip {{
    display: inline-block; padding: 0.16rem 0.55rem; border-radius: 2px;
    font-size: 0.7rem; letter-spacing: 0.02em;
    font-weight: 600; border: 1px solid; margin-right: 0.35rem;
  }}
  .chip.mint {{ color: {MINT}; border-color: {MINT}44; background: {MINT}14; }}
  .chip.coral {{ color: {CORAL}; border-color: {CORAL}44; background: {CORAL}14; }}
  .chip.amber {{ color: {AMBER}; border-color: {AMBER}44; background: {AMBER}14; }}
  .chip.sky {{ color: {SKY}; border-color: {SKY}44; background: {SKY}14; }}
  .chip.dim {{ color: {MUTED}; border-color: {LINE}; background: transparent; }}
  .chip.sand {{ color: {SAND}; border-color: {SAND}44; background: {SAND}14; }}

  /* ---------- client header ---------- */
  .clienthdr {{
    background: linear-gradient(90deg, {CARD} 0%, {PANEL} 100%);
    border: 1px solid {LINE}; border-left: 3px solid {MINT};
    padding: 0.85rem 1.15rem; margin-bottom: 1rem; border-radius: 2px;
  }}
  .clienthdr .name {{ font-family: {DISPLAY}; font-weight: 600; font-size: 1.25rem; color: {TEXT}; }}
  .clienthdr .sub {{ font-size: 0.82rem; color: {MUTED}; margin-top: 0.2rem; line-height: 1.7; }}

  /* ---------- alerts ---------- */
  .kalert {{
    border-left: 3px solid {LINE}; background: {CARD};
    padding: 0.7rem 0.95rem; margin-bottom: 0.5rem; border-radius: 0 2px 2px 0;
    font-size: 0.86rem; line-height: 1.75; color: {TEXT};
  }}
  .kalert.breach {{ border-left-color: {CORAL}; background: {CORAL}0F; }}
  .kalert.watch {{ border-left-color: {AMBER}; background: {AMBER}0D; }}
  .kalert.ok {{ border-left-color: {MINT}; background: {MINT}0D; }}
  .kalert.info {{ border-left-color: {SKY}; background: {SKY}0D; }}
  .kalert b {{ color: {TEXT}; font-weight: 600; }}
  .kalert .src {{ color: {DIM}; font-size: 0.72rem; }}

  /* ---------- section rule ---------- */
  .krule {{
    display: flex; align-items: center; gap: 0.7rem; margin: 1.5rem 0 0.8rem 0;
  }}
  .krule .t {{
    font-size: 0.78rem; letter-spacing: 0.06em;
    color: {MUTED}; font-weight: 600; white-space: nowrap;
  }}
  .krule .l {{ flex: 1; height: 1px; background: {LINE}; }}

  /* ---------- tabs ---------- */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 0; border-bottom: 1px solid {LINE}; background: transparent;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent; border: none; border-bottom: 2px solid transparent;
    border-radius: 0; padding: 0.6rem 1.15rem; color: {MUTED};
    font-size: 0.83rem; letter-spacing: 0.03em;
  }}
  .stTabs [aria-selected="true"] {{
    color: {TEXT} !important; border-bottom-color: {MINT} !important;
    background: transparent !important;
  }}
  .stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
  .stTabs [data-baseweb="tab-border"] {{ display: none; }}

  /* ---------- widgets ---------- */
  .stButton > button {{
    background: {MINT}; color: {BG}; border: none; border-radius: 2px;
    font-weight: 600; font-size: 0.82rem; letter-spacing: 0.03em;
    padding: 0.42rem 1.1rem;
  }}
  .stButton > button:hover {{ background: #57FFB8; color: {BG}; }}
  .stDownloadButton > button {{
    background: transparent; color: {MINT}; border: 1px solid {MINT}66;
    border-radius: 2px; font-size: 0.78rem;
  }}
  /* Force control text onto the dark canvas. Streamlit only ships its dark
     palette when the theme is configured for the working directory it was
     launched from, and this app is often launched from a parent folder — so
     the base theme's near-black input text can otherwise land on teal. */
  div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
  div[data-baseweb="base-input"] {{
    background: {PANEL}; border-color: {LINE}; border-radius: 2px;
  }}
  div[data-baseweb="select"] *, div[data-baseweb="input"] *,
  div[data-baseweb="base-input"] *, div[data-baseweb="popover"] li,
  .stNumberInput input, .stTextInput input, .stSelectbox div,
  .stMultiSelect div, textarea {{
    color: {TEXT} !important;
  }}
  div[data-baseweb="select"] svg, div[data-baseweb="input"] svg {{
    fill: {MUTED};
  }}
  div[data-baseweb="popover"] div[role="listbox"],
  div[data-baseweb="menu"], ul[role="listbox"] {{
    background: {PANEL} !important; border: 1px solid {LINE};
  }}
  li[role="option"]:hover, div[role="option"]:hover {{
    background: {CARD_HI} !important;
  }}
  .stNumberInput button {{ background: {CARD} !important; color: {MUTED} !important; }}
  [data-testid="stWidgetLabel"] p, .stSlider label, .stCheckbox label span,
  [data-testid="stMarkdownContainer"] p {{ color: {TEXT}; }}
  /* Slider read-out and end labels */
  [data-testid="stTickBar"], [data-testid="stTickBarMin"],
  [data-testid="stTickBarMax"], .stSlider [data-testid="stThumbValue"] {{
    color: {MUTED} !important;
  }}
  .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {MINT}; }}
  [data-testid="stMetricValue"] {{ font-family: {SERIF}; color: {TEXT}; }}
  [data-testid="stMetricLabel"] {{ color: {DIM}; }}

  /* ---------- expander ---------- */
  .streamlit-expanderHeader, [data-testid="stExpander"] summary {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 2px;
    color: {MUTED}; font-size: 0.82rem;
  }}
  [data-testid="stExpander"] {{ border: none; }}
  [data-testid="stExpander"] details {{
    border: 1px solid {LINE}; border-radius: 2px; background: {CARD}80;
  }}

  /* ---------- dataframe ---------- */
  [data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 2px; }}
  [data-testid="stTable"] td, [data-testid="stTable"] th {{ border-color: {LINE}; }}

  /* ---------- misc ---------- */
  .kfoot {{
    border-top: 1px solid {LINE}; margin-top: 2.5rem; padding-top: 0.8rem;
    font-size: 0.7rem; color: {DIM}; line-height: 1.7;
  }}
  .kbar-track {{
    height: 5px; background: {LINE}; border-radius: 3px; overflow: hidden;
    margin-top: 0.35rem;
  }}
  .kbar-fill {{ height: 100%; border-radius: 3px; }}
  .ksmall {{ font-size: 0.8rem; color: {MUTED}; line-height: 1.75; }}
</style>
"""


def apply() -> None:
    """Inject the stylesheet and register the Plotly template."""
    st.markdown(CSS, unsafe_allow_html=True)
    register_template()


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def masthead(title_plain: str, title_accent: str, meta: str) -> None:
    st.markdown(
        f'<div class="masthead"><span class="mark">{title_plain}'
        f'<b>{title_accent}</b></span><span class="rule"></span>'
        f'<span class="meta">{meta}</span></div>',
        unsafe_allow_html=True,
    )


def rule(text: str) -> None:
    st.markdown(f'<div class="krule"><span class="t">{text}</span>'
                f'<span class="l"></span></div>', unsafe_allow_html=True)


def chip(text: str, tone: str = "dim") -> str:
    return f'<span class="chip {tone}">{text}</span>'


def metric_card(label: str, value: str, note: str = "", tone: str = "",
                delta: Optional[str] = None, delta_tone: str = "") -> str:
    delta_html = (f'<div class="kdelta {delta_tone}">{delta}</div>'
                  if delta else "")
    note_html = f'<div class="knote">{note}</div>' if note else ""
    return (f'<div class="kcard"><div class="klabel">{label}</div>'
            f'<div class="kvalue {tone}">{value}</div>{delta_html}{note_html}</div>')


def metric_row(cards: Sequence[str]) -> None:
    cols = st.columns(len(cards), gap="small")
    for col, html in zip(cols, cards):
        with col:
            st.markdown(html, unsafe_allow_html=True)


def alert(text: str, tone: str = "info") -> None:
    st.markdown(f'<div class="kalert {tone}">{text}</div>', unsafe_allow_html=True)


def bar(fraction: float, color: str = MINT) -> str:
    pct = max(0.0, min(float(fraction), 1.0)) * 100
    return (f'<div class="kbar-track"><div class="kbar-fill" '
            f'style="width:{pct:.1f}%;background:{color};"></div></div>')


def client_header(name: str, subtitle: str, chips: Iterable[str]) -> None:
    chip_html = "".join(chips)
    st.markdown(
        f'<div class="clienthdr"><div class="name">{name}</div>'
        f'<div class="sub">{subtitle}</div>'
        f'<div style="margin-top:0.5rem">{chip_html}</div></div>',
        unsafe_allow_html=True,
    )


def caption(text: str) -> None:
    st.markdown(f'<div class="ksmall">{text}</div>', unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown(f'<div class="kfoot">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Table styling
# --------------------------------------------------------------------------- #
def style_elementwise(styler, func, subset):
    """``Styler.map`` on pandas ≥ 2.1, ``Styler.applymap`` before that."""
    apply_fn = getattr(styler, "map", None) or styler.applymap
    return apply_fn(func, subset=subset)


def style_frame(frame, percent_cols=(), number_cols=(), signed_cols=(),
                precision: int = 2):
    """A dark-theme pandas Styler consistent with the rest of the app."""
    styler = frame.style

    fmt: Dict[str, object] = {}
    for col in percent_cols:
        if col in frame.columns:
            fmt[col] = "{:.1%}"
    for col in number_cols:
        if col in frame.columns:
            fmt[col] = f"{{:.{precision}f}}"
    if fmt:
        styler = styler.format(fmt, na_rep="—")

    for col in signed_cols:
        if col in frame.columns:
            styler = style_elementwise(
                styler,
                lambda v: f"color: {MINT if (v or 0) >= 0 else CORAL}", [col])

    return styler.set_properties(**{
        "background-color": CARD,
        "color": TEXT,
        "border-color": LINE,
        "font-family": SANS,
        "font-size": "0.82rem",
    }).set_table_styles([
        {"selector": "th", "props": [
            ("background-color", PANEL), ("color", DIM),
            ("font-family", SANS), ("font-size", "0.76rem"),
            ("letter-spacing", "0.02em"), ("border-color", LINE)]},
    ])
