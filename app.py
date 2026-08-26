"""K-ADVISOR — portfolio suggestion workbench for relationship managers.

Three sections, in tabs:

1. **Present portfolio** — what the client owns today: allocation, look-through
   to the SEC suitability buckets, performance, attribution, risk contribution,
   historical stress replay and a forward Monte Carlo.
2. **Suggested portfolio** — a suitability-constrained optimisation against the
   objective the client cares about, compared like for like with what they hold.
3. **Market & literacy** — the macro environment, live news and search trends,
   plain-language explainers, and the engine that pushes those themes back into
   sections 1 and 2 as portfolio-specific cautions.

Run with:  streamlit run app.py --server.port 8570
"""

from __future__ import annotations

import warnings
import datetime as dt
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from advisor import assumptions as cma
from advisor import attribution as attrib
from advisor import cautions as cautionlib
from advisor import charts
from advisor import clients as clientlib
from advisor import compat
from advisor import data as dataio
from advisor import engine as eng
from advisor import market as marketlib
from advisor import metrics
from advisor import montecarlo as mc
from advisor import notes as noteslib
from advisor import optimizer as opt
from advisor import prefs
try:
    from advisor import report as reportlib
except ImportError:                      # reported by the deployment check below
    reportlib = None
from advisor import risk as risklib
from advisor import stress as stresslib
from advisor import th
from advisor import theme as T
from advisor import universe as uni

st.set_page_config(
    page_title="K-ADVISOR · เครื่องมือจัดพอร์ตการลงทุน",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
T.apply()


# --------------------------------------------------------------------------- #
# Deployment self-check
# --------------------------------------------------------------------------- #
# app.py and the advisor package are pushed together or not at all. When only
# some of them arrive, Python raises AttributeError deep inside a callback and
# Streamlit Cloud *redacts the message*, leaving a traceback that names a line
# but not the cause. Checking the handful of symbols this file needs turns a
# half-finished upload into a sentence naming the file to push again.
_REQUIRED_SYMBOLS = [
    ("advisor/data.py", dataio, ["common_floor", "first_day"]),
    ("advisor/charts.py", charts, ["GROWTH_BASE", "baht_label"]),
    ("advisor/metrics.py", metrics, ["annual_return_table"]),
    ("advisor/th.py", th, ["finding_severity"]),
    ("advisor/theme.py", T, ["style_frame", "contrast_ratio"]),
    ("advisor/report.py", reportlib,
     ["build_pdf", "figure_png", "CURRENT_BLOCKS", "PROPOSED_BLOCKS"]),
]

_stale: List[tuple] = []
for _file, _module, _symbols in _REQUIRED_SYMBOLS:
    if _module is None:
        _stale.append((_file, ["(ไม่พบไฟล์นี้บนเซิร์ฟเวอร์)"]))
        continue
    _missing = [n for n in _symbols if not hasattr(_module, n)]
    if _missing:
        _stale.append((_file, _missing))

if _stale:
    st.error("### ไฟล์บนเซิร์ฟเวอร์ไม่ตรงกับ app.py")
    st.markdown(
        "แอปนี้ถูกอัปโหลดขึ้นไปไม่ครบ — `app.py` เป็นเวอร์ชันใหม่ "
        "แต่ไฟล์ในโฟลเดอร์ `advisor/` ยังเป็นของเก่า จึงเรียกใช้ฟังก์ชัน"
        "ที่ยังไม่มีบนเซิร์ฟเวอร์\n\n**ไฟล์ที่ต้องอัปโหลดใหม่:**")
    for _file, _missing in _stale:
        st.markdown(f"- `{_file}` — ขาด {', '.join(f'`{m}`' for m in _missing)}")
    st.markdown(
        "อัปโหลดทั้งโฟลเดอร์ `advisor/` พร้อมกับ `app.py`, `requirements.txt` "
        "และโฟลเดอร์ `assets/` (ฟอนต์ภาษาไทยสำหรับ PDF) แล้ว reboot แอป")
    st.stop()


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="กำลังโหลดข้อมูลกองทุน…")
def load_data():
    fd = dataio.load_workbook()
    return fd, uni.build_universe(fd.codes)


def _check_module_versions() -> None:
    """Catch a half-deployed checkout and name the file that is stale.

    Streamlit Cloud serves whatever is in the repo. If only some files of
    advisor/ are pushed, the app fails deep inside a render with an
    AttributeError that Cloud redacts — unreadable to whoever is demoing it.
    Comparing the schema version the modules agree on turns that into one
    sentence naming the file to update.
    """
    found = getattr(marketlib, "SCHEMA_VERSION", 1)
    needed = getattr(cautionlib, "REQUIRED_SCHEMA", 1)
    if found >= needed:
        return
    st.error("ไฟล์ในโปรเจกต์ไม่ตรงเวอร์ชันกัน")
    st.markdown(
        f"`advisor/market.py` เป็นเวอร์ชันเก่า (schema {found}) "
        f"แต่ `advisor/cautions.py` ต้องการ schema {needed}\n\n"
        "หากเพิ่ง deploy ขึ้น Streamlit Cloud กรุณา commit และ push "
        "**โฟลเดอร์ `advisor/` ทั้งโฟลเดอร์** ไม่ใช่เฉพาะไฟล์ที่แก้ "
        "แล้วกด Reboot app\n\n"
        "ระหว่างนี้แอปยังทำงานได้ แต่ข้อความในการ์ดข้อควรระวัง"
        "จะเป็นข้อความสำรองที่ตัดมาจากคำอธิบายยาว")


def _fail_missing_workbook(exc: Exception) -> None:
    """The one dependency that cannot degrade — say exactly how to fix it."""
    st.error("ไม่พบไฟล์ข้อมูลกองทุน **Fund Return.xlsx**")
    st.markdown(
        "แอปนี้ต้องมีไฟล์ `Fund Return.xlsx` จึงจะทำงานได้ "
        "หากกำลัง deploy บน Streamlit Community Cloud กรุณาตรวจว่า:\n\n"
        "1. ไฟล์ถูก commit เข้า repository ที่พาธ `data/Fund Return.xlsx` "
        "(ขนาดราว 4.3 MB จึงอยู่ในขีดจำกัดของ GitHub ไม่ต้องใช้ Git LFS)\n"
        "2. ไฟล์ไม่ได้ถูกยกเว้นไว้ใน `.gitignore` — ไฟล์ `.xlsx` มักติดกฎ "
        "ยกเว้นโดยไม่ตั้งใจ\n"
        "3. หรือกำหนดตัวแปรสภาพแวดล้อม `FUND_RETURN_XLSX` "
        "ให้ชี้ไปยังตำแหน่งไฟล์\n\n"
        f"รายละเอียดข้อผิดพลาด: `{exc}`")
    st.stop()


@st.cache_data(show_spinner=False)
def cached_analysis(weights_items, start, end, rebalance, rf, cov_method, label):
    fd, universe = load_data()
    return eng.analyse(fd, dict(weights_items), universe, start, end,
                       rebalance, rf, cov_method, label)


@st.cache_data(show_spinner=False, ttl=1800)
def cached_market(with_news: bool, news_days: int):
    return marketlib.build_market_view(with_news=with_news, news_days=news_days,
                                       news_per_theme=4)


@st.cache_data(show_spinner=False)
def cached_stress(weights_items, allow_proxy: bool):
    fd, universe = load_data()
    return stresslib.results_frame(
        stresslib.run_all(fd.returns, dict(weights_items), universe,
                          allow_proxy=allow_proxy))


@st.cache_data(show_spinner=False)
def cached_shocks(weights_items, since):
    fd, _ = load_data()
    panel = stresslib.factor_panel(fd.returns, uni.FACTOR_PROXIES, start=since)
    return stresslib.shock_summary(fd.returns, dict(weights_items), panel)


@st.cache_data(show_spinner=False)
def cached_simulation(weights_items, method, n_paths, n_periods, period,
                      rebalance, drift_scale, student_df, block_size,
                      mu_items, start):
    fd, _ = load_data()
    codes = [c for c, _ in weights_items]
    panel = fd.slice(start, None, codes).dropna(how="any")
    return mc.simulate(panel, dict(weights_items), method=method, n_paths=n_paths,
                       n_periods=n_periods, period=period, rebalance=rebalance,
                       drift_scale=drift_scale, student_df=student_df,
                       block_size=block_size,
                       mu_override=dict(mu_items) if mu_items else None)


@st.cache_data(show_spinner=False)
def cached_optimise(codes, objective, band_items, max_weight, max_satellite,
                    max_funds, min_position, rf, above_codes, above_budget,
                    mu_items, target, since, cov_method):
    fd, universe = load_data()
    panel = fd.slice(since, None, list(codes)).dropna(how="any")
    cov = risklib.covariance(panel, method=cov_method)
    mu = pd.Series(dict(mu_items))
    cons = opt.Constraints(
        max_weight=max_weight, bands=dict(band_items), max_satellite=max_satellite,
        max_funds=max_funds, min_position=min_position, rf=rf,
        above_level_codes=frozenset(above_codes), above_level_budget=above_budget)
    return opt.optimise(panel, list(codes), universe, objective, cons, mu, cov,
                        target=target)


@st.cache_data(show_spinner=False)
def cached_frontier(codes, band_items, max_weight, max_satellite, rf,
                    above_codes, above_budget, mu_items, since, cov_method):
    fd, universe = load_data()
    panel = fd.slice(since, None, list(codes)).dropna(how="any")
    cov = risklib.covariance(panel, method=cov_method)
    mu = pd.Series(dict(mu_items))
    cons = opt.Constraints(
        max_weight=max_weight, bands=dict(band_items), max_satellite=max_satellite,
        max_funds=None, min_position=0.0, rf=rf,
        above_level_codes=frozenset(above_codes), above_level_budget=above_budget)
    return opt.efficient_frontier(panel, list(codes), universe, cons, mu, cov,
                                  n_points=14)


@st.cache_data(show_spinner=False)
def cached_eligible(level, since, allow_above, min_periods):
    fd, universe = load_data()
    return opt.eligible_universe(fd.returns, universe, level,
                                 min_history_periods=min_periods,
                                 require_since=since,
                                 allow_above_level=allow_above)


_check_module_versions()

try:
    FUND_DATA, UNIVERSE = load_data()
except FileNotFoundError as exc:
    _fail_missing_workbook(exc)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def pct(x, dp: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dp}%}"


def num(x, dp: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dp}f}"


def thb(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"฿{x / 1_000_000:,.1f}M"
    if abs(x) >= 1_000:
        return f"฿{x / 1_000:,.0f}k"
    return f"฿{x:,.0f}"


def tone_for(value: float) -> str:
    return "pos" if value >= 0 else "neg"


def as_date(raw) -> dt.date:
    """Parse an ISO date out of the query string.

    Raises on anything else so ``prefs.resolve`` can fall back to the default —
    a hand-edited link must never put the analysis on a window that does not
    exist.
    """
    if isinstance(raw, dt.date):
        return raw
    return dt.date.fromisoformat(str(raw).strip())


def clamp_date(value: dt.date, low: dt.date, high: dt.date) -> dt.date:
    return max(low, min(value, high))


def weights_key(weights: Mapping[str, float]) -> tuple:
    return tuple(sorted((c, round(float(w), 8)) for c, w in weights.items()))


def fund_label(code: str) -> str:
    fund = UNIVERSE.get(code)
    return fund.name if fund else code


SEVERITY_TONE = {"Acute": "coral", "Elevated": "amber",
                 "Monitor": "sky", "Background": "dim"}

# Ledoit-Wolf, always. The alternatives stay in advisor/risk.py and under test,
# but a sample covariance over 70-odd funds and five years of daily data is
# dominated by estimation noise, and an RM has no basis for picking between
# estimators mid-conversation. Choosing one and standing behind it is the
# honest option.
COV_METHOD = risklib.COV_METHODS[0]

# The WealthPLUS glide path is the natural yardstick: they are the one-ticket
# portfolios an RM would otherwise have recommended, so the comparison answers
# "would the client have been better off just buying the ready-made fund?".
# Module scope because the analysis window has to know where the chosen
# benchmark's history begins before the sidebar can seed a start date.
BENCH_FUNDS = {
    "K-WealthPlus Balanced": "K-WPBALANCED",
    "K-WealthPlus SpeedUp": "K-WPSPEEDUP",
    "K-WealthPlus Ultimate": "K-WPULTIMATE",
}
BENCHES = ["ไม่เปรียบเทียบ"] + list(BENCH_FUNDS)
BENCH_DEFAULT = BENCHES[1]

# Printed at the foot of every exported document. An RM hands this to a client,
# so the limits of what the numbers are have to travel with them.
DISCLAIMER = (
    "เอกสารนี้จัดทำโดยเครื่องมือ K-ADVISOR เพื่อประกอบการพูดคุยระหว่างผู้แนะนำ"
    "การลงทุนกับลูกค้าเท่านั้น ไม่ใช่หนังสือชี้ชวนและไม่ใช่คำแนะนำการลงทุน"
    "เฉพาะบุคคล ตัวเลขทั้งหมดคำนวณจากผลตอบแทนย้อนหลังของกองทุนตามช่วงเวลา"
    "ที่ระบุไว้ในเอกสาร ผลการดำเนินงานในอดีตไม่ได้เป็นสิ่งยืนยันถึงผลการ"
    "ดำเนินงานในอนาคต การจำลองสถานการณ์เป็นเพียงการประมาณการภายใต้สมมติฐาน"
    "ที่ระบุไว้ และมูลค่าที่เกิดขึ้นจริงอาจแตกต่างออกไป ผู้ลงทุนควรศึกษา"
    "ข้อมูลในหนังสือชี้ชวนก่อนตัดสินใจลงทุน"
)

# Two ways to arrive at a proposal. The optimiser is the default because it is
# the one that respects the mandate by construction; the hand-built mode exists
# because an RM often already knows what they want to present and needs the
# same statistics computed on it, not a solver's opinion.
MODE_SOLVE = "Optimizer"
MODE_CUSTOM = "กำหนดเอง (Custom)"
BUILD_MODES: List[str] = [MODE_SOLVE, MODE_CUSTOM]

# Every setting the user can change, in the order they appear. Kept in one list
# so the URL sync and the reset button can never drift out of step with the
# controls themselves. Per-client keys ("aum_C06") are handled separately.
REMEMBERED: List[str] = [
    "client", "lookback", "rebalance", "rf", "news", "newsdays",
    "t1_from", "t1_to", "t1_span",
    "t1_bench", "t1_proxy", "t1_periods", "t1_period", "t1_paths",
    "t1_block", "t1_muw",
    "t2_mode", "t2_obj", "t2_tgt_r", "t2_tgt_v", "t2_maxw", "t2_maxn",
    "t2_sat", "t2_muw", "t2_above", "t2_abovebud", "t2_hist", "t2_custom",
    "t3_filter",
]


# --------------------------------------------------------------------------- #
# Hand-built portfolios
# --------------------------------------------------------------------------- #
CUSTOM_KEY = "t2_custom"


def _custom_weights() -> Dict[str, float]:
    """The hand-built book currently held in session state."""
    return dict(st.session_state.get("_custom_w", {}))


def _set_custom(weights: Mapping[str, float]) -> None:
    """Write the book to session state and to the URL-backed string together.

    Two representations because they serve different masters: the dict is what
    the widgets and the maths read, the packed string is what survives a Cloud
    reconnect. They are only ever written here, so they cannot drift.
    """
    clean = {c: float(w) for c, w in weights.items() if float(w) > 1e-9}
    st.session_state["_custom_w"] = clean
    st.session_state[CUSTOM_KEY] = prefs.encode_holdings(clean)


def _pick_order(codes) -> list:
    return sorted(codes, key=lambda c: (UNIVERSE[c].asset_class, fund_label(c)))


def _custom_gen() -> int:
    return int(st.session_state.get("_custom_gen", 0))


def _round_book(weights: Mapping[str, float]) -> Dict[str, float]:
    """Round a seeded allocation to whole basis points that still sum to 100%.

    Eight optimiser weights rounded to two decimals sum to 100.01% about as
    often as not, and a proposal that does not add to a hundred is one an RM has
    to explain. The residual goes on the largest position, where it is smallest
    in relative terms.
    """
    rounded = {c: round(float(w) * 100.0, 2) for c, w in weights.items()
               if float(w) > 1e-9}
    if not rounded:
        return {}
    residual = round(100.0 - sum(rounded.values()), 2)
    biggest = max(rounded, key=lambda c: rounded[c])
    rounded[biggest] = round(rounded[biggest] + residual, 2)
    return {c: w / 100.0 for c, w in rounded.items() if w > 0}


def _seed_custom(weights: Mapping[str, float]) -> None:
    """Replace the book wholesale, from a button rather than from typing.

    Deleting a widget's session-state entry is not enough to reseed it:
    Streamlit keeps the value in its own widget registry and writes it straight
    back on the next run, so the boxes would silently ignore the seed. Bumping a
    generation counter changes every widget key, which creates genuinely new
    widgets that do take the seeded value.
    """
    _set_custom(_round_book(weights))
    stale = [k for k in st.session_state if str(k).startswith(("_cw_", "_pick_"))]
    for key in stale:
        st.session_state.pop(key, None)
    st.session_state["_custom_gen"] = _custom_gen() + 1


def custom_portfolio_editor(eligible, panel_opt, mu_opt, client, solved,
                            max_weight, max_sat, max_funds, above_codes,
                            above_budget, rf_rate, aum):
    """Let the RM pick funds and weights, then score them like any proposal.

    Returns a :class:`opt.Solution` so everything downstream — allocation
    charts, suitability, stress, Monte Carlo, the trade list — runs unchanged,
    or ``None`` while the book is still empty.
    """
    T.rule("สร้างพอร์ตเอง — เลือกกองทุนและกำหนดน้ำหนัก")

    # Seed once per session from the URL, so a shared link opens the same book.
    if "_custom_w" not in st.session_state:
        prefs.remember(CUSTOM_KEY, "")
        restored = prefs.decode_holdings(st.session_state.get(CUSTOM_KEY, ""))
        restored = {c: w for c, w in restored.items() if c in FUND_DATA.returns}
        _set_custom(_round_book(restored or client.holdings))

    s1, s2, s3, s4 = st.columns([1, 1, 1, 2.4])
    with s1:
        if st.button("เริ่มจากพอร์ตปัจจุบัน", use_container_width=True):
            _seed_custom(client.holdings)
            st.rerun()
    with s2:
        if st.button("เริ่มจากผล Optimizer", use_container_width=True,
                     disabled=not solved.weights):
            _seed_custom(solved.weights)
            st.rerun()
    with s3:
        if st.button("ล้างทั้งหมด", use_container_width=True):
            _seed_custom({})
            st.rerun()
    with s4:
        T.caption(
            "เริ่มจากพอร์ตเดิมของลูกค้าแล้วปรับทีละกอง หรือเริ่มจากคำตอบของ "
            "Optimizer แล้วปรับให้ตรงกับที่คุยกับลูกค้าไว้ "
            "ทุกสถิติด้านล่างคำนวณด้วยวิธีเดียวกับโหมด Optimizer "
            "จึงเทียบกันได้ตรง ๆ")

    current = _custom_weights()
    # Anything the client already holds stays selectable even if the eligibility
    # filter would exclude it — the RM has to be able to model what is actually
    # in the account, including the positions they intend to sell.
    options = _pick_order(set(eligible) | set(current) | set(client.holdings))
    gen = _custom_gen()
    pick_key = f"_pick_{gen}"
    if pick_key not in st.session_state:
        st.session_state[pick_key] = _pick_order(current)
    picked = st.multiselect(
        "กองทุนในพอร์ต", options,
        format_func=lambda c: (
            f"{fund_label(c)} · {UNIVERSE[c].asset_class} · "
            f"ความเสี่ยงระดับ {UNIVERSE[c].risk_level}"),
        key=pick_key)

    if not picked:
        T.alert("<b>ยังไม่ได้เลือกกองทุน</b> — เลือกอย่างน้อย 1 กองทุน "
                "หรือกดปุ่มด้านบนเพื่อเริ่มจากพอร์ตที่มีอยู่แล้ว", "watch")
        _set_custom({})
        return None

    # A newly ticked fund starts at nothing rather than at an invented weight:
    # a number the RM did not choose has no business appearing in a proposal.
    weights: Dict[str, float] = {}
    rows = [picked[i:i + 4] for i in range(0, len(picked), 4)]
    for row in rows:
        cols = st.columns(4)
        for col, code in zip(cols, row):
            with col:
                seeded = float(current.get(code, 0.0)) * 100.0
                state_key = f"_cw_{gen}_{code}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = round(seeded, 2)
                weights[code] = st.number_input(
                    fund_label(code), min_value=0.0, max_value=100.0, step=1.0,
                    format="%.2f", key=state_key,
                    help=f"{UNIVERSE[code].asset_class} · "
                         f"{th.role(UNIVERSE[code].role)} · "
                         f"ความเสี่ยงระดับ {UNIVERSE[code].risk_level}") / 100.0

    total = sum(weights.values())
    _set_custom(weights)

    t1, t2c = st.columns([1, 3])
    with t1:
        off = abs(total - 1.0)
        T.metric_row([T.metric_card(
            "น้ำหนักรวม", f"{total:.2%}",
            tone="pos" if off < 5e-4 else "neg",
            note="ครบ 100%" if off < 5e-4 else f"ต่างจาก 100% อยู่ {total - 1:+.2%}")])
    with t2c:
        if off >= 5e-4:
            st.write("")
            if st.button("ปรับให้รวมเป็น 100% (คงสัดส่วนเดิม)",
                         type="primary"):
                if total > 0:
                    _seed_custom({c: w / total for c, w in weights.items()})
                    st.rerun()
            T.caption(
                "สถิติทั้งหมดด้านล่างคำนวณบนพอร์ตที่ปรับให้รวมเป็น 100% แล้ว "
                "เพราะค่าอย่างความผันผวนและ Max Drawdown นิยามไว้บนพอร์ต"
                "ที่ลงทุนเต็มจำนวนเท่านั้น ตัวเลขที่กรอกไว้ไม่ถูกแก้")

    cons = opt.Constraints(
        max_weight=max_weight, max_satellite=max_sat, max_funds=max_funds,
        min_position=0.0, rf=rf_rate,
        above_level_codes=frozenset(above_codes),
        above_level_budget=above_budget)
    cov_custom = risklib.covariance(
        FUND_DATA.slice(panel_opt.index[0], None,
                        sorted(weights)).dropna(how="any"),
        method=COV_METHOD)
    return opt.evaluate(FUND_DATA.returns, weights, UNIVERSE, mu_opt,
                        cov_custom, cons=cons, rf=rf_rate)


# --------------------------------------------------------------------------- #
# PDF export
# --------------------------------------------------------------------------- #
def _frame_rows(frame, formatters: Mapping[str, object]) -> List[List[str]]:
    """A DataFrame as strings, formatted per column, index first."""
    rows = []
    for idx, row in frame.iterrows():
        line = [str(idx)]
        for col in frame.columns:
            fmt = formatters.get(col)
            value = row[col]
            line.append(fmt(value) if fmt else str(value))
        rows.append(line)
    return rows


def _pct_or_dash(value, dp: int = 1) -> str:
    return pct(value, dp)


def export_panel(kind: str, blocks, defaults, build, filename_stem: str) -> None:
    """The tick-list and Generate button that closes each tab.

    Building the PDF is deliberately behind a button rather than reactive:
    assembling every section costs real work, and an RM changes the ticks a few
    times before they mean it.
    """
    T.rule("ส่งออกเป็น PDF — เลือกหัวข้อที่จะใช้คุยกับลูกค้า")

    ok, reason = reportlib.available()
    if not ok:
        T.alert(f"<b>ยังส่งออก PDF ไม่ได้</b> — {reason}", "watch")
        return

    state_key, gen_key = f"_pdfsel_{kind}", f"_pdfgen_n_{kind}"
    if state_key not in st.session_state:
        st.session_state[state_key] = list(defaults)
    chosen = set(st.session_state[state_key])

    def _reseat(keys) -> None:
        # Same reason as the portfolio editor: a tick box keeps its own state,
        # so "select all" only lands if the boxes are rebuilt under new keys.
        st.session_state[state_key] = list(keys)
        st.session_state[gen_key] = int(st.session_state.get(gen_key, 0)) + 1

    a1, a2, a3 = st.columns([1, 1, 3])
    with a1:
        if st.button("เลือกทั้งหมด", key=f"_pdfall_{kind}",
                     use_container_width=True):
            _reseat([b.key for b in blocks])
            st.rerun()
    with a2:
        if st.button("ล้างที่เลือก", key=f"_pdfnone_{kind}",
                     use_container_width=True):
            _reseat([])
            st.rerun()
    with a3:
        T.caption(
            "กราฟในเอกสารใช้สีสำหรับงานพิมพ์บนพื้นขาว จึงไม่เหมือนบนหน้าจอ"
            " แต่เป็นข้อมูลชุดเดียวกันทุกตัว · หัวข้อในเอกสารจะเรียงตามลำดับ"
            " ที่แสดงด้านล่าง โดยตารางมาก่อนกราฟ")

    charts_ok, charts_why = reportlib.charts_available()
    picked: List[str] = []
    gen = int(st.session_state.get(gen_key, 0))

    def _tick_grid(group) -> None:
        columns = st.columns(3)
        for i, block in enumerate(group):
            with columns[i % 3]:
                box_key = f"_pdfbox_{kind}_{gen}_{block.key}"
                if box_key not in st.session_state:
                    st.session_state[box_key] = block.key in chosen
                on = st.checkbox(block.title, key=box_key,
                                 disabled=block.is_chart and not charts_ok,
                                 help=block.note or None)
                if on and not (block.is_chart and not charts_ok):
                    picked.append(block.key)

    st.markdown("**ตารางและข้อความ**")
    _tick_grid([b for b in blocks if not b.is_chart])
    st.markdown("**กราฟ**")
    if not charts_ok:
        T.alert(f"<b>ใส่กราฟลงใน PDF ไม่ได้</b> — {charts_why}", "watch")
    _tick_grid([b for b in blocks if b.is_chart])
    st.session_state[state_key] = picked

    g1, g2 = st.columns([1, 3])
    with g1:
        generate = st.button(f"สร้าง PDF ({len(picked)} หัวข้อ)",
                             key=f"_pdfgen_{kind}", type="primary",
                             use_container_width=True, disabled=not picked)
    with g2:
        if not picked:
            T.caption("เลือกอย่างน้อยหนึ่งหัวข้อก่อนจึงจะสร้างเอกสารได้")

    if generate:
        with st.spinner("กำลังสร้างเอกสาร…"):
            try:
                st.session_state[f"_pdfdata_{kind}"] = build(picked)
                st.session_state[f"_pdferr_{kind}"] = ""
            except Exception as exc:                  # pragma: no cover
                st.session_state[f"_pdfdata_{kind}"] = None
                st.session_state[f"_pdferr_{kind}"] = str(exc)

    if st.session_state.get(f"_pdferr_{kind}"):
        T.alert(f"<b>สร้างเอกสารไม่สำเร็จ</b> — "
                f"{st.session_state[f'_pdferr_{kind}']}", "breach")
    data = st.session_state.get(f"_pdfdata_{kind}")
    if data:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        st.download_button(
            f"ดาวน์โหลด PDF ({len(data) / 1024:,.0f} KB)", data=data,
            file_name=f"{filename_stem}-{client.id}-{stamp}.pdf",
            mime="application/pdf", key=f"_pdfdl_{kind}")
        T.caption("ไฟล์นี้สร้างจากค่าที่ตั้งไว้ ณ ตอนกดปุ่ม "
                  "หากเปลี่ยนค่าด้านบนแล้ว ให้กดสร้างใหม่อีกครั้ง")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        f'<div style="font-family:{T.SERIF};font-size:1.35rem;color:{T.TEXT};'
        f'letter-spacing:-0.02em;">K<b style="color:{T.MINT};font-weight:400">'
        f'-ADVISOR</b></div>'
        f'<div style="font-size:0.66rem;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:{T.DIM};margin-top:0.1rem;">'
        f'เครื่องมือจัดพอร์ตการลงทุนสำหรับผู้แนะนำการลงทุน</div>',
        unsafe_allow_html=True)

    st.markdown("## ลูกค้า")
    by_profile = clientlib.clients_by_profile()
    options: List[str] = []
    for level, group in by_profile.items():
        for c in group:
            options.append(c.id)
    # Every control below is remembered: seeded from the query string, owned by
    # session state, and written back to the URL at the end of the run. That
    # keeps settings across reruns *and* across a dropped Cloud session, and
    # makes any configured view shareable as a link.
    prefs.remember("client", "C06", valid=options)
    client_id = st.selectbox(
        "เลือกลูกค้า", options, key="client",
        format_func=lambda cid: (
            f"{cid} · {clientlib.CLIENTS_BY_ID[cid].name}  "
            f"(ความเสี่ยง {clientlib.CLIENTS_BY_ID[cid].risk_profile})"),
        label_visibility="collapsed")
    client = clientlib.get_client(client_id)
    profile = client.profile

    # Portfolio value is remembered per client, so editing it for one client
    # neither leaks into the next nor gets lost on the way back.
    aum_key = f"aum_{client_id}"
    prefs.remember(aum_key, float(client.aum_thb), cast=float)
    aum = st.number_input("มูลค่าพอร์ต (บาท)", min_value=1_000_000.0,
                          step=1_000_000.0, format="%.0f", key=aum_key)

    st.markdown("## ช่วงข้อมูลที่ใช้วิเคราะห์")
    # Years rather than a list of presets, so an RM can ask for the exact window
    # a mandate or a review cycle is written against.
    #
    # The floor is what *this* comparison actually has, not what the workbook
    # holds. A portfolio is only as long as its shortest holding, and the
    # benchmark it is drawn against is another constraint on top: asking for 20
    # years of a book whose newest fund launched in 2021 used to print
    # "2006 – 2026" in the header while analysing five years, which is a header
    # that lies about its own numbers.
    # Two different floors, and conflating them would be wrong in both
    # directions. The portfolio floor is a hard limit — no analysis can reach
    # behind the newest holding's launch. The comparison floor sits at or after
    # it and only seeds the *default*, because the stats, attribution and stress
    # on this tab need no benchmark and should not be shortened by one.
    _portfolio_floor = (dataio.common_floor(FUND_DATA.returns, client.holdings)
                        or pd.Timestamp(FUND_DATA.start))
    _bench_first = dataio.first_day(FUND_DATA.returns, BENCH_FUNDS.get(
        st.session_state.get("t1_bench", BENCH_DEFAULT), ""))
    _compare_floor = (max(_portfolio_floor, _bench_first) if _bench_first is not None
                      else _portfolio_floor)

    _available_years = max(
        1, int((pd.Timestamp(FUND_DATA.end) - _portfolio_floor).days // 365))
    _MAX_YEARS = max(2, _available_years)

    # Whether the RM has actually asked for a period, as opposed to landing on
    # the built-in default. An explicit request is honoured exactly; only the
    # untouched default gets nudged to where the benchmark starts.
    _lookback_explicit = ("lookback" in st.query_params
                          or "lookback" in st.session_state)
    # A saved link can still carry a period longer than this client's history —
    # the slider's ceiling moves with the client, the bookmark does not. Pull
    # the stored value into range *before* the widget renders, so the slider,
    # the URL and the dates below all state the same period; leaving it out of
    # range makes the slider show one number while the rest of the app uses
    # another.
    _requested_years = int(prefs.remember("lookback", 5, cast=int))
    _over_run = _requested_years > _MAX_YEARS
    if _over_run:
        st.session_state["lookback"] = _MAX_YEARS

    _years = int(prefs.number_slider(
        "ข้อมูลย้อนหลัง (ปี)", "lookback", 1, _MAX_YEARS, 1, ratio=(2, 1),
        help="ช่วงข้อมูลที่ใช้คำนวณผลตอบแทน ความผันผวน และสหสัมพันธ์ "
             f"พอร์ตนี้มีข้อมูลย้อนหลังได้ {_available_years} ปี "
             f"ตั้งแต่ {th.date(_portfolio_floor.date())} "
             f"ซึ่งเป็นวันที่กองทุนที่ตั้งใหม่ที่สุดในพอร์ตเริ่มมีข้อมูล"))

    # Asking for more than exists opens the window on the first day there is
    # data, rather than on a date the analysis cannot honour.
    window_start = pd.Timestamp(FUND_DATA.end) - pd.DateOffset(years=_years)
    if _over_run or window_start < _portfolio_floor:
        window_start = _portfolio_floor
        T.caption(
            f"พอร์ตนี้มีข้อมูลไม่ถึง {_requested_years} ปี "
            f"จึงเริ่มที่ <b>{th.date(_portfolio_floor.date())}</b> "
            f"ซึ่งเป็นข้อมูลเท่าที่มีทั้งหมด")

    prefs.remember("rebalance", "Q", valid=["Q", "M", "A", "none"])
    rebalance = st.selectbox(
        "Rebalancing", ["Q", "M", "A", "none"], key="rebalance",
        format_func=lambda x: th.REBALANCE[x])
    prefs.remember("rf", 1.75, cast=float)
    rf_rate = prefs.number_slider(
        "Risk-free Rate (% ต่อปี)", "rf", 0.0, 6.0, 0.25, fmt="%.2f",
        help="ใช้แทนอัตราดอกเบี้ยนโยบายไทย สำหรับคำนวณ Sharpe และ Sortino",
        ratio=(2, 1)) / 100.0

    st.markdown("## ข้อมูลตลาด")
    prefs.remember("news", True, cast=prefs.as_bool)
    live_news = st.toggle("ดึงข่าวและเทรนด์แบบเรียลไทม์", key="news",
                          help="ใช้ Google News และ Google Trends RSS "
                               "ปิดเพื่อทำงานจากข้อมูลในเครื่องทั้งหมด")
    prefs.remember("newsdays", 7, cast=int)
    news_days = prefs.number_slider("ช่วงข่าวย้อนหลัง (วัน)", "newsdays", 1, 30, 1,
                                    disabled=not live_news, ratio=(2, 1))
    _b1, _b2 = st.columns(2)
    if _b1.button("รีเฟรชข้อมูลตลาด", use_container_width=True):
        cached_market.clear()
        st.rerun()
    if _b2.button("ล้างค่าที่ตั้งไว้", use_container_width=True,
                  help="คืนทุกการตั้งค่าในทุกหน้ากลับเป็นค่าเริ่มต้น"):
        prefs.reset(REMEMBERED + [k for k in st.session_state
                                  if str(k).startswith("aum_")])
        st.rerun()

    with st.expander("แหล่งที่มาของข้อมูล"):
        T.caption(
            f"<b>ผลตอบแทนกองทุน</b> · {FUND_DATA.source.name}<br>"
            f"{len(FUND_DATA.codes)} กองทุน · "
            f"{th.month(FUND_DATA.start)} – {th.month(FUND_DATA.end)} · "
            f"{len(FUND_DATA.returns):,} ข้อมูล<br>"
            f"คัดข้อมูล {len(FUND_DATA.artefacts)} รายการออก "
            f"เพราะเป็นการปรับ NAV ไม่ใช่การเคลื่อนไหวของตลาด<br><br>"
            f"<b>ข้อมูลตลาด</b> · Yahoo Finance, Google News RSS, "
            f"Google Trends RSS — ดึงสดขณะใช้งานและเก็บแคชไว้ 30 นาที<br><br>"
            f"<b>สภาพแวดล้อม</b> · {compat.REPORT}<br>"
            f"streamlit {st.__version__} · "
            f"schema {getattr(marketlib, 'SCHEMA_VERSION', 1)}")


MARKET = cached_market(live_news, news_days)
CAUTIONS = cautionlib.build_cautions(client.holdings, UNIVERSE, MARKET, limit=6)


# --------------------------------------------------------------------------- #
# Masthead & client header
# --------------------------------------------------------------------------- #
temp_level, temp_msg = MARKET.risk_temperature
_market_state = {"live": "ข้อมูลตลาดสด", "offline": "ไม่มีข้อมูลตลาด"}.get(
    MARKET.macro.status, MARKET.macro.status)
T.masthead("K", "-ADVISOR",
           f"ข้อมูลถึง {th.date(FUND_DATA.end)} · {_market_state} · "
           f"{th.date(pd.Timestamp.now())} {datetime.now():%H:%M}")

findings = clientlib.check_suitability(client.holdings, profile, UNIVERSE)
status = clientlib.suitability_status(findings)
conc = clientlib.concentration_findings(client.holdings, UNIVERSE)

status_tone = {"Breach": "coral", "Watch": "amber", "Compliant": "mint"}[status]
T.client_header(
    f"{client.name}",
    f"{client.persona} · {client.objective} · ระยะเวลาลงทุน "
    f"{client.horizon_years} ปี",
    [
        T.chip(f"ระดับความเสี่ยง {client.risk_profile}", "mint"),
        T.chip(profile.name_th, "dim"),
        T.chip(f"ลงทุนได้ {profile.acceptable_levels}", "dim"),
        T.chip(thb(aum), "sand"),
        T.chip(f"ความเหมาะสม: {th.status(status)}", status_tone),
        T.chip(f"ภาวะตลาด{th.TEMPERATURE[temp_level]}",
               th.TEMPERATURE_TONE[temp_level]),
    ],
)


# --------------------------------------------------------------------------- #
# Shared analysis of the current book
# --------------------------------------------------------------------------- #
# ---- RM's own notes on this client ---------------------------------------- #
with st.expander(
        f"บันทึกเกี่ยวกับลูกค้า — {client.name}"
        + ("  ●" if noteslib.get(client_id).strip() else ""),
        expanded=False):
    st.markdown(
        f'<div class="ksmall" style="margin-bottom:0.6rem">'
        f'<b>ข้อสังเกตจากพอร์ตปัจจุบัน:</b> {client.notes}</div>',
        unsafe_allow_html=True)

    note_key = f"note_{client_id}"
    if note_key not in st.session_state:
        st.session_state[note_key] = noteslib.get(client_id)

    st.text_area(
        "บันทึกของผู้แนะนำการลงทุน", key=note_key, height=140,
        placeholder="เช่น ลูกค้ากังวลเรื่องความผันผวนของหุ้นจีน "
                    "นัดทบทวนพอร์ตอีกครั้งไตรมาสหน้า "
                    "ยังไม่ต้องการเพิ่มสัดส่วนสินทรัพย์ทางเลือก",
        help="บันทึกนี้เก็บแยกตามลูกค้าแต่ละราย และยังอยู่เมื่อสลับลูกค้าหรือ"
             "โหลดหน้าใหม่")

    _n1, _n2 = st.columns([1, 3])
    if _n1.button("บันทึก", use_container_width=True, key="save_note"):
        ok, err = noteslib.save(client_id, st.session_state[note_key])
        if ok:
            _n2.success("บันทึกแล้ว", icon="✅")
        else:
            _n2.warning(f"บันทึกลงไฟล์ไม่สำเร็จ ({err}) — "
                        f"ข้อความยังอยู่จนกว่าจะปิดแอป")
    else:
        _n2.markdown(
            '<div class="ksmall" style="padding-top:0.55rem">'
            'บันทึกเก็บไว้ในไฟล์ของแอป ไม่ได้ส่งออกไปที่ใด '
            'หากรันบน Streamlit Cloud ข้อมูลจะหายเมื่อแอปถูก reboot</div>',
            unsafe_allow_html=True)


# The analysis window, as two dates. The sidebar's year count is a preset that
# seeds them; typing a date wins until the preset is moved again, which is what
# "t1_span" records. An RM needs the exact dates because a review is written
# against the period the client actually held the book, not a round number of
# years back from today.
# The pickers stop at the newest holding's launch. Offering earlier dates would
# let an RM select a window the analysis silently truncates.
DATA_FLOOR = _portfolio_floor.date()
DATA_CEIL = pd.Timestamp(FUND_DATA.end).date()
_preset_from = clamp_date(window_start.date(), DATA_FLOOR, DATA_CEIL)

# On a fresh load the window opens where the portfolio *and* its benchmark both
# have history, so the first thing an RM sees is a comparison that starts from
# the same money on the same day. An explicitly chosen period is left alone —
# if they asked for ten years they get ten years, and the chart trims itself
# and says so instead.
_default_from = clamp_date(
    (window_start if _lookback_explicit
     else max(window_start, _compare_floor)).date(),
    DATA_FLOOR, DATA_CEIL)

prefs.remember("t1_span", _years, cast=int)
prefs.remember("t1_from", _default_from, cast=as_date)
prefs.remember("t1_to", DATA_CEIL, cast=as_date)

def _reseat_window(start: dt.date, end: dt.date) -> None:
    """Move the window programmatically and rebuild the two pickers.

    ``t1_from`` / ``t1_to`` are the stored values the URL carries; the pickers
    get their own generation-stamped keys. Streamlit ignores ``value=`` once a
    widget key holds state, and warns if a keyed widget is also written through
    the Session State API — so a new key is the only way to reseat a date, and
    the split keeps the two rules from colliding.
    """
    st.session_state["t1_from"] = start
    st.session_state["t1_to"] = end
    st.session_state["_date_gen"] = int(st.session_state.get("_date_gen", 0)) + 1


if st.session_state.get("t1_span") != _years:
    # The preset moved, so the dates follow it.
    st.session_state["t1_span"] = _years
    _reseat_window(_preset_from, DATA_CEIL)

win_from = clamp_date(st.session_state["t1_from"], DATA_FLOOR, DATA_CEIL)
win_to = clamp_date(st.session_state["t1_to"], DATA_FLOOR, DATA_CEIL)
WINDOW_INVERTED = win_from >= win_to
if WINDOW_INVERTED:
    # Analyse something rather than crash, and say so where the control is.
    win_from, win_to = _preset_from, DATA_CEIL

analysis_start = pd.Timestamp(win_from)
analysis_end = pd.Timestamp(win_to)

try:
    CURRENT = cached_analysis(weights_key(client.holdings), analysis_start,
                              analysis_end,
                              rebalance, rf_rate, COV_METHOD, "Current")
except Exception as exc:  # pragma: no cover - defensive
    st.error(f"ไม่สามารถวิเคราะห์พอร์ตนี้ได้: {exc}")
    st.stop()


def render_caution_ribbon(cautions, heading: str, limit: int = 3) -> None:
    """The Part-3 → Part-1/2 bridge, rendered at the top of a tab."""
    if not cautions:
        T.alert("ขณะนี้ไม่มีประเด็นตลาดใดที่กระทบสัดส่วนสำคัญของพอร์ตนี้", "ok")
        return
    T.rule(heading)
    T.caption(cautionlib.headline_risk_summary(cautions))
    st.write("")
    cols = st.columns(min(limit, len(cautions)), gap="small")
    for col, c in zip(cols, cautions[:limit]):
        with col:
            tone = SEVERITY_TONE.get(c.severity_key, "dim")
            names = " · ".join(n for _, n, _ in c.holdings[:2])
            if len(c.holdings) > 2:
                names += f" และอีก {len(c.holdings) - 2} กองทุน"
            # The live number, when there is one, sits under the sentence as a
            # separate line so the plain-language part is never crowded out.
            # The reading row is always emitted, hidden when a theme has no live
            # number, so all three cards keep the same structure and the row can
            # equalise their heights instead of one card overflowing.
            reading = (
                f'<div class="knote" style="color:{T.SKY};margin-top:0.4rem'
                f'{"" if c.reading else ";visibility:hidden"}">'
                f'ตัวเลขล่าสุด · {c.reading or "—"}</div>')
            st.markdown(
                f'<div class="kcard flush kcaution">'
                f'<div>{T.chip(c.severity_label, tone)}</div>'
                f'<div style="font-family:{T.SANS};font-weight:600;'
                f'font-size:0.95rem;line-height:1.5;'
                f'margin:0.5rem 0 0.35rem 0;color:{T.TEXT}">{c.short}</div>'
                f'<div class="knote">{c.headline}</div>'
                f'{reading}'
                f'<div class="knote" style="color:{T.DIM};margin-top:0.45rem">'
                f'กองทุนที่เกี่ยวข้อง · {names}</div>'
                f'</div>', unsafe_allow_html=True)


# =========================================================================== #
# TABS
# =========================================================================== #
# No numbering. These are three ways of looking at the same client, not three
# steps in a sequence — an RM opens whichever one the conversation needs, and
# numbering them implies an order that does not exist. The padding that used to
# be faked with spaces is CSS now.
tab1, tab2, tab3 = st.tabs([
    "พอร์ตปัจจุบันของลูกค้า",
    "พอร์ตที่แนะนำ",
    "ภาวะตลาดและความรู้การลงทุน",
])


# --------------------------------------------------------------------------- #
# TAB 1 — Present portfolio
# --------------------------------------------------------------------------- #
with tab1:
    s = CURRENT.stats
    T.metric_row([
        T.metric_card("ผลตอบแทนต่อปี", pct(s.cagr), tone=tone_for(s.cagr),
                      note=f"ตั้งแต่ {th.month(CURRENT.window[0])}"),
        T.metric_card("ความผันผวน", pct(s.volatility),
                      note=f"Sharpe {num(s.sharpe)} · Sortino {num(s.sortino)}"),
        T.metric_card("Max Drawdown", pct(s.max_drawdown), tone="neg",
                      note=(f"{th.month(s.max_dd_start)} → "
                            f"{th.month(s.max_dd_trough)}"
                            if s.max_dd_start is not None else "")),
        T.metric_card("VaR (1 วัน, 95%)", pct(s.var_95),
                      tone="neg",
                      note=f"CVaR {pct(s.cvar_95)} · ขาดทุน "
                           f"{thb(abs(s.cvar_95) * aum)} ในวันที่แย่"),
        T.metric_card("Effective Positions", num(CURRENT.effective_bets, 1),
                      note=f"ถือ {len(CURRENT.weights)} กองทุน · "
                           f"Diversification Ratio "
                           f"{num(CURRENT.diversification_ratio)}"),
    ])

    if CURRENT.coverage_note:
        st.write("")
        T.alert(CURRENT.coverage_note, "info")

    render_caution_ribbon(CAUTIONS, "ข้อควรระวังจากภาวะตลาดสำหรับพอร์ตนี้")

    # ---- suitability -------------------------------------------------------
    T.rule("Suitability — ตรวจตามเกณฑ์ระดับความเสี่ยง")
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.plotly_chart(
            charts.band_compliance(CURRENT.bucket_exposure, profile.bands),
            use_container_width=True, config={"displayModeBar": False})
        T.caption(
            f"<b>{profile.name_th}</b> · คะแนนประเมินความเหมาะสม "
            f"{profile.score_range} · ลงทุนในผลิตภัณฑ์"
            f"{profile.acceptable_levels} ได้ — {profile.description}")
    with c2:
        if not findings and not conc:
            T.alert("<b>ผ่านเกณฑ์ทั้งหมด</b> ทุกกองทุนที่ถืออยู่ในระดับความเสี่ยง"
                    "ที่ลูกค้ารับได้ และสัดส่วนการลงทุนอยู่ในกรอบสำหรับ"
                    "ผู้ลงทุนประเภทนี้", "ok")
        for f in findings[:5]:
            T.alert(f"<b>{th.kind(f.kind)}</b> — {f.detail}",
                    "breach" if f.severity == "breach" else "watch")
        for f in conc[:2]:
            T.alert(f"<b>เกณฑ์ภายในบริษัท</b> — {f.detail}",
                    "breach" if f.severity == "breach" else "watch")

    # ---- allocation --------------------------------------------------------
    T.rule("Asset Allocation")
    a1, a2, a3 = st.columns([1, 1, 1], gap="large")
    with a1:
        st.markdown("**แยกตามกองทุน**")
        st.plotly_chart(
            charts.allocation_donut(CURRENT.weights, CURRENT.fund_labels(UNIVERSE),
                                    centre=f"{len(CURRENT.weights)} กองทุน"),
            use_container_width=True, config={"displayModeBar": False})
    with a2:
        st.markdown("**แยกตาม Asset Class**")
        st.plotly_chart(
            charts.allocation_donut(CURRENT.class_weights,
                                    {k: k for k in CURRENT.class_weights},
                                    colors=T.CLASS_COLORS, centre="กลุ่ม"),
            use_container_width=True, config={"displayModeBar": False})
    with a3:
        st.markdown("**Look-through ตามกลุ่มเกณฑ์ความเหมาะสม**")
        st.plotly_chart(
            charts.allocation_donut(CURRENT.bucket_exposure,
                                    {k: k for k in CURRENT.bucket_exposure},
                                    colors=T.BUCKET_COLORS, centre="กลุ่ม"),
            use_container_width=True, config={"displayModeBar": False})
        T.caption("กองทุนผสมถูกแยกออกเป็นสินทรัพย์อ้างอิงตามจริง "
                  "ดังนั้นกองทุนผสมจะถูกนับเป็นตราสารทุนบางส่วน")

    holdings_rows = []
    for code, w in sorted(CURRENT.weights.items(), key=lambda kv: -kv[1]):
        fund = UNIVERSE[code]
        rc = CURRENT.risk_frame.loc[code] if code in CURRENT.risk_frame.index else None
        holdings_rows.append({
            "กองทุน": code,
            "ชื่อกองทุน": fund.name,
            "Asset Class": fund.asset_class,
            "ภูมิภาค": fund.region,
            "ระดับ": fund.risk_level,
            # Core/Satellite describes how a portfolio was *designed*. The
            # client's existing book was not built on that framework, so
            # printing a role here would invent an intent nobody had. It is
            # stated on the proposal, where the split is a real decision.
            "Core / Satellite": th.NO_DATA,
            "น้ำหนัก": w * 100.0,
            "มูลค่า": f"฿{w * aum:,.0f}",
            "Risk Contribution": (float(rc["risk_share"]) * 100.0
                                  if rc is not None else np.nan),
        })
    holdings = pd.DataFrame(holdings_rows).set_index("กองทุน")
    st.dataframe(
        holdings, use_container_width=True,
        column_config={
            "น้ำหนัก": st.column_config.ProgressColumn(
                "น้ำหนัก", format="%.1f%%", min_value=0.0, max_value=100.0),
            "Risk Contribution": st.column_config.ProgressColumn(
                "Risk Contribution", format="%.1f%%",
                min_value=0.0, max_value=100.0),
            "มูลค่า": st.column_config.TextColumn("มูลค่า (บาท)"),
            "ระดับ": st.column_config.NumberColumn(
                "ระดับความเสี่ยง",
                help="ระดับความเสี่ยงผลิตภัณฑ์ 1-8 ตามหนังสือชี้ชวน"),
        })

    # ---- performance -------------------------------------------------------
    T.rule("ผลการดำเนินงานย้อนหลัง")

    # Placed here rather than in the sidebar because this is the block it is
    # about, but it governs every backward-looking figure on the tab — the
    # headline statistics and the proposal's like-for-like comparison read the
    # same window, so nothing on screen is measured over a different period.
    w1, w2, w3, w4 = st.columns([1, 1, 1, 2.2])
    # date_input range-checks its implicit "today" default before it consults
    # session state, and the workbook ends before today, so the stored value has
    # to be handed in as `value=`. See _reseat_window for why the pickers carry
    # their own keys rather than writing straight to t1_from / t1_to.
    _dgen = int(st.session_state.get("_date_gen", 0))
    with w1:
        picked_from = st.date_input(
            "วันที่เริ่มต้น", value=st.session_state["t1_from"],
            key=f"_dfrom_{_dgen}", min_value=DATA_FLOOR, max_value=DATA_CEIL,
            format="DD/MM/YYYY")
    with w2:
        picked_to = st.date_input(
            "วันที่สิ้นสุด", value=st.session_state["t1_to"],
            key=f"_dto_{_dgen}", min_value=DATA_FLOOR, max_value=DATA_CEIL,
            format="DD/MM/YYYY")
    if (picked_from, picked_to) != (st.session_state["t1_from"],
                                    st.session_state["t1_to"]):
        # Everything on the tab was computed from the old window further up the
        # script, so a new date has to go round again rather than render a
        # chart that disagrees with the control above it.
        st.session_state["t1_from"] = picked_from
        st.session_state["t1_to"] = picked_to
        st.rerun()
    with w3:
        st.write("")
        if st.button("กลับไปใช้ช่วงตาม sidebar", use_container_width=True,
                     help=f"ย้อนกลับไปที่ {_years} ปีล่าสุด "
                          f"ตามที่ตั้งไว้ในแถบด้านซ้าย"):
            _reseat_window(_preset_from, DATA_CEIL)
            st.rerun()
    with w4:
        if WINDOW_INVERTED:
            T.alert("<b>วันที่เริ่มต้นต้องมาก่อนวันที่สิ้นสุด</b> — "
                    "ตัวเลขด้านล่างยังคำนวณจากช่วงเดิมไว้ก่อน", "breach")
        else:
            _days = (win_to - win_from).days
            T.caption(
                f"ช่วงที่วิเคราะห์ <b>{th.date(win_from)} – {th.date(win_to)}</b> "
                f"({_days / 365.25:.1f} ปี · ข้อมูลจริง {len(CURRENT.returns):,} วัน)"
                f"<br>ใช้กับผลการดำเนินงานย้อนหลัง Return Attribution และ "
                f"Risk Contribution รวมถึงสถิติสรุปด้านบนและการเปรียบเทียบ"
                f"กับพอร์ตที่แนะนำ")

    p1, p2 = st.columns([1.6, 1], gap="large")
    with p1:
        prefs.remember("t1_bench", BENCH_DEFAULT, valid=BENCHES)
        bench_choice = st.selectbox("เปรียบเทียบกับ", BENCHES, key="t1_bench")
        series_map = {"พอร์ตปัจจุบัน": CURRENT.returns}
        bench_series = None
        if bench_choice in BENCH_FUNDS:
            bench_series = (FUND_DATA.returns[BENCH_FUNDS[bench_choice]]
                            .reindex(CURRENT.returns.index).dropna())
            if bench_series.empty:
                bench_series = None
            else:
                series_map[bench_choice] = bench_series

        # Both curves start at the same money on the same day or the picture
        # lies. The benchmark funds launched in Aug 2021, so a window that opens
        # earlier used to draw the portfolio from its start and the benchmark
        # from *its* start, both at 10M — handing the benchmark whatever the
        # portfolio did in between as a free head start. Trimming to the common
        # first date is what a factsheet does, and the caption says how much
        # history it cost.
        chart_map, trimmed_from = series_map, None
        if bench_series is not None:
            common = max(CURRENT.returns.index[0], bench_series.index[0])
            if common > CURRENT.returns.index[0]:
                chart_map = {name: s.loc[common:] for name, s in series_map.items()}
                trimmed_from = common

        st.plotly_chart(charts.growth_chart(chart_map),
                        use_container_width=True, config={"displayModeBar": False})
        if trimmed_from is not None:
            T.caption(
                f"กราฟเริ่มที่ <b>{th.date(trimmed_from.date())}</b> ซึ่งเป็นวันแรกที่ "
                f"<b>{bench_choice}</b> มีข้อมูล ไม่ใช่ {th.date(win_from)} "
                f"ตามช่วงที่เลือกไว้ — เพื่อให้ทั้งสองเส้นเริ่มนับจากเงินก้อน"
                f"เดียวกันในวันเดียวกัน สถิติด้านบนยังคำนวณจากช่วงเต็มที่เลือก")
        st.plotly_chart(charts.drawdown_chart(chart_map),
                        use_container_width=True, config={"displayModeBar": False})
    with p2:
        st.markdown("**ผลตอบแทนรายปีปฏิทิน**")
        st.plotly_chart(
            charts.calendar_bars(metrics.calendar_year_returns(CURRENT.returns)),
            use_container_width=True, config={"displayModeBar": False})
        if bench_series is not None and len(bench_series) > 20:
            tr = metrics.tracking_stats(CURRENT.returns, bench_series, rf_rate)
            st.markdown("**เทียบกับตัวเปรียบเทียบ**")
            T.caption(
                f"Beta <b>{num(tr['beta'])}</b> · Alpha <b>{pct(tr['alpha'])}</b><br>"
                f"Tracking error <b>{pct(tr['tracking_error'])}</b> · "
                f"Information ratio <b>{num(tr['information_ratio'])}</b><br>"
                f"จับผลตอบแทนขาขึ้น <b>{pct(tr['up_capture'], 0)}</b> · "
                f"ขาลง <b>{pct(tr['down_capture'], 0)}</b><br>"
                f"สหสัมพันธ์ <b>{num(tr['correlation'])}</b>")

    with st.expander("ตารางผลตอบแทนรายปี"):
        annual = metrics.annual_return_table(CURRENT.returns)
        if annual.empty:
            T.caption("ข้อมูลในช่วงที่เลือกไม่พอสำหรับสรุปรายปี")
        else:
            st.dataframe(
                T.style_frame(annual,
                              percent_cols=["ผลตอบแทน", "ความผันผวน",
                                            "Max Drawdown",
                                            "เดือนที่ดีที่สุด",
                                            "เดือนที่แย่ที่สุด"],
                              signed_cols=["ผลตอบแทน", "Max Drawdown",
                                           "เดือนที่ดีที่สุด",
                                           "เดือนที่แย่ที่สุด"]),
                use_container_width=True)
            T.caption(
                "ปีแรกและปีสุดท้ายมักไม่เต็มปี เพราะถูกตัดด้วยช่วงวันที่ที่เลือกไว้"
                " ด้านบน คอลัมน์วันทำการบอกว่าแต่ละปีมีข้อมูลกี่วัน")

    # ---- attribution -------------------------------------------------------
    T.rule("Return Attribution")
    at1, at2 = st.columns([1.3, 1], gap="large")
    contrib = CURRENT.contributions
    total_ret = contrib.attrs.get("portfolio_total", 0.0)
    with at1:
        st.plotly_chart(
            charts.contribution_waterfall(contrib["contribution"], total_ret,
                                          CURRENT.fund_labels(UNIVERSE)),
            use_container_width=True, config={"displayModeBar": False})
        T.caption(
            f"คำนวณจากน้ำหนักที่เปลี่ยนแปลงตามจริงของแต่ละกองทุน และรวมกันได้"
            f"เท่ากับผลตอบแทนรวมของพอร์ต {total_ret:+.1%} พอดี ในช่วง "
            f"{th.month(CURRENT.window[0])} – {th.month(CURRENT.window[1])}")
    with at2:
        by_class = attrib.contribution_by_group(
            CURRENT.panel, CURRENT.weights,
            {c: UNIVERSE[c].asset_class for c in CURRENT.weights}, rebalance)
        st.markdown("**แยกตาม Asset Class**")
        st.dataframe(
            T.style_frame(by_class.rename(columns={
                "weight": "น้ำหนัก", "contribution": "Contribution",
                "share": "สัดส่วนของผลตอบแทนรวม"}),
                percent_cols=["น้ำหนัก", "Contribution",
                              "สัดส่วนของผลตอบแทนรวม"],
                signed_cols=["Contribution"]),
            use_container_width=True)

    # ---- risk --------------------------------------------------------------
    T.rule("Risk Contribution")
    r1, r2 = st.columns([1.3, 1], gap="large")
    with r1:
        st.plotly_chart(
            charts.risk_vs_weight(CURRENT.risk_frame, CURRENT.fund_labels(UNIVERSE)),
            use_container_width=True, config={"displayModeBar": False})
        top = CURRENT.top_risk
        if top:
            T.caption(
                f"หากแท่งสีมิ้นต์ยาวกว่าแท่งสีเขียวน้ำทะเล กองทุนนั้นใช้ความเสี่ยง"
                f"มากกว่าเงินลงทุนที่ใส่ไป — <b>{fund_label(top[0])}</b> "
                f"คิดเป็น {CURRENT.weights[top[0]]:.0%} ของเงินลงทุน "
                f"แต่เป็น {top[1]:.0%} ของความเสี่ยง")
    with r2:
        if len(CURRENT.weights) > 1:
            st.plotly_chart(
                charts.correlation_heatmap(
                    risklib.correlation_matrix(CURRENT.panel),
                    {c: c.replace("K-", "") for c in CURRENT.panel.columns}),
                use_container_width=True, config={"displayModeBar": False})
        else:
            T.alert("พอร์ตที่มีกองทุนเดียวไม่มีโครงสร้างสหสัมพันธ์ภายในให้แสดง "
                    "ซึ่งตัวมันเองก็คือข้อสังเกตที่สำคัญ", "watch")

    obs = cautionlib.structural_observations(
        CURRENT.weights, UNIVERSE, CURRENT.effective_bets,
        CURRENT.diversification_ratio, CURRENT.top_risk)
    if obs:
        for o in obs:
            T.alert(o.message, "breach" if o.severity == "high" else "watch")

    # ---- stress ------------------------------------------------------------
    T.rule("Stress Test")
    s1, s2 = st.columns([1.4, 1], gap="large")
    with s1:
        prefs.remember("t1_proxy", True, cast=prefs.as_bool)
        allow_proxy = st.checkbox(
            "ใช้กองทุนตัวแทนกลุ่มสินทรัพย์ สำหรับกองที่ยังไม่จัดตั้งในช่วงนั้น",
            key="t1_proxy",
            help="กองทุน K-Asset ส่วนใหญ่จัดตั้งหลังปี 2018 หากไม่ใช้ตัวแทน "
                 "จะไม่สามารถย้อนทดสอบวิกฤตในอดีตกับกองเหล่านั้นได้เลย")
        stress_frame = cached_stress(weights_key(client.holdings), allow_proxy)
        if stress_frame.empty:
            T.alert("ไม่มีช่วงเหตุการณ์ในอดีตที่ทับซ้อนกับข้อมูลของพอร์ตนี้",
                    "watch")
        else:
            st.plotly_chart(charts.stress_bars(stress_frame),
                            use_container_width=True,
                            config={"displayModeBar": False})
            worst = stress_frame.iloc[0]
            T.caption(
                f"เหตุการณ์ที่แย่ที่สุด: <b>{stress_frame.index[0]}</b> ที่ "
                f"{worst['ผลตอบแทน']:.1%} หรือคิดเป็น "
                f"{thb(worst['ผลตอบแทน'] * aum)} จากมูลค่าพอร์ตปัจจุบัน "
                f"เหตุการณ์ที่มีสัดส่วนใช้ตัวแทนสูง อาศัยกองทุนตัวแทน"
                f"กลุ่มสินทรัพย์ ไม่ใช่ตัวกองทุนจริง")
            st.dataframe(
                T.style_frame(stress_frame,
                              percent_cols=["ผลตอบแทน", "Max Drawdown",
                                            "วันที่แย่ที่สุด",
                                            "สัดส่วนที่มีข้อมูลจริง",
                                            "สัดส่วนที่ใช้ตัวแทน"],
                              signed_cols=["ผลตอบแทน"]),
                use_container_width=True)
    with s2:
        st.markdown("**Factor Shock — ทดสอบช็อกรายปัจจัย**")
        shock_frame = cached_shocks(weights_key(client.holdings),
                                    pd.Timestamp("2018-01-01"))
        if not shock_frame.empty:
            display = shock_frame[["ผลกระทบต่อพอร์ต"]].copy()
            display["ผลกระทบเป็นมูลค่า"] = display["ผลกระทบต่อพอร์ต"] * aum
            st.dataframe(
                display.style.format({"ผลกระทบต่อพอร์ต": "{:.1%}",
                                      "ผลกระทบเป็นมูลค่า": "฿{:,.0f}"},
                                     na_rep="—"),
                use_container_width=True)
            T.caption(
                "ประมาณการตอบสนองของแต่ละกองทุนจากค่า beta ในอดีตต่อหุ้นไทย "
                "หุ้นต่างประเทศ ดอกเบี้ยไทย ทองคำ และน้ำมัน แล้วถ่วงน้ำหนัก "
                "ต่างจากการย้อนทดสอบเหตุการณ์จริง วิธีนี้ใช้ได้กับกองทุนที่มี"
                "ประวัติสั้น")

    # ---- monte carlo -------------------------------------------------------
    T.rule("Monte Carlo Simulation — การคาดการณ์อนาคต")
    # One generator, stated rather than chosen: block bootstrap is the only one
    # that keeps volatility clustering and trend intact, which is what makes
    # its drawdown numbers usable in front of a client.
    sim_method = mc.DEFAULT_METHOD
    student_df = 6.0                    # unused by block bootstrap

    m1, m2, m3 = st.columns(3)
    with m1:
        prefs.remember("t1_periods", 60, cast=int)
        n_periods = prefs.number_slider("จำนวนงวดข้างหน้า", "t1_periods",
                                        6, 120, 6)
    with m2:
        prefs.remember("t1_period", list(mc.PERIOD_LABELS)[0],
                       valid=list(mc.PERIOD_LABELS))
        period_unit = st.selectbox("หน่วยเวลา", list(mc.PERIOD_LABELS),
                                   key="t1_period")
    with m3:
        prefs.remember("t1_paths", 10000, cast=int,
                       valid=[1000, 2500, 5000, 10000, 20000])
        n_paths = st.select_slider("จำนวน Path",
                                   [1000, 2500, 5000, 10000, 20000],
                                   key="t1_paths")

    adv1, adv2 = st.columns([1, 1.3])
    with adv1:
        prefs.remember("t1_block", 6, cast=int)
        block_size = prefs.number_slider(
            "Block Length", "t1_block", 2, 24, 1, ratio=(2, 1),
            help="ความยาวช่วงที่สุ่มต่อเนื่องกัน ยิ่งยาว ยิ่งรักษาแนวโน้มและ"
                 "การกระจุกตัวของความผันผวนไว้มาก")
        prefs.remember("t1_muw", int(cma.DEFAULT_CMA_WEIGHT * 100), cast=int)
        cma_weight_1 = prefs.number_slider(
            "น้ำหนัก Kasikorn Asset CMA (%)", "t1_muw", 0, 100, 5, ratio=(2, 1),
            help="ส่วนที่เหลือใช้ผลตอบแทนย้อนหลังของกองทุนจริง "
                 "0% = ใช้ข้อมูลย้อนหลังล้วน · 100% = ใช้ CMA ล้วน "
                 "ความผันผวนและสหสัมพันธ์ใช้ข้อมูลจริงเสมอ") / 100.0
    with adv2:
        T.caption(
            f"<b>วิธีจำลอง · {sim_method}</b><br>"
            f"{mc.METHOD_NOTES.get(sim_method, '')}<br><br>"
            f"<b>ผลตอบแทนคาดหวัง · {cma.blend_label(cma_weight_1)}</b><br>"
            f"ผสมมุมมองระยะยาวของ Kasikorn Asset กับสิ่งที่กองทุน"
            f"ทำได้จริงในอดีต ปรับน้ำหนักได้ตามความเชื่อมั่นในแต่ละแหล่ง")

    hist_mu = (1.0 + CURRENT.panel).prod() ** (252 / len(CURRENT.panel)) - 1.0
    mu_forward = cma.mixed_mu(list(CURRENT.weights), UNIVERSE, hist_mu,
                              cma_weight_1)
    mu_items = tuple(sorted(mu_forward.items()))

    try:
        sim = cached_simulation(
            weights_key(CURRENT.weights), sim_method, n_paths, n_periods,
            period_unit, rebalance != "none", 1.0, student_df, block_size,
            mu_items, CURRENT.window[0])
    except Exception as exc:
        sim = None
        st.error(f"การจำลองล้มเหลว: {exc}")

    if sim is not None:
        ts = sim.terminal_stats(aum)
        f1, f2 = st.columns([1.7, 1], gap="large")
        with f1:
            st.plotly_chart(
                charts.fan_chart(sim.percentiles, period_unit, aum),
                use_container_width=True, config={"displayModeBar": False})
        with f2:
            st.plotly_chart(
                charts.terminal_distribution(sim.terminal, aum),
                use_container_width=True, config={"displayModeBar": False})
            T.caption(
                f"{n_paths:,} เส้นทาง · ระยะเวลา {sim.horizon_years:.1f} ปี · "
                f"แนวโน้มผลตอบแทนจาก {cma.blend_label(cma_weight_1)}<br>"
                f"ความผันผวนที่จำลองได้ {pct(sim.diagnostics['sim_vol_ann'])} "
                f"เทียบกับที่เกิดขึ้นจริง "
                f"{pct(sim.diagnostics['hist_vol_ann'])}")

        T.metric_row([
            T.metric_card("กรณีกลาง (Median)", thb(ts["median"]),
                          note=f"{pct(ts['median_cagr'])} ต่อปี"),
            T.metric_card("กรณีแย่ (Percentile 5)", thb(ts["p5"]), tone="neg",
                          note=f"{pct(ts['cagr_p5'])} ต่อปี"),
            T.metric_card("กรณีดี (Percentile 95)", thb(ts["p95"]), tone="pos",
                          note=f"{pct(ts['cagr_p95'])} ต่อปี"),
            T.metric_card("โอกาสขาดทุน", pct(ts["prob_loss"], 0),
                          tone="neg" if ts["prob_loss"] > 0.25 else "",
                          note="มูลค่าปลายทางต่ำกว่าวันนี้"),
            T.metric_card("Max Drawdown (ทั่วไป)", pct(ts["median_max_dd"]),
                          tone="neg",
                          note=f"กรณี 1 ใน 100: {pct(ts['worst_max_dd'])}"),
        ])

    # ---- PDF export --------------------------------------------------------
    def _current_sections(picked: List[str]) -> bytes:
        want = set(picked)
        out: List[reportlib.Section] = []

        if "profile" in want:
            out.append(reportlib.Section(
                "ข้อมูลลูกค้าและระดับความเสี่ยง",
                bullets=[
                    f"<b>{client.name}</b> ({client.id}) · {client.persona}",
                    f"ระดับความเสี่ยงที่รับได้: <b>ระดับ {profile.level}</b> — "
                    f"{profile.description}",
                    f"วัตถุประสงค์: {client.objective} · "
                    f"ระยะเวลาลงทุน {client.horizon_years} ปี",
                    f"มูลค่าพอร์ต: <b>{thb(aum)}</b> · "
                    f"ถือ {len(CURRENT.weights)} กองทุน",
                    f"ช่วงข้อมูลที่ใช้: {th.date(win_from)} – {th.date(win_to)} "
                    f"({len(CURRENT.returns):,} วันทำการ)",
                ]))

        if "summary" in want:
            st_ = CURRENT.stats
            out.append(reportlib.Section(
                "สรุปสถิติพอร์ต",
                lead=f"คำนวณจากช่วง {th.date(win_from)} – {th.date(win_to)} "
                     f"โดยตั้งสมมติฐาน Rebalancing {th.REBALANCE[rebalance]}",
                columns=["รายการ", "ค่า"],
                rows=[["ผลตอบแทนสะสม", pct(st_.total_return)],
                      ["ผลตอบแทนต่อปี (CAGR)", pct(st_.cagr)],
                      ["ความผันผวนต่อปี", pct(st_.volatility)],
                      ["Sharpe Ratio", num(st_.sharpe)],
                      ["Sortino Ratio", num(st_.sortino)],
                      ["Max Drawdown", pct(st_.max_drawdown)],
                      ["VaR (1 วัน, 95%)", pct(st_.var_95)],
                      ["CVaR (1 วัน, 95%)", pct(st_.cvar_95)],
                      ["Effective Positions", num(CURRENT.effective_bets, 1)],
                      ["Diversification Ratio",
                       num(CURRENT.diversification_ratio)]],
                align_right=[1]))

        if "holdings" in want:
            rows = []
            for code, w in sorted(CURRENT.weights.items(), key=lambda kv: -kv[1]):
                fund = UNIVERSE[code]
                rc = (CURRENT.risk_frame.loc[code]
                      if code in CURRENT.risk_frame.index else None)
                rows.append([code, fund.name, fund.asset_class,
                             str(fund.risk_level), pct(w), thb(w * aum),
                             pct(float(rc["risk_share"])) if rc is not None else "—"])
            out.append(reportlib.Section(
                "รายการกองทุนที่ถืออยู่",
                lead="Risk Contribution คือสัดส่วนความเสี่ยงที่กองทุนนั้น"
                     "ใส่เข้ามาในพอร์ต ซึ่งมักไม่เท่ากับน้ำหนักเงินลงทุน",
                columns=["กองทุน", "ชื่อกองทุน", "Asset Class", "ระดับ",
                         "น้ำหนัก", "มูลค่า", "Risk Contribution"],
                rows=rows, align_right=[3, 4, 5, 6]))

        if "allocation" in want:
            rows = [[k, pct(v), thb(v * aum)]
                    for k, v in sorted(CURRENT.class_weights.items(),
                                       key=lambda kv: -kv[1])]
            out.append(reportlib.Section(
                "สัดส่วนตาม Asset Class", columns=["Asset Class", "น้ำหนัก",
                                                   "มูลค่า"],
                rows=rows, align_right=[1, 2]))
            band_rows = []
            for bucket, band in profile.bands.items():
                held = CURRENT.bucket_exposure.get(bucket, 0.0)
                inside = band[0] - 1e-9 <= held <= band[1] + 1e-9
                band_rows.append([bucket, pct(held),
                                  f"{band[0]:.0%} – {band[1]:.0%}",
                                  "อยู่ในกรอบ" if inside else "นอกกรอบ"])
            out.append(reportlib.Section(
                "สัดส่วนเทียบกรอบความเหมาะสม",
                lead=f"กรอบตามแบบประเมินความเสี่ยงระดับ {profile.level}",
                columns=["ประเภทสินทรัพย์", "ถืออยู่", "กรอบที่แนะนำ", "สถานะ"],
                rows=band_rows, align_right=[1, 2]))

        if "annual" in want:
            annual_pdf = metrics.annual_return_table(CURRENT.returns)
            if not annual_pdf.empty:
                out.append(reportlib.Section(
                    "ผลตอบแทนรายปี",
                    lead="ปีแรกและปีสุดท้ายอาจไม่เต็มปี เพราะถูกตัดด้วยช่วง"
                         "วันที่ที่เลือกไว้",
                    columns=["ปี"] + list(annual_pdf.columns),
                    rows=_frame_rows(annual_pdf, {
                        "ผลตอบแทน": _pct_or_dash,
                        "ความผันผวน": _pct_or_dash,
                        "Max Drawdown": _pct_or_dash,
                        "เดือนที่ดีที่สุด": _pct_or_dash,
                        "เดือนที่แย่ที่สุด": _pct_or_dash,
                        "วันทำการ": lambda v: f"{int(v):,}"}),
                    align_right=[1, 2, 3, 4, 5, 6]))

        if "attribution" in want:
            rows = [[fund_label(code), pct(r["weight"]),
                     pct(r["total_return"]), pct(r["contribution"]),
                     pct(r["share"])]
                    for code, r in CURRENT.contributions.iterrows()]
            out.append(reportlib.Section(
                "Return Attribution",
                lead=f"ผลตอบแทนรวมของพอร์ต {pct(total_ret)} "
                     f"เท่ากับผลรวมของคอลัมน์ส่วนที่สร้างให้พอร์ต พอดี",
                columns=["กองทุน", "น้ำหนักเริ่มต้น", "ผลตอบแทนของกองทุน",
                         "ส่วนที่สร้างให้พอร์ต", "สัดส่วนของผลตอบแทนรวม"],
                rows=rows, align_right=[1, 2, 3, 4]))

        if "risk" in want and not CURRENT.risk_frame.empty:
            rows = [[fund_label(code), pct(r["weight"]),
                     pct(r["risk_share"]),
                     f"{r['risk_share'] / r['weight']:.2f}x"
                     if r["weight"] > 0 else "—"]
                    for code, r in CURRENT.risk_frame.sort_values(
                        "risk_share", ascending=False).iterrows()]
            out.append(reportlib.Section(
                "Risk Contribution",
                lead="ตัวคูณมากกว่า 1 เท่า แปลว่ากองทุนนั้นใช้ความเสี่ยง"
                     "มากกว่าเงินลงทุนที่ใส่ไป",
                columns=["กองทุน", "น้ำหนักเงินลงทุน", "น้ำหนักความเสี่ยง",
                         "ตัวคูณ"],
                rows=rows, align_right=[1, 2, 3]))

        if "suitability" in want:
            if findings:
                rows = [[th.kind(f.kind), th.finding_severity(f.severity), f.detail]
                        for f in findings]
            else:
                rows = [["—", "ผ่านเกณฑ์", "ไม่พบข้อที่ต้องแก้ไข"]]
            out.append(reportlib.Section(
                "ผลตรวจความเหมาะสม",
                lead=f"สถานะโดยรวม: <b>{th.status(status)}</b>",
                columns=["ประเภท", "ระดับ", "รายละเอียด"], rows=rows))

        if "stress" in want:
            frame = cached_stress(weights_key(client.holdings), allow_proxy)
            if not frame.empty:
                rows = [[idx, r["ช่วงเวลา"], pct(r["ผลตอบแทน"]),
                         pct(r["Max Drawdown"]), thb(r["ผลตอบแทน"] * aum)]
                        for idx, r in frame.iterrows()]
                out.append(reportlib.Section(
                    "Stress Test — ถ้าเหตุการณ์ในอดีตเกิดขึ้นอีก",
                    lead="คำนวณโดยนำผลตอบแทนจริงของแต่ละกองทุนในช่วง"
                         "เหตุการณ์นั้นมาถ่วงน้ำหนักตามพอร์ตปัจจุบัน",
                    columns=["เหตุการณ์", "ช่วงเวลา", "ผลกระทบต่อพอร์ต",
                             "Max Drawdown", "คิดเป็นเงิน"],
                    rows=rows, align_right=[2, 3, 4]))

        if "montecarlo" in want and sim is not None:
            out.append(reportlib.Section(
                "Monte Carlo Simulation",
                lead=f"{n_paths:,} เส้นทาง · ระยะเวลา {sim.horizon_years:.1f} ปี "
                     f"· วิธี {sim_method} · แนวโน้มผลตอบแทนจาก "
                     f"{cma.blend_label(cma_weight_1)}",
                columns=["กรณี", "มูลค่าปลายทาง", "ผลตอบแทนต่อปี"],
                rows=[["กรณีกลาง (Median)", thb(ts["median"]),
                       pct(ts["median_cagr"])],
                      ["กรณีแย่ (Percentile 5)", thb(ts["p5"]),
                       pct(ts["cagr_p5"])],
                      ["กรณีดี (Percentile 95)", thb(ts["p95"]),
                       pct(ts["cagr_p95"])],
                      ["โอกาสขาดทุน", pct(ts["prob_loss"], 0), "—"],
                      ["Max Drawdown (ทั่วไป)", pct(ts["median_max_dd"]), "—"],
                      ["Max Drawdown (1 ใน 100)", pct(ts["worst_max_dd"]), "—"]],
                align_right=[1, 2]))

        if "cautions" in want and CAUTIONS:
            out.append(reportlib.Section(
                "ข้อควรระวังจากภาวะตลาด",
                lead="ประเด็นตลาดที่กระทบสัดส่วนสำคัญของพอร์ตนี้โดยตรง",
                columns=["ระดับ", "ประเด็น", "ผลต่อพอร์ตนี้", "สัดส่วนที่กระทบ"],
                rows=[[th.severity(c.severity), c.title,
                       cautionlib.theme_caution(c), pct(c.exposure, 0)]
                      for c in CAUTIONS],
                align_right=[3]))

        if "notes" in want:
            body = (st.session_state.get(f"note_{client.id}") or "").strip()
            out.append(reportlib.Section(
                "บันทึกของผู้แนะนำการลงทุน",
                bullets=[line for line in body.splitlines() if line.strip()]
                        or ["ยังไม่มีบันทึกสำหรับลูกค้ารายนี้"]))

        # Charts last, and each one rasterised from the very figure the tab
        # renders — the document cannot drift from the screen because there is
        # only one chart builder.
        def _fig(figure, title, lead="", width_mm=178.0, height=460):
            png = reportlib.figure_png(figure, height=height)
            if png:
                out.append(reportlib.Section(title, lead=lead, image=png,
                                             image_width_mm=width_mm))

        if "chart_growth" in want:
            _fig(charts.growth_chart(series_map),
                 "ผลการดำเนินงานย้อนหลัง",
                 f"สมมติลงทุน {charts.baht_label(charts.GROWTH_BASE)} "
                 f"ณ วันเริ่มต้น · {th.date(win_from)} – {th.date(win_to)}")
        if "chart_drawdown" in want:
            _fig(charts.drawdown_chart(series_map), "Drawdown",
                 "ระยะที่พอร์ตอยู่ต่ำกว่าจุดสูงสุดเดิม และใช้เวลานานเท่าไรกว่า"
                 "จะกลับมา", height=360)
        if "chart_calendar" in want:
            _fig(charts.calendar_bars(
                     metrics.calendar_year_returns(CURRENT.returns)),
                 "ผลตอบแทนรายปี", height=380)
        if "chart_allocation" in want:
            _fig(charts.allocation_donut(
                     CURRENT.weights, CURRENT.fund_labels(UNIVERSE),
                     centre=f"{len(CURRENT.weights)} กองทุน"),
                 "สัดส่วนรายกองทุน", width_mm=130, height=440)
            _fig(charts.allocation_donut(
                     CURRENT.class_weights,
                     {k: k for k in CURRENT.class_weights}, centre="กลุ่ม"),
                 "สัดส่วนตาม Asset Class", width_mm=130, height=440)
        if "chart_bands" in want:
            _fig(charts.band_compliance(CURRENT.bucket_exposure, profile.bands),
                 "สัดส่วนเทียบกรอบความเหมาะสม",
                 f"กรอบตามแบบประเมินความเสี่ยงระดับ {profile.level}",
                 height=380)
        if "chart_attribution" in want:
            _fig(charts.contribution_waterfall(
                     contrib["contribution"], total_ret,
                     CURRENT.fund_labels(UNIVERSE)),
                 "Return Attribution",
                 f"ผลตอบแทนรวมของพอร์ต {pct(total_ret)}", height=430)
        if "chart_risk" in want and not CURRENT.risk_frame.empty:
            _fig(charts.risk_vs_weight(CURRENT.risk_frame,
                                       CURRENT.fund_labels(UNIVERSE)),
                 "Risk Contribution",
                 "แท่งบนคือน้ำหนักเงินลงทุน แท่งล่างคือน้ำหนักความเสี่ยง",
                 height=420)
        if "chart_correlation" in want and len(CURRENT.weights) > 1:
            _fig(charts.correlation_heatmap(
                     risklib.correlation_matrix(CURRENT.panel),
                     {c: c.replace("K-", "") for c in CURRENT.panel.columns}),
                 "สหสัมพันธ์ระหว่างกองทุนในพอร์ต",
                 "ยิ่งเข้มไปทางเขียว ยิ่งเคลื่อนไหวไปทางเดียวกัน "
                 "การกระจายความเสี่ยงจริงอยู่ที่ตัวเลขเหล่านี้",
                 width_mm=140, height=430)
        if "chart_stress" in want:
            _sf = cached_stress(weights_key(client.holdings), allow_proxy)
            if not _sf.empty:
                _fig(charts.stress_bars(_sf), "Stress Test",
                     "ผลกระทบต่อพอร์ตถ้าเหตุการณ์ในอดีตเกิดขึ้นอีก", height=520)
        if "chart_montecarlo" in want and sim is not None:
            _fig(charts.fan_chart(sim.percentiles, period_unit, aum),
                 "Monte Carlo Simulation",
                 f"{n_paths:,} เส้นทาง · {sim_method} · "
                 f"ระยะเวลา {sim.horizon_years:.1f} ปี", height=440)

        return reportlib.build_pdf(
            f"พอร์ตปัจจุบัน — {client.name}",
            f"{client.id} · ระดับความเสี่ยง {profile.level} · "
            f"ข้อมูล {th.date(win_from)} – {th.date(win_to)}",
            out,
            footer=f"K-ADVISOR · จัดทำ {datetime.now():%d/%m/%Y %H:%M}",
            disclaimer=DISCLAIMER)

    export_panel("t1", reportlib.CURRENT_BLOCKS, reportlib.DEFAULT_CURRENT,
                 _current_sections, "current-portfolio")


# --------------------------------------------------------------------------- #
# TAB 2 — Suggested portfolio
# --------------------------------------------------------------------------- #
with tab2:
    prefs.remember("t2_mode", MODE_SOLVE, valid=BUILD_MODES)
    build_mode = st.radio("วิธีจัดพอร์ต", BUILD_MODES, key="t2_mode",
                          horizontal=True,
                          help="Optimizer หาน้ำหนักที่ดีที่สุดให้ตาม Objective "
                               "ส่วนกำหนดเองให้ผู้แนะนำเลือกกองทุนและน้ำหนัก"
                               "ด้วยตัวเอง แล้วระบบจะวัดผลด้วยวิธีเดียวกัน")
    custom_mode = build_mode == MODE_CUSTOM

    T.rule("Mandate — เงื่อนไขการจัดพอร์ต")
    o1, o2, o3, o4 = st.columns([1.4, 1, 1, 1])
    with o1:
        prefs.remember("t2_obj", opt.OBJECTIVES[1], valid=opt.OBJECTIVES)
        objective = st.selectbox("Objective", opt.OBJECTIVES, key="t2_obj",
                                 disabled=custom_mode,
                                 help="ใช้กับโหมด Optimizer เท่านั้น "
                                      "ในโหมดกำหนดเองจะแสดงไว้เพื่อเปรียบเทียบ"
                                 if custom_mode else None)
    with o2:
        target_value = None
        if objective == opt.TARGET_RETURN:
            prefs.remember("t2_tgt_r", 6.0, cast=float)
            target_value = prefs.number_slider(
                "Target Return (% ต่อปี)", "t2_tgt_r", 1.0, 15.0, 0.5,
                fmt="%.1f", ratio=(2, 1)) / 100.0
        elif objective == opt.TARGET_VOL:
            prefs.remember("t2_tgt_v", 8.0, cast=float)
            target_value = prefs.number_slider(
                "Target Volatility (% ต่อปี)", "t2_tgt_v", 1.0, 25.0, 0.5,
                fmt="%.1f", ratio=(2, 1)) / 100.0
        else:
            st.markdown(
                f'<div class="klabel" style="margin-top:0.4rem">คำอธิบาย Objective</div>'
                f'<div class="ksmall">{opt.OBJECTIVE_NOTES[objective][:90]}…</div>',
                unsafe_allow_html=True)
    with o3:
        prefs.remember("t2_maxw", 35, cast=int)
        max_weight = prefs.number_slider("น้ำหนักสูงสุดต่อกองทุน (%)", "t2_maxw",
                                         10, 100, 5, ratio=(2, 1)) / 100.0
        prefs.remember("t2_maxn", 8, cast=int)
        max_funds = prefs.number_slider(
            "จำนวนกองทุนเป้าหมาย", "t2_maxn", 3, 15, 1, ratio=(2, 1),
            help="อาจมากกว่านี้ได้ หากกรอบสัดส่วนความเหมาะสม"
                 "จำเป็นต้องใช้กองทุนมากกว่า")
    with o4:
        prefs.remember("t2_sat", 35, cast=int)
        max_sat = prefs.number_slider(
            "Satellite สูงสุด (%)", "t2_sat", 0, 60, 5, ratio=(2, 1),
            help="จำกัดรวมกันของกองทุน Thematic, Sector, "
                 "รายประเทศ และสินทรัพย์ทางเลือก") / 100.0
        prefs.remember("t2_muw", int(cma.DEFAULT_CMA_WEIGHT * 100), cast=int)
        cma_weight_2 = prefs.number_slider(
            "น้ำหนัก Kasikorn Asset CMA (%)", "t2_muw", 0, 100, 5, ratio=(2, 1),
            help="ผลตอบแทนคาดหวังที่ป้อนให้ Optimizer ผสมระหว่างมุมมองระยะยาว"
                 "ของ Kasikorn Asset กับผลตอบแทนย้อนหลังของกองทุนจริง "
                 "ส่วนความผันผวนและสหสัมพันธ์ใช้ข้อมูลย้อนหลังเสมอ") / 100.0
        mu_source_2 = cma.blend_label(cma_weight_2)

    with st.expander("ตั้งค่าขั้นสูง — Fund Universe และเพดานระดับความเสี่ยง"):
        e1, e2, e3 = st.columns(3)
        with e1:
            prefs.remember("t2_above", True, cast=prefs.as_bool)
            allow_above = st.toggle(
                "อนุญาตผลิตภัณฑ์ที่เสี่ยงสูงกว่าระดับของลูกค้า",
                key="t2_above",
                help="แนวปฏิบัติในไทยอนุญาตได้เมื่อลูกค้าลงนามรับทราบความเสี่ยง "
                     "และมักเป็นทางเดียวที่จะเติมกรอบสัดส่วนให้ครบ เพราะกองทุน"
                     "สินทรัพย์ทางเลือกทุกกองอยู่ที่ระดับ 8")
        with e2:
            prefs.remember("t2_abovebud", 20, cast=int)
            above_budget = prefs.number_slider(
                "วงเงินที่ลูกค้ารับทราบความเสี่ยง (%)", "t2_abovebud",
                0, 40, 5, disabled=not allow_above, ratio=(2, 1)) / 100.0
        with e3:
            prefs.remember("t2_hist", "5 ปี",
                           valid=["2 ปี", "3 ปี", "5 ปี", "7 ปี"])
            min_history = st.select_slider(
                "อายุกองทุนขั้นต่ำ", ["2 ปี", "3 ปี", "5 ปี", "7 ปี"],
                key="t2_hist",
                help="กองทุนที่อายุน้อยกว่านี้จะถูกคัดออก เพราะการใส่กองทุนใหม่"
                     "เข้าไปจะตัดช่วงข้อมูลร่วมของกองอื่นทั้งหมดให้สั้นลง "
                     "และทำให้เมทริกซ์ความแปรปรวนร่วมกลายเป็นสัญญาณรบกวน")

    hist_years = {"2 ปี": 2, "3 ปี": 3, "5 ปี": 5, "7 ปี": 7}[min_history]
    since = max(pd.Timestamp(FUND_DATA.end) - pd.DateOffset(years=hist_years),
                pd.Timestamp(FUND_DATA.start))

    eligible = cached_eligible(profile.level, since, allow_above, 250)
    above_codes = tuple(sorted(c for c in eligible
                               if UNIVERSE[c].risk_level > profile.level))
    if not allow_above:
        above_codes = tuple()

    band_res = opt.resolve_bands(eligible, UNIVERSE, profile.bands, max_weight,
                                 frozenset(above_codes),
                                 above_budget if allow_above else 0.0)

    if band_res.adjusted:
        T.alert(
            f"<b>ผ่อนคลายกรอบ Asset Allocation เพื่อให้จัดพอร์ตได้</b> — "
            f"{band_res.reason} " + "; ".join(band_res.relaxations) + " "
            f"การผ่อนคลายเลือกให้เพิ่มความเสี่ยงน้อยที่สุด "
            f"คือยอมถือเงินสดมากขึ้นดีกว่าถือหุ้นมากขึ้น", "watch")
    elif not band_res.feasible_as_printed:
        T.alert(f"<b>ไม่สามารถจัดพอร์ตได้</b> — {band_res.reason}", "breach")

    panel_opt = FUND_DATA.slice(since, None, eligible).dropna(how="any")
    if panel_opt.empty or panel_opt.shape[1] < 2:
        st.error("ข้อมูลย้อนหลังที่ทับซ้อนกันไม่เพียงพอสำหรับการตั้งค่านี้ "
                 "กรุณาลดอายุกองทุนขั้นต่ำ")
        st.stop()

    hist_mu_2 = (1.0 + panel_opt).prod() ** (252 / len(panel_opt)) - 1.0
    mu_opt = cma.mixed_mu(list(panel_opt.columns), UNIVERSE, hist_mu_2,
                          cma_weight_2)

    T.caption(
        f"Fund Universe ที่ลงทุนได้: <b>{len(eligible)} กองทุน</b> "
        f"(มี {len(above_codes)} กองที่เสี่ยงสูงกว่าระดับของลูกค้า "
        f"จำกัดไม่เกิน {above_budget:.0%}) · ข้อมูลร่วม "
        f"{th.month(panel_opt.index[0])} – {th.month(panel_opt.index[-1])} "
        f"({len(panel_opt):,} ข้อมูล) · ผลตอบแทนคาดหวังจาก {mu_source_2}")

    with st.spinner("กำลังคำนวณพอร์ตที่เหมาะสม…"):
        solved = cached_optimise(
            tuple(panel_opt.columns), objective, tuple(sorted(band_res.bands.items())),
            max_weight, max_sat, max_funds, 0.03, rf_rate, above_codes,
            above_budget if allow_above else 0.0,
            tuple(sorted(mu_opt.items())), target_value, since, COV_METHOD)

    if custom_mode:
        solution = custom_portfolio_editor(
            eligible, panel_opt, mu_opt, client, solved, max_weight, max_sat,
            max_funds, above_codes, above_budget if allow_above else 0.0,
            rf_rate, aum)
        if solution is None:
            st.stop()
    else:
        solution = solved

    if not solution.weights:
        st.error(f"ไม่สามารถจัดพอร์ตได้: {solution.message}")
        st.stop()
    if solution.status == "warning" and solution.message:
        T.alert(f"<b>หมายเหตุ</b> — {solution.message}", "watch")

    PROPOSED = cached_analysis(weights_key(solution.weights), analysis_start,
                               analysis_end,
                               rebalance, rf_rate, COV_METHOD, "Proposed")
    # Funds above the client's level are in the proposal only because the RM
    # enabled the acknowledgement budget, so they are a disclosed exception
    # rather than an undisclosed breach. The band edges are where an optimiser
    # is *supposed* to sit, so the proximity warning is suppressed here.
    acknowledged = {c for c in solution.weights
                    if UNIVERSE[c].risk_level > profile.level} if allow_above else set()
    prop_findings = clientlib.check_suitability(
        solution.weights, profile, UNIVERSE, watch_margin=0.0,
        acknowledged=acknowledged)
    prop_status = clientlib.suitability_status(prop_findings)
    prop_cautions = cautionlib.build_cautions(solution.weights, UNIVERSE, MARKET,
                                              limit=6)

    # ---- headline ----------------------------------------------------------
    comp = eng.comparison_table(CURRENT, PROPOSED, rf_rate)
    flags = eng.improvement_flags(comp) if not comp.empty else {}
    turn = opt.turnover(client.holdings, solution.weights)

    _split = opt.core_satellite_split(solution.weights, UNIVERSE)
    T.metric_row([
        T.metric_card("Expected Return", pct(solution.expected_return),
                      note=f"จาก {cma.blend_short(cma_weight_2)}"),
        T.metric_card("Expected Volatility", pct(solution.volatility),
                      note=f"Sharpe {num(solution.sharpe)}"),
        T.metric_card("Max Drawdown ที่เคยเกิด",
                      pct(solution.diagnostics.get("max_drawdown_sample", np.nan)),
                      tone="neg", note="วัดจากจุดสูงสุดถึงต่ำสุดในข้อมูล"),
        T.metric_card("จำนวนกองทุน", str(len(solution.weights)),
                      note=f"Core {_split['Core']:.0%} · "
                           f"Satellite {_split['Satellite']:.0%}"),
        T.metric_card("Turnover", pct(turn, 0),
                      note=f"ซื้อขายด้านเดียว {thb(turn * aum)}"),
        T.metric_card("ความเหมาะสม", th.status(prop_status),
                      tone="pos" if prop_status == "Compliant" else "neg",
                      note=f"ก่อนหน้านี้: {th.status(status)}"),
    ])

    if prop_findings:
        for f in prop_findings[:4]:
            T.alert(f"<b>{th.kind(f.kind)}</b> — {f.detail}",
                    "breach" if f.severity == "breach" else "watch")
    else:
        T.alert("<b>ผ่านเกณฑ์ทั้งหมด</b> ทุกกองทุนที่แนะนำอยู่ในระดับความเสี่ยง"
                "ที่ลูกค้ารับได้ และสัดส่วนการลงทุนอยู่ในกรอบสำหรับผู้ลงทุน"
                "ประเภทนี้", "ok")

    render_caution_ribbon(prop_cautions,
                          "ข้อควรระวังจากภาวะตลาดสำหรับพอร์ตที่แนะนำ")

    # ---- allocation --------------------------------------------------------
    T.rule("Asset Allocation ที่แนะนำ")
    pa1, pa2, pa3 = st.columns([1, 1, 1], gap="large")
    with pa1:
        st.markdown("**แยกตามกองทุน**")
        st.plotly_chart(
            charts.allocation_donut(solution.weights,
                                    {c: fund_label(c) for c in solution.weights},
                                    centre=f"{len(solution.weights)} กองทุน"),
            use_container_width=True, config={"displayModeBar": False})
    with pa2:
        st.markdown("**แยกตาม Asset Class**")
        st.plotly_chart(
            charts.allocation_donut(PROPOSED.class_weights,
                                    {k: k for k in PROPOSED.class_weights},
                                    colors=T.CLASS_COLORS, centre="กลุ่ม"),
            use_container_width=True, config={"displayModeBar": False})
    with pa3:
        st.markdown("**Core / Satellite**")
        split = opt.core_satellite_split(solution.weights, UNIVERSE)
        st.plotly_chart(charts.core_satellite_bar(split),
                        use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(
            charts.band_compliance(PROPOSED.bucket_exposure, profile.bands,
                                   height=190),
            use_container_width=True, config={"displayModeBar": False})
        T.caption("Core คือกองทุนที่กระจายกว้าง เป็นกลยุทธ์ระยะยาว "
                  "ส่วน Satellite คือกองทุน Thematic รายประเทศ หรือสินทรัพย์"
                  "ทางเลือก ซึ่งใช้แสดงมุมมอง และควรกำหนดขนาดโดยเผื่อไว้ว่า"
                  "มุมมองอาจผิด")

    prop_rows = []
    for code, w in sorted(solution.weights.items(), key=lambda kv: -kv[1]):
        fund = UNIVERSE[code]
        prop_rows.append({
            "กองทุน": code, "ชื่อกองทุน": fund.name,
            "Asset Class": fund.asset_class,
            "ภูมิภาค": fund.region, "ระดับ": fund.risk_level,
            "Core / Satellite": th.role(fund.role),
            "น้ำหนัก": w * 100.0, "มูลค่า": f"฿{w * aum:,.0f}",
            "สูงกว่าระดับลูกค้า": "ใช่" if fund.risk_level > profile.level else "",
        })
    st.dataframe(
        pd.DataFrame(prop_rows).set_index("กองทุน"), use_container_width=True,
        column_config={
            "น้ำหนัก": st.column_config.ProgressColumn(
                "น้ำหนัก", format="%.1f%%", min_value=0.0, max_value=100.0),
            "มูลค่า": st.column_config.TextColumn("มูลค่า (บาท)"),
            "ระดับ": st.column_config.NumberColumn("ระดับความเสี่ยง"),
        })

    if any(UNIVERSE[c].risk_level > profile.level for c in solution.weights):
        flagged = [f"{fund_label(c)} (ระดับ {UNIVERSE[c].risk_level}, {w:.0%})"
                   for c, w in solution.weights.items()
                   if UNIVERSE[c].risk_level > profile.level]
        T.alert(
            f"<b>ต้องให้ลูกค้าลงนามรับทราบความเสี่ยง</b> — "
            f"{'; '.join(flagged)} สูงกว่าเพดานระดับ {profile.level} "
            f"ของลูกค้ารายนี้ ที่รวมไว้เพราะไม่เช่นนั้นจะเติมกรอบสัดส่วน"
            f"สำหรับผู้ลงทุนประเภทนี้ให้ครบไม่ได้", "watch")

    # ---- comparison --------------------------------------------------------
    T.rule("เปรียบเทียบพอร์ตปัจจุบันกับพอร์ตที่แนะนำ")
    cc1, cc2 = st.columns([1.25, 1], gap="large")
    with cc1:
        st.plotly_chart(
            charts.weight_comparison(
                client.holdings, solution.weights,
                {c: fund_label(c) for c in set(client.holdings) | set(solution.weights)}),
            use_container_width=True, config={"displayModeBar": False})
    with cc2:
        if not comp.empty:
            show = comp[["ปัจจุบัน", "เสนอใหม่", "เปลี่ยนแปลง"]].copy()
            is_pct = comp["fmt"] == "pct"

            def _fmt(value, metric, signed=False):
                if pd.isna(value):
                    return "—"
                if is_pct.get(metric, False):
                    return f"{value:+.2%}" if signed else f"{value:.2%}"
                return f"{value:+,.2f}" if signed else f"{value:,.2f}"

            display = pd.DataFrame({
                "ปัจจุบัน": [_fmt(v, m) for m, v in show["ปัจจุบัน"].items()],
                "เสนอใหม่": [_fmt(v, m) for m, v in show["เสนอใหม่"].items()],
                "เปลี่ยนแปลง": [_fmt(v, m, True)
                               for m, v in show["เปลี่ยนแปลง"].items()],
            }, index=show.index)
            st.dataframe(
                display.style.apply(
                    lambda col: [
                        f"color: {T.MINT if flags.get(i, False) else T.CORAL}"
                        for i in col.index], subset=["เปลี่ยนแปลง"]),
                use_container_width=True, height=460)
            win = comp.attrs.get("window")
            if win:
                T.caption(f"วัดบนช่วง {th.month(win[0])} – {th.month(win[1])} "
                          f"ที่ทั้งสองพอร์ตมีข้อมูลร่วมกัน สีมิ้นต์หมายถึง"
                          f"พอร์ตที่แนะนำดีกว่าในตัวชี้วัดนั้น")

    g1, g2 = st.columns(2, gap="large")
    a_ret, b_ret = eng.align_windows(CURRENT, PROPOSED)
    with g1:
        st.plotly_chart(
            charts.growth_chart({"ปัจจุบัน": a_ret, "เสนอใหม่": b_ret}),
            use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(
            charts.drawdown_chart({"ปัจจุบัน": a_ret, "เสนอใหม่": b_ret},
                                  height=340),
            use_container_width=True, config={"displayModeBar": False})

    # ---- frontier ----------------------------------------------------------
    T.rule("Efficient Frontier ภายใต้เงื่อนไขของลูกค้ารายนี้")
    fr1, fr2 = st.columns([1.4, 1], gap="large")
    with fr1:
        with st.spinner("กำลังคำนวณ Efficient Frontier…"):
            frontier = cached_frontier(
                tuple(panel_opt.columns), tuple(sorted(band_res.bands.items())),
                max_weight, max_sat, rf_rate, above_codes,
                above_budget if allow_above else 0.0,
                tuple(sorted(mu_opt.items())), since, COV_METHOD)
        cur_mu = float(sum(mu_opt.get(c, 0.0) * w for c, w in CURRENT.weights.items()))
        points = [
            ("พอร์ตที่แนะนำ", solution.volatility, solution.expected_return,
             T.MINT, "diamond"),
            ("พอร์ตปัจจุบัน", CURRENT.stats.volatility, cur_mu, T.CORAL, "x"),
        ]
        st.plotly_chart(charts.frontier_chart(frontier, points),
                        use_container_width=True, config={"displayModeBar": False})
        T.caption(
            "เส้นนี้คำนวณ<i>ภายใน</i>กรอบความเหมาะสมของลูกค้ารายนี้ "
            "ไม่ใช่บน Fund Universe แบบไม่มีเงื่อนไข ดังนั้นทุกจุดบนเส้นคือ"
            "พอร์ตที่ลูกค้าลงทุนได้จริง")
    with fr2:
        st.markdown("**เปรียบเทียบ Objective อื่น**")
        rows = []
        for obj in opt.OBJECTIVES:
            try:
                alt = cached_optimise(
                    tuple(panel_opt.columns), obj,
                    tuple(sorted(band_res.bands.items())), max_weight, max_sat,
                    max_funds, 0.03, rf_rate, above_codes,
                    above_budget if allow_above else 0.0,
                    tuple(sorted(mu_opt.items())), None, since, COV_METHOD)
            except Exception:
                continue
            if alt.weights:
                rows.append({
                    "Objective": obj.split(" (")[0],
                    "Return": alt.expected_return,
                    "Volatility": alt.volatility,
                    "Sharpe": alt.sharpe,
                    "Max DD": alt.diagnostics.get("max_drawdown_sample", np.nan),
                    "กองทุน": len(alt.weights),
                })
        if rows:
            st.dataframe(
                T.style_frame(pd.DataFrame(rows).set_index("Objective"),
                              percent_cols=["Return", "Volatility", "Max DD"],
                              number_cols=["Sharpe"]),
                use_container_width=True)

    # ---- stress & MC comparison -------------------------------------------
    T.rule("Stress Test และ Monte Carlo — เทียบข้างกัน")
    sc1, sc2 = st.columns([1.2, 1], gap="large")
    with sc1:
        cur_stress = cached_stress(weights_key(client.holdings), True)
        prop_stress = cached_stress(weights_key(solution.weights), True)
        if not cur_stress.empty:
            st.plotly_chart(
                charts.stress_bars(cur_stress, comparison=prop_stress),
                use_container_width=True, config={"displayModeBar": False})
            T.caption("แท่งคือพอร์ตปัจจุบัน สัญลักษณ์เพชรคือพอร์ตที่แนะนำ "
                      "หากเพชรอยู่ขวาของแท่ง หมายความว่าพอร์ตที่แนะนำ"
                      "จะขาดทุนน้อยกว่าในเหตุการณ์นั้น")
    with sc2:
        st.markdown("**Exposure ต่อประเด็นตลาดที่กำลังเกิดขึ้น**")
        theme_cmp = cautionlib.compare_cautions(CAUTIONS, prop_cautions)
        if not theme_cmp.empty:
            st.plotly_chart(charts.theme_exposure_chart(theme_cmp),
                            use_container_width=True,
                            config={"displayModeBar": False})
            reduced = theme_cmp[theme_cmp["เปลี่ยนแปลง"] < -0.02]
            if not reduced.empty:
                T.caption(
                    f"พอร์ตที่แนะนำลดความเกี่ยวข้องกับ "
                    f"{', '.join(reduced.index[:3])}")

    mcc1, mcc2 = st.columns([1.7, 1], gap="large")
    with mcc1:
        try:
            sim_prop = cached_simulation(
                weights_key(solution.weights), sim_method, n_paths, n_periods,
                period_unit, rebalance != "none", 1.0, student_df, block_size,
                tuple(sorted(cma.mixed_mu(list(solution.weights), UNIVERSE,
                                          hist_mu_2, cma_weight_2).items())),
                PROPOSED.window[0])
            sim_cur_aligned = cached_simulation(
                weights_key(CURRENT.weights), sim_method, n_paths, n_periods,
                period_unit, rebalance != "none", 1.0, student_df, block_size,
                mu_items, CURRENT.window[0])
            st.plotly_chart(
                charts.fan_chart(sim_cur_aligned.percentiles, period_unit, aum,
                                 overlay=sim_prop.percentiles,
                                 overlay_name="พอร์ตที่แนะนำ"),
                use_container_width=True, config={"displayModeBar": False})
            T.caption(
                f"แถบสีมิ้นต์และเส้นทึบ คือพอร์ตปัจจุบัน เส้นประสีทราย คือค่ากลาง"
                f"และเปอร์เซ็นไทล์ที่ 5/95 ของพอร์ตที่แนะนำ · "
                f"{sim_method} · {n_paths:,} เส้นทาง")
        except Exception as exc:
            sim_prop = None
            st.warning(f"ไม่สามารถจำลองพอร์ตที่แนะนำได้: {exc}")
    with mcc2:
        if sim_prop is not None:
            a = sim_cur_aligned.terminal_stats(aum)
            b = sim_prop.terminal_stats(aum)
            proj = pd.DataFrame({
                "ปัจจุบัน": [a["median"], a["p5"], a["p95"], a["prob_loss"],
                            a["median_max_dd"]],
                "เสนอใหม่": [b["median"], b["p5"], b["p95"], b["prob_loss"],
                             b["median_max_dd"]],
            }, index=["กรณีกลาง (Median)", "Percentile 5", "Percentile 95",
                      "โอกาสขาดทุน", "Max Drawdown (ทั่วไป)"])
            st.dataframe(
                proj.style.format(lambda v: f"฿{v:,.0f}" if abs(v) > 10
                                  else f"{v:.1%}"),
                use_container_width=True)

    # ---- trade list --------------------------------------------------------
    T.rule("Implementation — รายการซื้อขาย")
    trades = opt.trade_list(client.holdings, solution.weights, UNIVERSE, aum)
    if trades.empty:
        T.alert("พอร์ตที่แนะนำตรงกับพอร์ตปัจจุบัน ไม่ต้องทำรายการซื้อขาย", "ok")
    else:
        trade_view = trades.copy()
        for col in ("ปัจจุบัน", "เสนอใหม่", "เปลี่ยนแปลง"):
            trade_view[col] = trade_view[col] * 100.0
        st.dataframe(
            trade_view.set_index("กองทุน"), use_container_width=True,
            column_config={
                "ปัจจุบัน": st.column_config.NumberColumn(format="%.1f%%"),
                "เสนอใหม่": st.column_config.NumberColumn(format="%.1f%%"),
                "เปลี่ยนแปลง": st.column_config.NumberColumn(format="%+.1f%%"),
                "มูลค่า (บาท)": st.column_config.NumberColumn(format="%.0f"),
            })
        st.download_button(
            "ดาวน์โหลดรายการซื้อขาย (CSV)",
            trades.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{client.id}_proposed_trades.csv", mime="text/csv")

    # ---- PDF export --------------------------------------------------------
    def _proposed_sections(picked: List[str]) -> bytes:
        want = set(picked)
        out: List[reportlib.Section] = []
        split = opt.core_satellite_split(solution.weights, UNIVERSE)

        if "mandate" in want:
            bullets = [
                f"วิธีจัดพอร์ต: <b>{build_mode}</b>"
                + (f" · Objective <b>{objective}</b>" if not custom_mode else ""),
                f"เพดานน้ำหนักต่อกองทุน {max_weight:.0%} · "
                f"จำนวนกองทุนเป้าหมาย {max_funds} · "
                f"Satellite ไม่เกิน {max_sat:.0%}",
                f"Fund Universe ที่ลงทุนได้ {len(eligible)} กองทุน "
                f"(เสี่ยงสูงกว่าระดับลูกค้า {len(above_codes)} กอง "
                f"จำกัดไม่เกิน {above_budget:.0%})",
                f"ผลตอบแทนคาดหวังจาก {cma.blend_label(cma_weight_2)} · "
                f"ข้อมูลร่วม {th.month(panel_opt.index[0])} – "
                f"{th.month(panel_opt.index[-1])} ({len(panel_opt):,} วัน)",
            ]
            if solution.message:
                bullets.append(f"หมายเหตุจากระบบ: {solution.message}")
            out.append(reportlib.Section("เงื่อนไขการจัดพอร์ต", bullets=bullets))

        if "summary" in want:
            out.append(reportlib.Section(
                "สรุปสถิติพอร์ตที่แนะนำ",
                columns=["รายการ", "ค่า"],
                rows=[["Expected Return (ต่อปี)", pct(solution.expected_return)],
                      ["Expected Volatility (ต่อปี)", pct(solution.volatility)],
                      ["Sharpe Ratio", num(solution.sharpe)],
                      ["Max Drawdown ที่เคยเกิดในข้อมูล",
                       pct(solution.diagnostics.get("max_drawdown_sample",
                                                    np.nan))],
                      ["จำนวนกองทุน", f"{len(solution.weights)}"],
                      ["Core / Satellite",
                       f"{split['Core']:.0%} / {split['Satellite']:.0%}"],
                      ["Turnover", pct(turn, 0)],
                      ["มูลค่าที่ต้องซื้อขายด้านเดียว", thb(turn * aum)]],
                align_right=[1]))

        if "holdings" in want:
            rows = []
            for code, w in sorted(solution.weights.items(), key=lambda kv: -kv[1]):
                fund = UNIVERSE[code]
                rows.append([code, fund.name, fund.asset_class,
                             str(fund.risk_level), th.role(fund.role),
                             pct(w), thb(w * aum)])
            out.append(reportlib.Section(
                "รายการกองทุนที่แนะนำ",
                columns=["กองทุน", "ชื่อกองทุน", "Asset Class", "ระดับ",
                         "Core / Satellite", "น้ำหนัก", "มูลค่า"],
                rows=rows, align_right=[3, 5, 6]))

        if "allocation" in want:
            klass: Dict[str, float] = {}
            for code, w in solution.weights.items():
                klass[UNIVERSE[code].asset_class] = (
                    klass.get(UNIVERSE[code].asset_class, 0.0) + w)
            out.append(reportlib.Section(
                "สัดส่วนตาม Asset Class",
                columns=["Asset Class", "น้ำหนัก", "มูลค่า"],
                rows=[[k, pct(v), thb(v * aum)]
                      for k, v in sorted(klass.items(), key=lambda kv: -kv[1])],
                align_right=[1, 2]))
            band_rows = []
            for bucket, band in profile.bands.items():
                held = PROPOSED.bucket_exposure.get(bucket, 0.0)
                inside = band[0] - 1e-9 <= held <= band[1] + 1e-9
                band_rows.append([bucket, pct(held),
                                  f"{band[0]:.0%} – {band[1]:.0%}",
                                  "อยู่ในกรอบ" if inside else "นอกกรอบ"])
            out.append(reportlib.Section(
                "สัดส่วนเทียบกรอบความเหมาะสม",
                columns=["ประเภทสินทรัพย์", "พอร์ตที่แนะนำ", "กรอบที่แนะนำ",
                         "สถานะ"],
                rows=band_rows, align_right=[1, 2]))

        if "comparison" in want and not comp.empty:
            out.append(reportlib.Section(
                "เปรียบเทียบกับพอร์ตปัจจุบัน",
                lead=f"วัดบนช่วงข้อมูลเดียวกันทั้งสองพอร์ต "
                     f"({th.date(win_from)} – {th.date(win_to)}) "
                     f"เพื่อให้เทียบกันได้ตรง ๆ",
                columns=["รายการ"] + list(comp.columns),
                rows=[[idx] + [f"{v:,.2f}" if isinstance(v, float) else str(v)
                               for v in row]
                      for idx, row in comp.iterrows()],
                align_right=list(range(1, len(comp.columns) + 1))))

        if "trades" in want:
            if trades.empty:
                out.append(reportlib.Section(
                    "รายการซื้อขายที่ต้องทำ",
                    bullets=["พอร์ตที่แนะนำตรงกับพอร์ตปัจจุบัน "
                             "ไม่ต้องทำรายการซื้อขาย"]))
            else:
                out.append(reportlib.Section(
                    "รายการซื้อขายที่ต้องทำ",
                    lead=f"Turnover {pct(turn, 0)} · "
                         f"มูลค่าซื้อขายด้านเดียว {thb(turn * aum)}",
                    columns=["กองทุน", "ชื่อกองทุน", "ปัจจุบัน", "เสนอใหม่",
                             "เปลี่ยนแปลง", "รายการ", "มูลค่า"],
                    rows=[[r["กองทุน"], r["ชื่อกองทุน"], pct(r["ปัจจุบัน"]),
                           pct(r["เสนอใหม่"]), f"{r['เปลี่ยนแปลง']:+.1%}",
                           r["รายการ"], thb(r["มูลค่า (บาท)"])]
                          for _, r in trades.iterrows()],
                    align_right=[2, 3, 4, 6]))

        if "suitability" in want:
            if prop_findings:
                rows = [[th.kind(f.kind), th.finding_severity(f.severity), f.detail]
                        for f in prop_findings]
            else:
                rows = [["—", "ผ่านเกณฑ์", "ไม่พบข้อที่ต้องแก้ไข"]]
            out.append(reportlib.Section(
                "ผลตรวจความเหมาะสมของพอร์ตที่แนะนำ",
                lead=f"สถานะโดยรวม: <b>{th.status(prop_status)}</b> "
                     f"(พอร์ตปัจจุบัน: {th.status(status)})",
                columns=["ประเภท", "ระดับ", "รายละเอียด"], rows=rows))

        if "stress" in want:
            prop_stress = cached_stress(weights_key(solution.weights),
                                        allow_proxy)
            cur_stress = cached_stress(weights_key(client.holdings), allow_proxy)
            if not prop_stress.empty:
                rows = []
                for idx, r in prop_stress.iterrows():
                    before = (cur_stress.loc[idx, "ผลตอบแทน"]
                              if idx in cur_stress.index else np.nan)
                    rows.append([idx, pct(before), pct(r["ผลตอบแทน"]),
                                 f"{r['ผลตอบแทน'] - before:+.1%}"
                                 if before == before else "—"])
                out.append(reportlib.Section(
                    "Stress Test เทียบสองพอร์ต",
                    lead="ตัวเลขบวกในคอลัมน์สุดท้ายแปลว่าพอร์ตที่แนะนำ"
                         "ขาดทุนน้อยกว่าในเหตุการณ์นั้น",
                    columns=["เหตุการณ์", "พอร์ตปัจจุบัน", "พอร์ตที่แนะนำ",
                             "ส่วนต่าง"],
                    rows=rows, align_right=[1, 2, 3]))

        if "montecarlo" in want and sim_prop is not None:
            b = sim_prop.terminal_stats(aum)
            out.append(reportlib.Section(
                "Monte Carlo Simulation ของพอร์ตที่แนะนำ",
                lead=f"{n_paths:,} เส้นทาง · วิธี {sim_method} · "
                     f"ผลตอบแทนคาดหวังจาก {cma.blend_label(cma_weight_2)}",
                columns=["กรณี", "มูลค่าปลายทาง", "ผลตอบแทนต่อปี"],
                rows=[["กรณีกลาง (Median)", thb(b["median"]),
                       pct(b["median_cagr"])],
                      ["กรณีแย่ (Percentile 5)", thb(b["p5"]),
                       pct(b["cagr_p5"])],
                      ["กรณีดี (Percentile 95)", thb(b["p95"]),
                       pct(b["cagr_p95"])],
                      ["โอกาสขาดทุน", pct(b["prob_loss"], 0), "—"],
                      ["Max Drawdown (ทั่วไป)", pct(b["median_max_dd"]), "—"]],
                align_right=[1, 2]))

        if "cautions" in want and prop_cautions:
            out.append(reportlib.Section(
                "ข้อควรระวังจากภาวะตลาดสำหรับพอร์ตที่แนะนำ",
                columns=["ระดับ", "ประเด็น", "ผลต่อพอร์ตนี้", "สัดส่วนที่กระทบ"],
                rows=[[th.severity(c.severity), c.title,
                       cautionlib.theme_caution(c), pct(c.exposure, 0)]
                      for c in prop_cautions],
                align_right=[3]))

        def _fig(figure, title, lead="", width_mm=178.0, height=460):
            png = reportlib.figure_png(figure, height=height)
            if png:
                out.append(reportlib.Section(title, lead=lead, image=png,
                                             image_width_mm=width_mm))

        if "chart_allocation" in want:
            _fig(charts.allocation_donut(
                     solution.weights,
                     {c: UNIVERSE[c].name for c in solution.weights},
                     centre=f"{len(solution.weights)} กองทุน"),
                 "สัดส่วนรายกองทุนของพอร์ตที่แนะนำ", width_mm=130, height=440)
            _fig(charts.core_satellite_bar(split), "Core / Satellite",
                 height=170)
        if "chart_weights" in want:
            _fig(charts.weight_comparison(
                     client.holdings, solution.weights,
                     {c: UNIVERSE[c].name
                      for c in set(client.holdings) | set(solution.weights)}),
                 "เทียบน้ำหนักรายกองทุนสองพอร์ต",
                 "แท่งบนคือพอร์ตปัจจุบัน แท่งล่างคือพอร์ตที่แนะนำ", height=520)
        if "chart_frontier" in want:
            # Reuse the figure the tab already drew, points and all, so the
            # document cannot disagree with the screen it was exported from.
            _fig(charts.frontier_chart(frontier, points),
                 "Efficient Frontier ภายใต้เงื่อนไขของลูกค้ารายนี้",
                 "ทุกจุดบนเส้นคือพอร์ตที่ลูกค้ารายนี้ลงทุนได้จริง "
                 "ไม่ใช่เส้นขอบบน Fund Universe แบบไม่มีเงื่อนไข", height=440)
        if "chart_stress" in want:
            _ps = cached_stress(weights_key(solution.weights), allow_proxy)
            _cs = cached_stress(weights_key(client.holdings), allow_proxy)
            if not _cs.empty:
                _fig(charts.stress_bars(_cs, comparison=_ps),
                     "Stress Test เทียบสองพอร์ต",
                     "แท่งคือพอร์ตปัจจุบัน สัญลักษณ์เพชรคือพอร์ตที่แนะนำ",
                     height=520)
        if "chart_montecarlo" in want and sim_prop is not None:
            _fig(charts.fan_chart(sim_cur_aligned.percentiles, period_unit, aum,
                                  overlay=sim_prop.percentiles,
                                  overlay_name="พอร์ตที่แนะนำ"),
                 "Monte Carlo เทียบสองพอร์ต",
                 f"{n_paths:,} เส้นทาง · {sim_method}", height=440)
        if "chart_exposure" in want and not theme_cmp.empty:
            _fig(charts.theme_exposure_chart(theme_cmp),
                 "Exposure ต่อประเด็นตลาดที่กำลังเกิดขึ้น",
                 "เทียบว่าพอร์ตที่แนะนำเพิ่มหรือลดความเกี่ยวข้องกับแต่ละประเด็น",
                 height=430)

        return reportlib.build_pdf(
            f"พอร์ตที่แนะนำ — {client.name}",
            f"{client.id} · ระดับความเสี่ยง {profile.level} · "
            f"{build_mode}" + ("" if custom_mode else f" · {objective}"),
            out,
            footer=f"K-ADVISOR · จัดทำ {datetime.now():%d/%m/%Y %H:%M}",
            disclaimer=DISCLAIMER)

    export_panel("t2", reportlib.PROPOSED_BLOCKS, reportlib.DEFAULT_PROPOSED,
                 _proposed_sections, "proposed-portfolio")


# --------------------------------------------------------------------------- #
# TAB 3 — Market & financial literacy
# --------------------------------------------------------------------------- #
with tab3:
    _fetched = MARKET.fetched_at.astimezone()
    st.markdown(
        f'<div class="kalert {["ok", "info", "watch", "breach"][temp_level]}">'
        f'<b>ภาวะตลาดโดยรวม: {th.TEMPERATURE[temp_level]}</b> — '
        f'{temp_msg} · ดึงข้อมูลเมื่อ '
        f'{th.date(pd.Timestamp(_fetched))} {_fetched:%H:%M}</div>',
        unsafe_allow_html=True)

    # ---- macro dashboard ---------------------------------------------------
    T.rule("ภาพรวมเศรษฐกิจและตลาด")
    if MARKET.macro.ok:
        table = MARKET.macro.table
        for group in ["ไทย", "หุ้นต่างประเทศ", "อัตราดอกเบี้ย",
                      "ความเชื่อมั่นต่อความเสี่ยง", "ค่าเงิน",
                      "สินค้าโภคภัณฑ์", "เอเชีย"]:
            rows = table[table["group"] == group]
            if rows.empty:
                continue
            st.markdown(f"**{group}**")
            cols = st.columns(max(len(rows), 3), gap="small")
            for col, (_, r) in zip(cols, rows.iterrows()):
                with col:
                    chg = r["chg_3m"]
                    unit = r["unit"]
                    level = (f"{r['level']:.2f}%" if unit == "percent"
                             else f"{r['level']:,.2f}" if unit == "fx"
                             else f"{r['level']:,.0f}")
                    pctile = r["pctile_3y"]
                    st.markdown(
                        T.metric_card(
                            r["label"], level,
                            note=(f"Percentile 3 ปี "
                                  f"{pctile:.0%}" if pd.notna(pctile) else ""),
                            delta=(f"{chg:+.1%} ใน 3 เดือน"
                                   if pd.notna(chg) else ""),
                            delta_tone=("pos" if pd.notna(chg) and chg >= 0
                                        else "neg")),
                        unsafe_allow_html=True)
                    if r["ticker"] in MARKET.macro.history.columns:
                        st.plotly_chart(
                            charts.macro_sparkline(
                                MARKET.macro.history[r["ticker"]],
                                positive=bool(pd.notna(chg) and chg >= 0)),
                            use_container_width=True,
                            config={"displayModeBar": False})
            st.write("")
    else:
        T.alert("ไม่สามารถดึงข้อมูลตลาดแบบเรียลไทม์ได้ แต่ประเด็นเชิงโครงสร้าง"
                "ด้านล่างยังใช้ได้ กรุณาตรวจการเชื่อมต่อ แล้วกด "
                "<b>รีเฟรชข้อมูลตลาด</b> ในแถบด้านซ้าย", "watch")

    # ---- signals -----------------------------------------------------------
    if MARKET.signals:
        T.rule("ข้อมูลกำลังบอกอะไร")
        sig_cols = st.columns(min(3, len(MARKET.signals)), gap="small")
        for i, sig in enumerate(MARKET.signals[:6]):
            with sig_cols[i % len(sig_cols)]:
                tone = ["mint", "sky", "amber", "coral"][int(np.clip(sig.severity, 0, 3))]
                st.markdown(
                    f'<div class="kcard flush" style="margin-bottom:0.55rem">'
                    f'{T.chip(sig.state, tone)}'
                    f'<div style="font-family:{T.SERIF};font-size:0.95rem;'
                    f'margin:0.4rem 0 0.2rem 0">{sig.label}</div>'
                    f'<div class="knote">{sig.detail}</div></div>',
                    unsafe_allow_html=True)

    # ---- hot topics --------------------------------------------------------
    T.rule("ประเด็นร้อนที่ต้องติดตาม")
    ht1, ht2 = st.columns([1.9, 1], gap="large")
    with ht1:
        client_tags = set()
        for code in client.holdings:
            fund = UNIVERSE.get(code)
            if fund:
                client_tags |= set(fund.tags)
        relevant_keys = {c.theme_key for c in CAUTIONS}

        prefs.remember("t3_filter", False, cast=prefs.as_bool)
        only_relevant = st.toggle(
            f"แสดงเฉพาะประเด็นที่เกี่ยวข้องกับพอร์ตของ{client.name}",
            key="t3_filter")

        for tv in MARKET.themes:
            if only_relevant and tv.theme.key not in relevant_keys:
                continue
            tone = SEVERITY_TONE.get(tv.severity_key, "dim")
            touches = tv.theme.key in relevant_keys
            exposure = next((c.exposure for c in CAUTIONS
                             if c.theme_key == tv.theme.key), 0.0)
            badge = (T.chip(f"{exposure:.0%} ของพอร์ตลูกค้ารายนี้", "mint")
                     if touches else "")
            with st.expander(f"{tv.severity_label} · {tv.theme.title}",
                             expanded=tv.severity >= 2 and touches):
                st.markdown(
                    f'{T.chip(tv.theme.category, "dim")}{badge}',
                    unsafe_allow_html=True)
                st.markdown(f"**ประเด็นนี้คืออะไร** — {tv.theme.what}")
                st.markdown(f"**กระทบพอร์ตอย่างไร** — {tv.theme.why}")
                if tv.signal:
                    T.alert(f"<b>ตัวเลขล่าสุด</b> — {tv.signal.detail}", "info")
                st.markdown("**สิ่งที่ต้องติดตาม**")
                st.markdown("\n".join(f"- {w}" for w in tv.theme.watch))
                if tv.headlines:
                    st.markdown("**ข่าวล่าสุด**")
                    for h in tv.headlines:
                        age = (f"{h.age_hours:.0f} ชม. ที่แล้ว"
                               if h.age_hours is not None and h.age_hours < 72
                               else (th.date(pd.Timestamp(h.published))
                                     if h.published else ""))
                        st.markdown(
                            f'<div class="kalert info" style="padding:0.45rem 0.7rem">'
                            f'<a href="{h.link}" target="_blank">{h.title}</a>'
                            f'<div class="src">{h.source} · {age}</div></div>',
                            unsafe_allow_html=True)
                if touches:
                    holds = next((c.holdings for c in CAUTIONS
                                  if c.theme_key == tv.theme.key), [])
                    T.caption(
                        "<b>กองทุนในพอร์ตที่เกี่ยวข้อง:</b> "
                        + ", ".join(f"{n} ({w:.0%})" for _, n, w in holds))
    with ht2:
        st.markdown("**คนไทยกำลังค้นหาอะไร**")
        if MARKET.trends:
            finance = [t for t in MARKET.trends if t.finance_related]
            other = [t for t in MARKET.trends if not t.finance_related]
            if finance:
                for t in finance[:8]:
                    st.markdown(
                        f'<div class="kalert ok" style="padding:0.4rem 0.7rem">'
                        f'<b>{t.term}</b> <span class="src">· ค้นหา '
                        f'{t.traffic} ครั้ง</span></div>', unsafe_allow_html=True)
                T.caption(
                    "คำค้นหาที่เกี่ยวกับการเงินจาก Google Trends ประเทศไทย "
                    "อัปเดตพร้อมข้อมูลตลาด หากคำว่า <i>ตลาดหุ้น</i> หรือ "
                    "<i>กองทุนรวม</i> พุ่งขึ้น มักหมายความว่าลูกค้ากำลังจะโทรมา")
            else:
                T.caption("ขณะนี้ไม่มีคำค้นหาเกี่ยวกับการเงินติดเทรนด์ในไทย "
                          "ซึ่งตัวมันเองก็เป็นสัญญาณว่าตลาดยังสงบ")
            with st.expander(f"คำค้นหาที่ติดเทรนด์ทั้งหมด "
                             f"{len(MARKET.trends)} คำ"):
                st.markdown("\n".join(
                    f"- {t.term} · {t.traffic}" for t in MARKET.trends))
        else:
            T.caption(f"ไม่สามารถดึงข้อมูลเทรนด์การค้นหาได้ "
                      f"({MARKET.trends_status})")

    # ---- literacy ----------------------------------------------------------
    T.rule("Financial Literacy — แนวคิดที่อยู่เบื้องหลังเครื่องมือนี้")
    l1, l2 = st.columns(2, gap="large")
    LESSONS = [
        ("ระดับความเสี่ยงคือเพดาน ไม่ใช่เป้าหมาย",
         "คะแนน Suitability บอกระดับความเสี่ยงผลิตภัณฑ์สูงสุดที่ลูกค้า"
         "ลงทุนได้ และกรอบ Asset Allocation ที่แนะนำ ลูกค้าระดับ 8 <i>ได้รับอนุญาต</i>ให้ถือ"
         "ผลิตภัณฑ์ระดับ 8 ไม่ใช่ถูกบังคับให้ถือ ในทางกลับกัน ลูกค้าระดับ 4 "
         "ที่ถือเงินสดทั้งพอร์ตก็ไม่ได้เรียกว่าระมัดระวัง แต่คือการปฏิเสธ"
         "ความสามารถในการรับความเสี่ยงที่ตนได้รับการประเมินแล้วว่ารับได้ "
         "และเงินเฟ้อกำลังเก็บค่าธรรมเนียมจากการปฏิเสธนั้น"),
        ("Diversification อยู่ที่สหสัมพันธ์ ไม่ใช่จำนวนกองทุน",
         "กองทุนสิบกองที่วิ่งตามปัจจัยเดียวกัน ก็คือสถานะเดียวที่สวมหมวกสิบใบ "
         "ค่า Effective Positions ในส่วนที่ 1 คำนวณจากส่วนกลับของดัชนี "
         "Herfindahl ของน้ำหนัก ส่วน Diversification Ratio เปรียบเทียบ"
         "ความผันผวนถ่วงน้ำหนักเฉลี่ยของกองทุนที่ถือ กับความผันผวนของพอร์ต "
         "ถ้าค่านี้ใกล้ 1.0 การรวมกองทุนเหล่านี้เข้าด้วยกันไม่ได้ช่วยอะไรเลย"),
        ("Volatility กับ Drawdown เป็นคำสัญญาที่ต่างกัน",
         "Volatility บอกความสั่นไหวโดยทั่วไป ส่วน Drawdown บอกสิ่งที่แย่ที่สุด"
         "ที่เกิดขึ้นจริง สิ่งที่ลูกค้ารู้สึกคือ Drawdown ไม่ใช่ส่วนเบี่ยงเบน"
         "มาตรฐาน พอร์ตหนึ่งอาจมี Volatility ไม่สูง แต่ยัง Drawdown ได้ถึง 30% "
         "หากช่วงเวลาที่แย่มาเกาะกลุ่มกัน ซึ่งเป็นสิ่งที่ Block Bootstrap "
         "ออกแบบมาเพื่อจับ และเป็นสิ่งที่แบบจำลองการแจกแจงปกติมองไม่เห็น"),
        ("ผลตอบแทนในอดีตไม่ใช่ผลตอบแทนคาดหวัง",
         "การป้อนผลตอบแทนย้อนหลังห้าปีเข้าไปใน Optimiser จะทำให้มันซื้อ"
         "สิ่งที่เพิ่งขึ้นมาแล้ว เครื่องมือนี้จึงตั้งค่าเริ่มต้นให้ใช้ Capital "
         "Market Assumptions แบบมองไปข้างหน้า ซึ่งสร้างจากดอกเบี้ยเงินสด บวก "
         "Term Premium บวก Equity Risk Premium ส่วน Volatility และสหสัมพันธ์"
         "ยังใช้ข้อมูลในอดีต เพราะสองสิ่งนี้มีความคงทน แต่ค่าเฉลี่ยผลตอบแทนไม่มี"),
        ("ค่าเงินคือสถานะที่คุณไม่ได้เลือกถือ",
         "กองทุนต่างประเทศที่ไม่ป้องกันความเสี่ยงค่าเงินคือเดิมพันสองอย่าง "
         "ทั้งตัวสินทรัพย์และค่าเงิน ในช่วงเวลาสั้น ๆ ขาของค่าเงินอาจใหญ่กว่า"
         "ขาของสินทรัพย์ เมื่อลูกค้าถามว่าทำไมกองทุนหุ้นสหรัฐฯ ไม่ขึ้นตาม "
         "S&P คำตอบมักคืออัตราแลกเปลี่ยน ไม่ใช่ฝีมือผู้จัดการกองทุน"),
        ("Rebalancing คือที่อยู่ของวินัยการลงทุน",
         "พอร์ตที่ปล่อยไว้จะไหลไปหาสิ่งที่ให้ผลตอบแทนดีที่สุด ซึ่งก็คือไหลไปหา"
         "สินทรัพย์ที่แพงที่สุด ในจังหวะที่มันเสี่ยงที่สุด การตั้งค่า Rebalancing "
         "ในแถบด้านซ้ายเปลี่ยนทุกตัวเลขในส่วนที่ 1 ลองสลับไปเป็น Buy & Hold "
         "แล้วดูว่า Max Drawdown แย่ลงเพียงใด"),
    ]
    for i, (title, body) in enumerate(LESSONS):
        with (l1 if i % 2 == 0 else l2):
            st.markdown(
                f'<div class="kcard" style="margin-bottom:0.7rem">'
                f'<div style="font-family:{T.SERIF};font-size:1.02rem;'
                f'margin-bottom:0.35rem;color:{T.TEXT}">{title}</div>'
                f'<div class="knote">{body}</div></div>',
                unsafe_allow_html=True)

    with st.expander("Capital Market Assumptions ที่ใช้ในการจัดพอร์ต"):
        st.dataframe(
            T.style_frame(cma.cma_table(), percent_cols=["ผลตอบแทนคาดหวัง"]),
            use_container_width=True)
        T.caption(
            "องค์ประกอบที่ใช้สร้าง: ดอกเบี้ยเงินสดไทย "
            f"{cma.THB_CASH_RATE:.2%}, term premium {cma.THB_TERM_PREMIUM:.2%}, "
            f"credit spread {cma.THB_CREDIT_SPREAD:.2%}, equity risk premium "
            f"{cma.EQUITY_RISK_PREMIUM:.2%}, ส่วนเพิ่มตลาดเกิดใหม่ "
            f"{cma.EM_EXTRA_PREMIUM:.2%} — ตัวเลขเหล่านี้เป็นสมมติฐาน "
            "ไม่ใช่การพยากรณ์ และถูกระบุไว้อย่างชัดเจนเพื่อให้ผู้ที่ทบทวน"
            "คำแนะนำสามารถโต้แย้งได้ตรงจุด")

    with st.expander("กรอบเกณฑ์ Suitability ฉบับเต็ม"):
        band_rows = []
        for lvl in [1, 4, 5, 7, 8]:
            p = clientlib.RISK_PROFILES[lvl]
            row = {"ระดับความเสี่ยง": f"{lvl} · {p.name_th}",
                   "คะแนน": p.score_range,
                   "ผลิตภัณฑ์ที่ลงทุนได้": p.acceptable_levels}
            for bucket in uni.SEC_BUCKETS:
                lo, hi = p.band(bucket)
                row[bucket] = (f"≥ {lo:.0%}" if lo > 0 and hi >= 1.0
                               else f"≤ {hi:.0%}" if lo == 0
                               else f"{lo:.0%} – {hi:.0%}")
            band_rows.append(row)
        st.dataframe(pd.DataFrame(band_rows).set_index("ระดับความเสี่ยง"),
                     use_container_width=True)
        T.caption(
            "ถอดความจากแบบประเมินความเหมาะสมในการลงทุนของ ก.ล.ต./AIMC "
            "โดยพันธบัตรรัฐบาลและหุ้นกู้ที่มีอายุเกินหนึ่งปีใช้เพดานร่วมกัน"
            "ในแบบฟอร์มนั้น จึงรวมเป็นกลุ่มเดียวกันที่นี่ · "
            "ข้อควรสังเกตคือความขัดแย้งเชิงโครงสร้างที่เครื่องมือนี้ต้องแก้ "
            "กองทุนสินทรัพย์ทางเลือกทุกกองมีความเสี่ยงระดับ 8 ดังนั้นลูกค้าที่มี"
            "ระดับต่ำกว่า 8 จะไม่สามารถเติมกรอบสินทรัพย์ทางเลือกได้เลยหากไม่มี"
            "การลงนามรับทราบความเสี่ยง — ส่วนที่ 2 จัดการเรื่องนี้อย่างเปิดเผย "
            "ไม่ใช่ซ่อนไว้")


# --------------------------------------------------------------------------- #
# Settings are written back to the URL last, once every widget has had its say.
# sync_url only writes when the result actually differs, because assigning to
# st.query_params triggers a rerun and an unconditional write would loop.
prefs.sync_url(
    REMEMBERED + [k for k in st.session_state if str(k).startswith("aum_")])

T.footer(
    "K-ADVISOR · เครื่องมือสำหรับผู้แนะนำการลงทุนในการพูดคุยกับลูกค้าสินทรัพย์สูง · "
    "ผลตอบแทนกองทุนจากแฟ้มข้อมูล K-Asset · ข้อมูลเศรษฐกิจจาก Yahoo Finance · "
    "ข่าวจาก Google News RSS · ความสนใจค้นหาจาก Google Trends RSS<br>"
    "พอร์ตลูกค้าทั้งหมดเป็นตัวอย่างสมมติเพื่อการสาธิต ผลตอบแทนคาดหวังเป็น"
    "สมมติฐานที่ระบุไว้ ไม่ใช่การพยากรณ์ การจำลองจากข้อมูลในอดีตไม่ใช่การรับประกัน"
    "ผลการดำเนินงานในอนาคต เครื่องมือนี้ใช้สนับสนุนการสนทนาเรื่องความเหมาะสม"
    "ในการลงทุน ไม่ใช่สิ่งที่ใช้แทนการสนทนานั้น"
)
