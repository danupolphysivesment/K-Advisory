"""Stress testing: historical event replay and factor-based hypothetical shocks.

Most K-Asset funds in the workbook launched after 2018, so a naive "replay the
GFC" would silently drop 90% of a portfolio and report a flattering number. The
replay here is explicit about coverage: a fund that did not exist during an
event is stood in for by a long-history proxy of the same asset class, and the
result is labelled with how much of the portfolio was proxied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .data import portfolio_returns
from .universe import PROXY_BY_CLASS


@dataclass(frozen=True)
class Event:
    key: str
    name: str
    start: str
    end: str
    blurb: str
    drivers: tuple          # exposure tags the event hits hardest


# Windows are peak-to-trough of the relevant market, not calendar quarters.
EVENTS: List[Event] = [
    Event("gfc", "วิกฤตการเงินโลก (GFC)", "2008-09-01", "2009-03-09",
          "Lehman ล้ม หุ้นทั่วโลกลดลงราวครึ่งหนึ่งและหุ้นไทยลงตามไปด้วย "
          "มีเพียงพันธบัตรรัฐบาลและเงินสดที่รักษาเงินต้นได้",
          ("th-equity", "global-equity", "credit-spread", "us-equity")),
    Event("euro", "วิกฤตหนี้ยุโรป", "2011-07-01", "2011-10-04",
          "ตลาดปรับราคาหนี้รัฐบาลยุโรปกลุ่มรอบนอกใหม่ สินทรัพย์เสี่ยงถูกเทขาย "
          "ขณะที่ทองคำและพันธบัตรคุณภาพสูงปรับขึ้น",
          ("eu-equity", "global-equity", "credit-spread")),
    Event("floods", "มหาอุทกภัยประเทศไทย", "2011-10-01", "2011-12-30",
          "ภาวะช็อกห่วงโซ่การผลิตในประเทศ พอร์ตที่มีแต่หุ้นไทยไม่มีอะไรรองรับ "
          "ขณะที่กองทุนต่างประเทศแทบไม่ได้รับผลกระทบ",
          ("th-equity",)),
    Event("taper", "Taper Tantrum (เฟดส่งสัญญาณลดคิวอี)", "2013-05-22", "2013-06-25",
          "เฟดส่งสัญญาณลดการอัดฉีด ตราสารหนี้อายุยาวและสินทรัพย์ตลาดเกิดใหม่ "
          "ถูกกระทบพร้อมกัน เป็นรูปแบบความล้มเหลวคลาสสิกของพอร์ตที่มีแต่ตราสารหนี้",
          ("duration", "long-duration", "em-equity", "th-equity", "thb-rates")),
    Event("rmb", "จีนลดค่าเงินหยวน", "2015-08-11", "2015-09-29",
          "ธนาคารกลางจีนลดค่าเงินหยวน หุ้นเอเชียและสินค้าโภคภัณฑ์ปรับลงพร้อมกัน",
          ("china", "asia-equity", "em-equity", "commodities")),
    Event("oil", "ราคาน้ำมันดิ่ง", "2015-11-01", "2016-02-11",
          "น้ำมันดิบหลุด 30 ดอลลาร์ กองทุนหุ้นพลังงานและสินค้าโภคภัณฑ์ขาดทุนนำตลาด",
          ("oil", "energy", "commodities")),
    Event("brexit", "ประชามติ Brexit", "2016-06-23", "2016-06-27",
          "การปรับราคาความเสี่ยงยุโรปและเงินปอนด์ภายในสองวัน",
          ("eu-equity", "global-equity")),
    Event("volmageddon", "Volmageddon (การคลายสถานะ short volatility)", "2018-02-01", "2018-02-08",
          "การคลายสถานะ short volatility สั้นแต่รุนแรง เป็นบททดสอบที่ดีว่าพอร์ต "
          "จะเป็นอย่างไรเมื่อสหสัมพันธ์พุ่งขึ้นโดยไม่มีสัญญาณเตือน",
          ("us-equity", "global-equity")),
    Event("tradewar", "สงครามการค้าสหรัฐฯ-จีน", "2018-10-01", "2018-12-24",
          "การขึ้นภาษีนำเข้าลากหุ้นเทคโนโลยีและผู้ส่งออกเอเชียลงไปด้วยกัน",
          ("china", "technology", "asia-equity", "us-equity")),
    Event("covid", "วิกฤตโควิด-19", "2020-02-19", "2020-03-23",
          "ตลาดหมีที่เร็วที่สุดในประวัติศาสตร์ ทุกสินทรัพย์ยกเว้นเงินสดและพันธบัตร "
          "รัฐบาลปรับลงพร้อมกัน แม้แต่ทองคำก็ถูกขายเพื่อหาสภาพคล่อง",
          ("th-equity", "global-equity", "us-equity", "credit-spread", "property",
           "gold", "oil")),
    Event("inflation22", "เงินเฟ้อและดอกเบี้ยพุ่ง ปี 2022", "2022-01-01", "2022-10-14",
          "ตราสารหนี้และหุ้นปรับลงพร้อมกันเป็นครั้งแรกในรอบหนึ่งชั่วอายุคน "
          "สมมติฐานการกระจายความเสี่ยงแบบ 60/40 ใช้ไม่ได้",
          ("duration", "long-duration", "us-rates", "thb-rates", "growth",
           "technology", "us-equity")),
    Event("chinatech", "จีนคุมเทคโนโลยีและวิกฤตอสังหาฯ", "2021-02-17", "2022-10-31",
          "การกำกับดูแลที่เข้มงวดบวกกับวิกฤตสินเชื่อผู้พัฒนาอสังหาริมทรัพย์ "
          "เป็นการทรุดตัวยืดเยื้อ 20 เดือน ไม่ใช่การร่วงเร็ว จึงทนถือได้ยากกว่า",
          ("china", "asia-equity", "technology", "em-equity")),
    Event("svb", "วิกฤตธนาคาร SVB", "2023-03-08", "2023-03-24",
          "การแห่ถอนเงินจากธนาคารภูมิภาคทำให้หุ้นกลุ่มการเงินถูกปรับราคาใหม่ "
          "และดันราคาทองคำขึ้น",
          ("financials", "credit-spread", "us-equity")),
    Event("carry24", "การคลาย Yen Carry Trade", "2024-07-31", "2024-08-05",
          "การขึ้นดอกเบี้ยของ BoJ บังคับให้เกิดการคลายสถานะ carry ทั่วโลก "
          "ภายในสามวันทำการ สถานะ momentum ที่หนาแน่นปรับลงเป็นเลขสองหลัก",
          ("jp-equity", "jpy", "technology", "growth", "global-equity")),
    Event("tariff25", "ภาวะช็อกจากภาษีนำเข้า ปี 2025", "2025-02-01", "2025-04-30",
          "การขึ้นภาษีนำเข้ารอบใหม่ หุ้นสหรัฐฯ และผู้ส่งออกเอเชียปรับลง "
          "ขณะที่ทองคำปรับขึ้นต่อ เป็นบททดสอบที่ชัดเจนของโครงสร้าง barbell",
          ("us-equity", "china", "asia-equity", "technology", "global-equity")),
    Event("setslump", "หุ้นไทยทรุด ปี 2025", "2025-01-01", "2025-06-30",
          "หุ้นไทยถูกปรับลดมูลค่าต่อเนื่องหกเดือนจากความกังวลด้านการเมืองและการเติบโต "
          "เป็นเหตุการณ์ที่กระทบเฉพาะพอร์ตที่กระจุกในประเทศ",
          ("th-equity", "th-value", "th-growth")),
    Event("gold26", "ทองคำปรับฐาน ปี 2026", "2026-02-28", "2026-06-30",
          "การคลายสถานะทองคำที่หนาแน่นอย่างรุนแรง หลังปรับขึ้นต่อเนื่องหลายปี",
          ("gold", "real-assets", "safe-haven")),
]

EVENTS_BY_KEY: Dict[str, Event] = {e.key: e for e in EVENTS}


@dataclass
class StressResult:
    event: Event
    total_return: float
    max_drawdown: float
    worst_day: float
    n_obs: int
    coverage: float                  # weight held in funds with real history
    proxied_weight: float
    proxy_map: Dict[str, str]
    contributions: pd.Series         # per-holding contribution to the event return
    path: pd.Series                  # cumulative growth over the window
    usable: bool


def _resolve_series(
    returns: pd.DataFrame,
    code: str,
    window: pd.DatetimeIndex,
    asset_class: str,
    allow_proxy: bool,
) -> tuple:
    """(series, proxy_code_or_None). Returns (None, None) if nothing usable."""
    if code in returns.columns:
        s = returns[code].reindex(window).dropna()
        if len(s) >= max(2, int(0.5 * len(window))):
            return returns[code].reindex(window), None
    if not allow_proxy:
        return None, None
    proxy = PROXY_BY_CLASS.get(asset_class)
    if proxy and proxy in returns.columns:
        s = returns[proxy].reindex(window).dropna()
        if len(s) >= max(2, int(0.5 * len(window))):
            return returns[proxy].reindex(window), proxy
    return None, None


def run_event(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    event: Event,
    universe: Mapping,
    allow_proxy: bool = True,
) -> StressResult:
    """Replay one historical window against a set of weights."""
    window = returns.loc[event.start:event.end].index
    empty = pd.Series(dtype=float)
    if len(window) < 2:
        return StressResult(event, np.nan, np.nan, np.nan, 0, 0.0, 0.0, {},
                            empty, empty, usable=False)

    series: Dict[str, pd.Series] = {}
    proxy_map: Dict[str, str] = {}
    covered = 0.0
    proxied = 0.0

    for code, w in weights.items():
        if w <= 0:
            continue
        fund = universe.get(code)
        asset_class = fund.asset_class if fund else ""
        s, proxy = _resolve_series(returns, code, window, asset_class, allow_proxy)
        if s is None:
            continue
        series[code] = s.fillna(0.0)
        if proxy:
            proxy_map[code] = proxy
            proxied += w
        else:
            covered += w

    if not series:
        return StressResult(event, np.nan, np.nan, np.nan, 0, 0.0, 0.0, {},
                            empty, empty, usable=False)

    panel = pd.DataFrame(series)
    live = {c: weights[c] for c in panel.columns}
    scale = sum(live.values())
    live = {c: w / scale for c, w in live.items()}

    port = portfolio_returns(panel, live, rebalance="none")
    growth = (1.0 + port).cumprod()
    total = float(growth.iloc[-1] - 1.0)
    dd = float((growth / growth.cummax() - 1.0).min())

    # Per-holding contribution over the window (buy & hold, so weight drifts).
    contrib = {}
    for c in panel.columns:
        standalone = float((1.0 + panel[c]).prod() - 1.0)
        contrib[c] = live[c] * standalone
    contributions = pd.Series(contrib).sort_values()

    return StressResult(
        event=event,
        total_return=total,
        max_drawdown=dd,
        worst_day=float(port.min()),
        n_obs=len(port),
        coverage=covered,
        proxied_weight=proxied,
        proxy_map=proxy_map,
        contributions=contributions,
        path=growth,
        usable=True,
    )


def run_all(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    universe: Mapping,
    events: Optional[Sequence[Event]] = None,
    allow_proxy: bool = True,
) -> List[StressResult]:
    chosen = events if events is not None else EVENTS
    out = [run_event(returns, weights, e, universe, allow_proxy) for e in chosen]
    return [r for r in out if r.usable]


def results_frame(results: Sequence[StressResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "เหตุการณ์": r.event.name,
                "ช่วงเวลา": f"{r.event.start} → {r.event.end}",
                "ผลตอบแทน": r.total_return,
                "ขาดทุนสูงสุด": r.max_drawdown,
                "วันที่แย่ที่สุด": r.worst_day,
                "สัดส่วนที่มีข้อมูลจริง": r.coverage,
                "สัดส่วนที่ใช้ตัวแทน": r.proxied_weight,
            }
            for r in results
        ]
    ).set_index("เหตุการณ์").sort_values("ผลตอบแทน")


# --------------------------------------------------------------------------- #
# Factor-based hypothetical shocks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Shock:
    key: str
    name: str
    moves: Dict[str, float]        # factor label -> instantaneous return
    blurb: str


SHOCKS: List[Shock] = [
    Shock("equity_bear", "หุ้นทั่วโลก −20%",
          {"หุ้นต่างประเทศ": -0.20, "หุ้นไทย": -0.15},
          "สถานการณ์ตลาดหมีมาตรฐาน ใช้กับปัจจัยหุ้นทั้งสองตัว"),
    Shock("th_crash", "หุ้นไทย −25%",
          {"หุ้นไทย": -0.25},
          "ภาวะช็อกในประเทศเท่านั้น จากการเมืองหรือการคลัง โดยไม่ลุกลามไปตลาดโลก"),
    Shock("rates_up", "ผลตอบแทนพันธบัตรไทย +150 bp",
          {"ดอกเบี้ยไทย (duration)": -0.06},
          "เส้นอัตราผลตอบแทนไทยขยับขึ้นทั้งเส้น คิดเป็นผลขาดทุนด้านราคาของพอร์ต "
          "ตราสารหนี้ที่มี duration ประมาณ 4 ปี"),
    Shock("stagflation", "Stagflation: หุ้น −15%, ดอกเบี้ย +100 bp, น้ำมัน +30%",
          {"หุ้นต่างประเทศ": -0.15, "หุ้นไทย": -0.12,
           "ดอกเบี้ยไทย (duration)": -0.04, "น้ำมัน": 0.30, "ทองคำ": 0.08},
          "การเติบโตชะลอขณะที่เงินเฟ้อยังสูง เป็นภาวะที่การกระจายความเสี่ยงแบบ "
          "60/40 ใช้ไม่ได้"),
    Shock("gold_unwind", "ทองคำ −20%",
          {"ทองคำ": -0.20},
          "การคลายสถานะในสินทรัพย์ปลอดภัยที่นักลงทุนถือหนาแน่น"),
    Shock("risk_on", "Risk-on: หุ้น +15%, ทองคำ −8%",
          {"หุ้นต่างประเทศ": 0.15, "หุ้นไทย": 0.12, "ทองคำ": -0.08},
          "กรณีตลาดขาขึ้น แสดงให้เห็นว่าพอร์ตเชิงรับต้องเสียโอกาสไปเท่าใด"),
]

SHOCKS_BY_KEY: Dict[str, Shock] = {s.key: s for s in SHOCKS}


def factor_panel(returns: pd.DataFrame, proxies: Mapping[str, str],
                 start=None) -> pd.DataFrame:
    cols = {label: returns[code] for label, code in proxies.items()
            if code in returns.columns}
    panel = pd.DataFrame(cols)
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    return panel


def shock_portfolio(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    shock: Shock,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate each holding's response to a factor shock via OLS betas."""
    from .risk import factor_betas

    rows = []
    for code, w in weights.items():
        if w <= 0 or code not in returns.columns:
            continue
        betas = factor_betas(returns[code], factors)
        impact = sum(betas.get(f, 0.0) * move for f, move in shock.moves.items())
        rows.append({
            "fund": code,
            "weight": w,
            "fund_impact": impact,
            "contribution": w * impact,
            **{f"β {f}": betas.get(f, 0.0) for f in shock.moves},
        })
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("fund")
    frame.attrs["portfolio_impact"] = float(frame["contribution"].sum())
    return frame.sort_values("contribution")


def shock_summary(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    factors: pd.DataFrame,
    shocks: Optional[Sequence[Shock]] = None,
) -> pd.DataFrame:
    chosen = shocks if shocks is not None else SHOCKS
    rows = []
    for s in chosen:
        frame = shock_portfolio(returns, weights, s, factors)
        impact = frame.attrs.get("portfolio_impact", np.nan) if not frame.empty else np.nan
        rows.append({"สถานการณ์": s.name, "ผลกระทบต่อพอร์ต": impact,
                     "คำอธิบาย": s.blurb})
    return pd.DataFrame(rows).set_index("สถานการณ์")
