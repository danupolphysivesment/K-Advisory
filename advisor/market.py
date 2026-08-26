"""Market intelligence: macro indicators, live news, search trends and themes.

This module produces the material for Part 3 of the app and, just as
importantly, the *signals* that Parts 1 and 2 use to caution on an allocation.

Three live sources, all keyless, all optional:

* **Yahoo Finance** (via ``yfinance``) for the macro dashboard. Every indicator
  is reported as a level, a set of changes, and a percentile against its own
  three-year history — a level alone tells an RM nothing.
* **Google News RSS** for headlines per theme.
* **Google Trends RSS** for what retail investors in Thailand are actually
  searching for right now.

Nothing here raises on a network failure. Each fetch returns a status and the
app renders whatever succeeded, so a demo on a locked-down conference network
still works.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Bumped whenever Theme or Signal gains a field other modules read. advisor
# .cautions and app.py check it at import, so deploying a stale copy of this
# file produces a named, actionable error instead of a redacted AttributeError
# from deep inside a render.
SCHEMA_VERSION = 2

# --------------------------------------------------------------------------- #
# Macro indicators
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Indicator:
    ticker: str
    label: str
    group: str
    unit: str            # "index" | "percent" | "usd" | "fx"
    higher_is: str       # "risk-on" | "risk-off" | "neutral"
    blurb: str


INDICATORS: List[Indicator] = [
    Indicator("^SET.BK", "SET Index", "ไทย", "index", "risk-on",
              "ตลาดหุ้นไทย กองทุนหุ้นไทยทุกกองในพอร์ตเคลื่อนไหวตามดัชนีนี้"),
    Indicator("THB=X", "USD / THB", "ไทย", "fx", "neutral",
              "เงินบาทแข็งค่า (ตัวเลขลดลง) จะลดมูลค่าในสกุลบาทของกองทุน"
              "ต่างประเทศทุกกองที่ไม่ได้ป้องกันความเสี่ยงค่าเงิน"),
    Indicator("^GSPC", "S&P 500", "หุ้นต่างประเทศ", "index", "risk-on",
              "แกนกลางของความเสี่ยงหุ้นทั่วโลก"),
    Indicator("^NDX", "Nasdaq 100", "หุ้นต่างประเทศ", "index", "risk-on",
              "ปัจจัยหุ้นเติบโตและเทคโนโลยี รวมถึงธีม AI"),
    Indicator("^VIX", "VIX", "ความเชื่อมั่นต่อความเสี่ยง", "index", "risk-off",
              "ความผันผวนคาดการณ์ 30 วันของ S&P 500 เป็นค่าที่ตลาดประเมิน"
              "ความกลัวของตัวเองออกมาเป็นตัวเลข"),
    Indicator("^TNX", "อัตราผลตอบแทนพันธบัตรสหรัฐฯ 10 ปี", "อัตราดอกเบี้ย", "percent", "neutral",
              "อัตราคิดลดของสินทรัพย์อายุยาวทั้งโลก เมื่อผลตอบแทนพันธบัตรขึ้น "
              "ตราสารหนี้และหุ้นเติบโตจะถูกกระทบพร้อมกัน"),
    Indicator("DX-Y.NYB", "ดัชนีดอลลาร์สหรัฐฯ", "ค่าเงิน", "index", "risk-off",
              "ดอลลาร์แข็งค่าทำให้ภาวะการเงินโลกตึงตัวและกดดันตลาดเกิดใหม่"),
    Indicator("GC=F", "ทองคำ", "สินค้าโภคภัณฑ์", "usd", "neutral",
              "สินทรัพย์ปลอดภัยและเครื่องมือป้องกันความเสี่ยงดอกเบี้ยแท้จริง"),
    Indicator("CL=F", "น้ำมันดิบ WTI", "สินค้าโภคภัณฑ์", "usd", "neutral",
              "ต้นทุนพลังงาน เป็นตัวขับเคลื่อนเงินเฟ้อและอัตราการค้าของไทย"),
    Indicator("000300.SS", "CSI 300", "เอเชีย", "index", "risk-on",
              "หุ้นจีนที่ซื้อขายในประเทศ"),
    Indicator("^N225", "Nikkei 225", "เอเชีย", "index", "risk-on",
              "หุ้นญี่ปุ่น และเป็นเครื่องวัดสถานะ yen carry trade"),
    Indicator("^VNINDEX.VN", "VN Index", "เอเชีย", "index", "risk-on",
              "หุ้นเวียดนาม ส่วนของตลาดชายขอบ"),
]

INDICATORS_BY_TICKER = {i.ticker: i for i in INDICATORS}


@dataclass
class MacroSnapshot:
    table: pd.DataFrame            # one row per indicator
    history: pd.DataFrame          # price history, one column per ticker
    status: str
    fetched_at: datetime
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.table.empty


def _pct_change(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    cutoff = s.index[-1] - pd.Timedelta(days=days)
    past = s[s.index <= cutoff]
    if past.empty:
        return np.nan
    return float(s.iloc[-1] / past.iloc[-1] - 1.0)


def _percentile(series: pd.Series, window_days: int = 365 * 3) -> float:
    s = series.dropna()
    if len(s) < 30:
        return np.nan
    cutoff = s.index[-1] - pd.Timedelta(days=window_days)
    s = s[s.index >= cutoff]
    return float((s <= s.iloc[-1]).mean())


def fetch_macro(period: str = "3y", tickers: Optional[Sequence[str]] = None) -> MacroSnapshot:
    """Download macro indicators and summarise level, change and percentile."""
    chosen = list(tickers) if tickers else [i.ticker for i in INDICATORS]
    errors: List[str] = []
    try:
        import yfinance as yf

        raw = yf.download(chosen, period=period, progress=False,
                          auto_adjust=True, threads=True)
        if raw is None or len(raw) == 0:
            raise RuntimeError("no data returned")
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            close = raw[["Close"]].copy()
            close.columns = chosen[:1]
    except Exception as exc:
        return MacroSnapshot(pd.DataFrame(), pd.DataFrame(), "offline",
                             datetime.now(timezone.utc), [str(exc)])

    close = close.dropna(how="all")
    rows = []
    for ticker in chosen:
        if ticker not in close.columns:
            errors.append(f"{ticker}: not returned")
            continue
        series = close[ticker].dropna()
        if series.empty:
            errors.append(f"{ticker}: empty")
            continue
        meta = INDICATORS_BY_TICKER.get(ticker)
        rows.append({
            "ticker": ticker,
            "label": meta.label if meta else ticker,
            "group": meta.group if meta else "Other",
            "unit": meta.unit if meta else "index",
            "higher_is": meta.higher_is if meta else "neutral",
            "blurb": meta.blurb if meta else "",
            "level": float(series.iloc[-1]),
            "as_of": series.index[-1],
            "chg_1w": _pct_change(series, 7),
            "chg_1m": _pct_change(series, 30),
            "chg_3m": _pct_change(series, 91),
            "chg_12m": _pct_change(series, 365),
            "pctile_3y": _percentile(series),
        })

    table = pd.DataFrame(rows)
    status = "live" if not table.empty else "offline"
    return MacroSnapshot(table, close, status, datetime.now(timezone.utc), errors)


# --------------------------------------------------------------------------- #
# Signals derived from the macro snapshot
# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    key: str
    label: str
    state: str            # short human-readable state
    severity: int         # 0 calm .. 3 stressed
    detail: str           # the number plus what to do about it
    tags: Tuple[str, ...]
    reading: str = ""     # the number alone, for the caution cards


def _row(table: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    if table.empty or "ticker" not in table:
        return None
    hit = table[table["ticker"] == ticker]
    return hit.iloc[0] if len(hit) else None


def derive_signals(macro: MacroSnapshot) -> List[Signal]:
    """Turn the macro table into a handful of stated, defensible signals.

    These are deliberately mechanical: an RM can point at the number that
    produced each one, which is not true of a narrative generated from thin air.
    """
    out: List[Signal] = []
    t = macro.table

    vix = _row(t, "^VIX")
    if vix is not None:
        level, pct = vix["level"], vix["pctile_3y"]
        if level >= 28:
            state, sev = "กดดัน", 3
        elif level >= 20:
            state, sev = "ตึงตัว", 2
        elif level >= 15:
            state, sev = "ปกติ", 1
        else:
            state, sev = "สงบ", 0
        note = ("ในภาวะนี้สหสัมพันธ์ระหว่างสินทรัพย์จะเข้าใกล้ 1 การกระจาย"
                "ความเสี่ยงจึงช่วยได้น้อยที่สุดในจังหวะที่ต้องการมากที่สุด"
                if sev >= 2 else
                "ความเชื่อมั่นต่อความเสี่ยงยังดี ต้นทุนการป้องกันความเสี่ยงจึงถูก")
        out.append(Signal(
            "volatility", "ภาวะความผันผวน", state, sev,
            f"VIX อยู่ที่ {level:.1f} คิดเป็นเปอร์เซ็นไทล์ที่ {pct:.0%} "
            f"ของสามปีย้อนหลัง {note}",
            ("global-equity", "us-equity", "th-equity", "em-equity", "growth"),
           reading=f"VIX {level:.1f} ({state})",
        ))

    tnx = _row(t, "^TNX")
    if tnx is not None:
        chg3 = tnx["chg_3m"]
        level = tnx["level"]
        if pd.notna(chg3) and chg3 > 0.10:
            state, sev = "ขึ้นแรง", 3
            note = ("ตราสารหนี้อายุยาวและหุ้นอายุยาว คือกลุ่มเติบโตและเทคโนโลยี "
                    "ถูกกระทบพร้อมกัน เป็นภาวะที่พอร์ตแบบ 60/40 ไม่ช่วยป้องกันอะไร")
        elif pd.notna(chg3) and chg3 > 0.03:
            state, sev = "ขึ้น", 2
            note = ("Duration กำลังเป็นผลลบต่อส่วนตราสารหนี้ ควรตรวจว่าตราสารหนี้ "
                    "ในพอร์ตเป็นอายุยาวมากน้อยเพียงใด")
        elif pd.notna(chg3) and chg3 < -0.08:
            state, sev = "ลงแรง", 1
            note = ("อัตราคิดลดที่ลดลงเป็นผลดีทั้งกับตราสารหนี้และหุ้นเติบโต "
                    "แต่เป็นแรงหนุนที่จะไม่เกิดซ้ำไปเรื่อย ๆ")
        else:
            state, sev = "เคลื่อนไหวในกรอบ", 0
            note = ("ขณะนี้อัตราดอกเบี้ยไม่ได้ขับเคลื่อนผลตอบแทนไปทางใดทางหนึ่ง "
                    "จึงเป็นจังหวะที่ควรกำหนด duration อย่างตั้งใจ "
                    "ไม่ใช่ตามสถานการณ์")
        out.append(Signal(
            "us_rates", "ทิศทางดอกเบี้ยสหรัฐฯ", state, sev,
            f"พันธบัตรสหรัฐฯ 10 ปีอยู่ที่ {level:.2f}% เปลี่ยนแปลง {chg3:+.1%} "
            f"ในสามเดือน {note}",
            ("us-rates", "duration", "long-duration", "growth", "technology",
             "property", "infrastructure"),
           reading=f"พันธบัตรสหรัฐฯ 10 ปี {level:.2f}% ({chg3:+.1%} ใน 3 เดือน)",
        ))

    dxy = _row(t, "DX-Y.NYB")
    thb = _row(t, "THB=X")
    if thb is not None:
        chg3 = thb["chg_3m"]
        if pd.notna(chg3) and chg3 < -0.04:
            state, sev = "บาทแข็งค่า", 2
            detail = ("เงินบาทที่แข็งค่าจะลดมูลค่าในสกุลบาทของสินทรัพย์ต่างประเทศ "
                      "ที่ไม่ได้ป้องกันความเสี่ยงค่าเงินโดยกลไก "
                      "ไม่ว่าสินทรัพย์นั้นจะให้ผลตอบแทนในสกุลเดิมดีเพียงใด")
        elif pd.notna(chg3) and chg3 > 0.04:
            state, sev = "บาทอ่อนค่า", 1
            detail = ("เงินบาทที่อ่อนค่าทำให้ผลตอบแทนกองทุนต่างประเทศดูดีขึ้น "
                      "อย่าเข้าใจผิดว่าแรงหนุนจากค่าเงินคือฝีมือผู้จัดการกองทุน")
        else:
            state, sev = "บาททรงตัว", 0
            detail = "ขณะนี้ค่าเงินยังไม่ใช่ปัจจัยสำคัญที่ขับเคลื่อนผลตอบแทน"
        out.append(Signal(
            "thb", "ทิศทางค่าเงินบาท", state, sev,
            f"USD/THB อยู่ที่ {thb['level']:.2f} เปลี่ยนแปลง {chg3:+.1%} "
            f"ในสามเดือน {detail}",
            ("fx-thb", "usd", "global-equity", "us-equity", "gold"),
           reading=f"USD/THB {thb['level']:.2f} ({state})",
        ))

    if dxy is not None and pd.notna(dxy["chg_3m"]) and dxy["chg_3m"] > 0.03:
        out.append(Signal(
            "dollar", "ความแข็งค่าของดอลลาร์", "ตึงตัวขึ้น", 2,
            f"ดัชนีดอลลาร์เปลี่ยนแปลง {dxy['chg_3m']:+.1%} ในสามเดือน "
            f"ดอลลาร์ที่แข็งค่าทำให้สภาพคล่องโลกตึงตัว และในอดีตกดดัน"
            f"หุ้นตลาดเกิดใหม่และสินค้าโภคภัณฑ์",
            ("usd", "em-equity", "china", "india", "vietnam", "asia-equity",
             "commodities"),
           reading=f"ดัชนีดอลลาร์ {dxy['chg_3m']:+.1%} ใน 3 เดือน",
        ))

    gold = _row(t, "GC=F")
    if gold is not None:
        chg12, pct = gold["chg_12m"], gold["pctile_3y"]
        if pd.notna(pct) and pct > 0.95 and pd.notna(chg12) and chg12 > 0.25:
            out.append(Signal(
                "gold", "สถานะการลงทุนทองคำ", "หนาแน่น", 2,
                f"ทองคำเปลี่ยนแปลง {chg12:+.0%} ในสิบสองเดือน และอยู่ที่"
                f"เปอร์เซ็นไทล์ที่ {pct:.0%} ของกรอบสามปี สถานะที่หนาแน่น"
                f"มักคลายตัวเร็ว จึงควรกำหนดขนาดสถานะเพื่อรับการคลายตัว "
                f"ไม่ใช่เพื่อไล่ตามแนวโน้ม",
                ("gold", "safe-haven", "real-assets"),
                reading=f"ทองคำ {chg12:+.0%} ใน 12 เดือน",
            ))
        elif pd.notna(chg12) and chg12 < -0.10:
            out.append(Signal(
                "gold", "สถานะการลงทุนทองคำ", "กำลังคลายตัว", 2,
                f"ทองคำเปลี่ยนแปลง {chg12:+.0%} ในสิบสองเดือน ส่วนสินทรัพย์"
                f"ปลอดภัยจึงกลายเป็นแหล่งขาดทุน ไม่ใช่เครื่องป้องกันความเสี่ยง",
                ("gold", "safe-haven", "real-assets"),
                reading=f"ทองคำ {chg12:+.0%} ใน 12 เดือน",
            ))

    setidx = _row(t, "^SET.BK")
    if setidx is not None:
        chg12 = setidx["chg_12m"]
        if pd.notna(chg12) and chg12 < -0.08:
            state, sev = "อ่อนแรง", 2
            detail = ("หุ้นไทยอยู่ในภาวะขาดทุน พอร์ตที่กระจุกในประเทศจะไม่มี"
                      "อะไรมาถ่วงดุลหากนี่เป็นความเสี่ยงหุ้นเพียงอย่างเดียว")
        elif pd.notna(chg12) and chg12 > 0.15:
            state, sev = "แข็งแรง", 1
            detail = ("หุ้นไทยปรับขึ้นมามาก ควรตรวจว่าสัดส่วนหุ้นไทยของลูกค้า"
                      "เคลื่อนสูงกว่าเป้าหมายจากผลตอบแทนที่ได้มาหรือไม่")
        else:
            state, sev = "ทรงตัว", 0
            detail = "หุ้นไทยเคลื่อนไหวในกรอบ"
        out.append(Signal(
            "thai_equity", "แนวโน้มหุ้นไทย", state, sev,
            f"SET อยู่ที่ {setidx['level']:,.0f} เปลี่ยนแปลง {chg12:+.1%} "
            f"ในสิบสองเดือน {detail}",
            ("th-equity", "th-value", "th-growth", "financials"),
           reading=f"SET {setidx['level']:,.0f} ({chg12:+.1%} ใน 12 เดือน)",
        ))

    csi = _row(t, "000300.SS")
    if csi is not None and pd.notna(csi["chg_3m"]):
        chg3 = csi["chg_3m"]
        sev = 2 if chg3 < -0.08 else (1 if chg3 < 0 else 0)
        out.append(Signal(
            "china", "แนวโน้มหุ้นจีน",
            "อ่อนแรง" if chg3 < -0.05 else ("ฟื้นตัว" if chg3 > 0.05 else "ทรงตัว"),
            sev,
            f"CSI 300 เปลี่ยนแปลง {chg3:+.1%} ในสามเดือน การลงทุนในจีนยังเป็น"
            f"สถานะที่ขับเคลื่อนด้วยนโยบายรัฐ ไม่ใช่ปัจจัยพื้นฐาน",
            ("china", "asia-equity", "em-equity", "technology"),
           reading=f"CSI 300 {chg3:+.1%} ใน 3 เดือน",
        ))

    return sorted(out, key=lambda s: -s.severity)


# --------------------------------------------------------------------------- #
# Themes — the literacy layer, and the bridge into Parts 1 and 2
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Theme:
    key: str
    title: str
    short: str           # mid-sentence form, e.g. "the US rate path"
    caution: str         # one plain sentence: what this does to the client's money
    category: str
    tags: Tuple[str, ...]
    what: str            # what it is, in plain language
    why: str             # why it moves a portfolio
    watch: Tuple[str, ...]
    news_query: str
    signal_key: Optional[str] = None
    base_severity: int = 1


THEMES: List[Theme] = [
    Theme(
        "fed", "นโยบายการเงินสหรัฐฯ และทิศทางดอกเบี้ย", "ทิศทางดอกเบี้ยสหรัฐฯ", 
        "ถ้าดอกเบี้ยสหรัฐฯ ขึ้น "
        "ตราสารหนี้อายุยาวกับหุ้นเติบโตจะลงพร้อมกัน "
        "พอร์ตที่ถือทั้งคู่จะไม่มีตัวช่วยพยุง",
        "นโยบายการเงิน",
        ("us-rates", "duration", "long-duration", "growth", "technology",
         "us-equity", "credit-spread", "property", "infrastructure"),
        "ธนาคารกลางสหรัฐฯ (เฟด) เป็นผู้กำหนดต้นทุนของเงินให้ทั้งโลก "
        "อัตราดอกเบี้ยนโยบาย "
        "และการคาดการณ์ของตลาดว่าดอกเบี้ยจะไปทางไหนต่อ "
        "เป็นตัวกำหนดอัตราผลตอบแทนพันธบัตรสหรัฐฯ อายุ 10 ปี",
        "อัตราผลตอบแทนนั้นคืออัตราคิดลดที่ใช้ประเมินมูลค่ากระแสเงินสดระยะยาวทุกรายการ "
        "เมื่อมันสูงขึ้น ราคาตราสารหนี้ลดลง "
        "และราคาหุ้นเติบโตที่มีมูลค่าอยู่ในอนาคตไกลก็ลดลงด้วย "
        "นี่คือเหตุผลที่ปี 2022 เจ็บหนัก ตราสารหนี้และหุ้นลงพร้อมกัน "
        "พอร์ต 60/40 จึงไม่มีที่หลบ",
        ("กำหนดการประชุมเฟดและ dot plot", "ตัวเลขเงินเฟ้อพื้นฐานสหรัฐฯ", "ส่วนต่างอัตราผลตอบแทนพันธบัตร 2 ปีกับ 10 ปี"),
        "อัตราดอกเบี้ยนโยบายเฟด แนวโน้ม",
        signal_key="us_rates", base_severity=2,
    ),
    Theme(
        "bot", "นโยบาย ธปท. และเส้นอัตราผลตอบแทนในประเทศ", "ดอกเบี้ยนโยบายไทยและเส้นอัตราผลตอบแทน", 
        "ถ้าดอกเบี้ยไทยขึ้น 1% "
        "กองทุนตราสารหนี้อายุยาวอาจติดลบหลายเปอร์เซ็นต์ "
        "ขณะที่กองทุนอายุสั้นแทบไม่ขยับ",
        "นโยบายการเงิน",
        ("thb-rates", "th-govt", "duration", "long-duration", "th-credit"),
        "ธนาคารแห่งประเทศไทยกำหนดอัตราดอกเบี้ยนโยบายในประเทศ "
        "ซึ่งเป็นหลักยึดของอัตราผลตอบแทนพันธบัตรรัฐบาลและหุ้นกู้ไทย",
        "กองทุนตราสารหนี้ไทยอายุยาวอาจขาดทุนหลายเปอร์เซ็นต์เมื่อเส้นอัตราผลตอบแทนขยับ "
        "100 bp ซึ่งเป็นการขาดทุนเงินต้นจริง ไม่ใช่เพียงตัวเลขทางบัญชี "
        "ขณะที่กองทุนอายุสั้นแทบไม่ขยับ "
        "ช่องว่างระหว่างสองผลลัพธ์นี้คือการตัดสินใจที่สำคัญที่สุดในพอร์ตแบบระมัดระวังของไทย",
        ("ผลการประชุม กนง.", "เงินเฟ้อทั่วไปและเงินเฟ้อพื้นฐานของไทย", "ความต้องการในการประมูลพันธบัตรรัฐบาลไทย"),
        "กนง ดอกเบี้ยนโยบาย ธปท",
        signal_key=None, base_severity=2,
    ),
    Theme(
        "thb", "ค่าเงินบาทกับการลงทุนต่างประเทศที่ไม่ป้องกันความเสี่ยง", "ผลของค่าเงินบาทต่อสินทรัพย์ต่างประเทศ", 
        "ถ้าบาทแข็ง 5% กองทุนต่างประเทศที่ไม่ป้องกันค่าเงินจะหายไป 5% "
        "ทันที แม้ราคาสินทรัพย์ต้นทางไม่ขยับเลย",
        "ค่าเงิน",
        ("fx-thb", "usd", "global-equity", "us-equity", "gold", "em-equity"),
        "กองทุนต่างประเทศของ K-Asset ส่วนใหญ่เสนอราคาเป็นเงินบาท "
        "แต่ถือสินทรัพย์ที่ตีราคาเป็นดอลลาร์ ยูโร หรือเยน "
        "บางกองป้องกันความเสี่ยงค่าเงินกลับมาเป็นบาท แต่หลายกองไม่ได้ทำ",
        "ในกองทุนที่ไม่ป้องกันความเสี่ยง "
        "ลูกค้าถือเดิมพันสองอย่างพร้อมกัน คือตัวสินทรัพย์ และค่าเงิน "
        "หุ้นสหรัฐฯ ขึ้น 10% พร้อมกับเงินบาทแข็งค่า 10% "
        "จะหักกลบกันเหลือเกือบศูนย์ในใบแจ้งยอดของลูกค้า "
        "ลูกค้ามักไม่รู้ตัวว่ากำลังถือสถานะค่าเงินอยู่ "
        "และนี่คือสาเหตุที่พบบ่อยที่สุดของคำถามว่าทำไมกองทุนไม่ขึ้นตามตลาด",
        ("อัตราแลกเปลี่ยน USD/THB", "ดุลบัญชีเดินสะพัดและรายได้จากการท่องเที่ยว", "กองทุนต่างประเทศแต่ละกองป้องกันความเสี่ยงค่าเงินหรือไม่"),
        "ค่าเงินบาท แนวโน้ม USD THB",
        signal_key="thb", base_severity=2,
    ),
    Theme(
        "ai", "วัฏจักรการลงทุน AI และเซมิคอนดักเตอร์", "วัฏจักรการลงทุน AI และเซมิคอนดักเตอร์", 
        "หุ้นเทคโนโลยีกระจุกอยู่ในบริษัทไม่กี่แห่ง ถ้าการลงทุน AI "
        "ชะลอลง พอร์ตส่วนนี้จะลงแรงกว่าตลาดโดยรวม",
        "Thematic",
        ("technology", "semiconductor", "ai", "growth", "us-equity",
         "concentrated", "asia-equity"),
        "การลงทุนในศูนย์ข้อมูล AI "
        "กลายเป็นปัจจัยเดียวที่มีอิทธิพลต่อกำไรของหุ้นทั่วโลกมากที่สุด "
        "และกระจุกตัวอยู่ในบริษัทขนาดใหญ่มากเพียงไม่กี่แห่ง",
        "กองทุนดัชนีหุ้นทั่วโลกในวันนี้มีน้ำหนักกลุ่มเทคโนโลยีสูงกว่าเมื่อห้าปีก่อนมาก "
        "ลูกค้าที่คิดว่าตนกระจายความเสี่ยงแล้ว จึงอาจถือสถานะ AI "
        "แบบกระจุกตัวอยู่ การเพิ่มกองทุนเทคโนโลยีเฉพาะทางเข้าไปอีก "
        "เท่ากับซ้อนเดิมพันเดิมสองชั้น",
        ("แผนการลงทุนของผู้ให้บริการคลาวด์รายใหญ่", "ระยะเวลาส่งมอบชิป", "น้ำหนักกลุ่มเทคโนโลยีในกองทุนทั่วโลกที่คิดว่ากระจายแล้ว"),
        "AI semiconductor capex ลงทุน ศูนย์ข้อมูล",
        signal_key=None, base_severity=2,
    ),
    Theme(
        "china", "นโยบายจีน มาตรการกระตุ้น และปัญหาอสังหาริมทรัพย์", "ความเสี่ยงเชิงนโยบายของจีน", 
        "ราคาหุ้นจีนขึ้นกับนโยบายรัฐมากกว่าผลประกอบการ "
        "ของถูกอาจถูกได้อีกหลายปี ขนาดที่ลงทุนสำคัญกว่าจังหวะซื้อ",
        "การเติบโต",
        ("china", "asia-equity", "em-equity", "technology", "commodities"),
        "หุ้นจีนถูกขับเคลื่อนด้วยความเต็มใจของรัฐในการกระตุ้นเศรษฐกิจและท่าทีต่อภาคเอกชน "
        "มากกว่าจะเป็นกำไรของบริษัท",
        "จีนจึงเป็นสถานะเชิงนโยบาย ไม่ใช่สถานะเชิงมูลค่า "
        "ของถูกอาจถูกอยู่อย่างนั้นได้หลายปี อย่างที่เห็นในปี 2021-2022 "
        "การกำหนดขนาดสถานะจึงสำคัญกว่าราคาที่เข้าซื้อ "
        "และการถือกองทุนหุ้นจีนประเทศเดียว 50% "
        "คือการเดิมพันบนการตัดสินใจของรัฐบาลเดียว",
        ("สัญญาณจากที่ประชุมกรมการเมืองและการประชุมงานเศรษฐกิจกลาง", "เหตุการณ์ผิดนัดชำระหนี้ของผู้พัฒนาอสังหาริมทรัพย์", "กระแสเงินลงทุนระหว่างหุ้นในประเทศและนอกประเทศ"),
        "จีน มาตรการกระตุ้นเศรษฐกิจ ตลาดหุ้น",
        signal_key="china", base_severity=2,
    ),
    Theme(
        "gold", "ทองคำ อัตราดอกเบี้ยแท้จริง และความต้องการของธนาคารกลาง", "วัฏจักรทองคำ", 
        "ทองคำช่วยกระจายความเสี่ยงได้จริง แต่เคยติดลบเกิน 20% มาแล้ว "
        "และเคยลงพร้อมหุ้นในช่วงที่ตลาดขาดสภาพคล่อง",
        "สินค้าโภคภัณฑ์",
        ("gold", "safe-haven", "real-assets", "usd"),
        "ทองคำไม่มีกระแสเงินสด "
        "ราคาจึงถูกขับเคลื่อนด้วยอัตราดอกเบี้ยแท้จริง ค่าเงินดอลลาร์ "
        "และการซื้อของภาครัฐ",
        "ทองคำเป็นเครื่อง Diversify ที่แท้จริง "
        "แต่ไม่ใช่สินทรัพย์ความเสี่ยงต่ำ ทองคำเคยขาดทุนเกิน 20% "
        "หลายครั้ง และอาจลงพร้อมหุ้นในภาวะขาดสภาพคล่อง "
        "อย่างที่เกิดในเดือนมีนาคม 2020 จึงควรมองเป็น Satellite "
        "ที่ผันผวน ไม่ใช่ของแทนตราสารหนี้",
        ("อัตราดอกเบี้ยแท้จริงของสหรัฐฯ", "การซื้อทองคำเข้าทุนสำรองของธนาคารกลาง", "ปริมาณการถือครองของกองทุน ETF เพื่อวัดความหนาแน่นของสถานะ"),
        "ราคาทองคำ แนวโน้ม ธนาคารกลางซื้อทอง",
        signal_key="gold", base_severity=1,
    ),
    Theme(
        "trade", "ภาษีนำเข้า นโยบายการค้า และห่วงโซ่อุปทาน", "นโยบายภาษีนำเข้าและห่วงโซ่อุปทาน", 
        "ถ้าภาษีนำเข้ากลับมาเข้มขึ้น "
        "ผู้ส่งออกเอเชียกับหุ้นเทคโนโลยีจะโดนพร้อมกัน "
        "แม้ชื่อกองทุนจะไม่มีคำว่าเอเชียก็ตาม",
        "ภูมิรัฐศาสตร์",
        ("us-equity", "china", "asia-equity", "technology", "asean", "vietnam",
         "em-equity", "semiconductor"),
        "นโยบายภาษีนำเข้าและการควบคุมการส่งออกเป็นตัวกำหนดว่าสินค้าจะถูกผลิตที่ไหน "
        "และใครจะได้ส่วนต่างกำไร",
        "ไทยและเวียดนามอยู่ในเส้นทางการย้ายฐานการผลิตโดยตรง "
        "บางครั้งเป็นผู้ได้ประโยชน์ บางครั้งเป็นผู้เสียหายพลอยได้ "
        "ความเสี่ยงนี้ไม่ได้จำกัดอยู่แค่กองทุนที่มีคำว่าเอเชียในชื่อ "
        "กองทุนเทคโนโลยีทั่วโลกก็ถือห่วงโซ่อุปทานเดียวกัน",
        ("ประกาศขึ้นภาษีและรายการสินค้าที่ได้รับการยกเว้น", "คำสั่งซื้อเพื่อส่งออกจากไทย เวียดนาม และเกาหลี", "มาตรการควบคุมการส่งออกเซมิคอนดักเตอร์"),
        "ภาษีนำเข้า สงครามการค้า ส่งออก เอเชีย",
        signal_key=None, base_severity=2,
    ),
    Theme(
        "thai", "การเติบโตในประเทศ การเมือง และกระแสเงินลงทุนไทย", "การเติบโตและกระแสเงินในไทย", 
        "ตลาดหุ้นไทยเล็กและขึ้นกับเงินต่างชาติ ถ้าต่างชาติขาย "
        "ราคาลงต่อเนื่องได้แม้พื้นฐานบริษัทไม่ได้เปลี่ยน",
        "การเติบโต",
        ("th-equity", "th-value", "th-growth", "financials", "index",
         "small-cap", "concentrated"),
        "SET ถูกขับเคลื่อนด้วยการท่องเที่ยว การบริโภคในประเทศ "
        "เสถียรภาพทางการเมือง และกระแสเงินของนักลงทุนต่างชาติ "
        "โดยเรียงตามลำดับความสำคัญคร่าว ๆ นี้",
        "ตลาดหุ้นไทยมีขนาดเล็กและถูกขับเคลื่อนด้วยกระแสเงิน "
        "แรงขายของต่างชาติจึงสามารถครอบงำปัจจัยพื้นฐานได้เป็นเวลานาน "
        "ลูกค้าที่มีสัดส่วนหุ้นทั้งหมดเป็นหุ้นไทย "
        "จึงเผชิญความเสี่ยงจากตลาดเดียวที่เล็ก สภาพคล่องต่ำ "
        "และอ่อนไหวต่อกระแสเงิน ซึ่งคือกับดักการกระจุกตัวในประเทศ",
        ("ยอดซื้อขายสุทธิของนักลงทุนต่างชาติใน SET", "จำนวนนักท่องเที่ยวต่างชาติ", "เสถียรภาพทางการเมืองและนโยบายการคลัง"),
        "ตลาดหุ้นไทย SET นักลงทุนต่างชาติ",
        signal_key="thai_equity", base_severity=2,
    ),
    Theme(
        "credit", "ส่วนต่างอัตราผลตอบแทนและวัฏจักรหุ้นกู้", "วัฏจักรสินเชื่อ", 
        "หุ้นกู้ให้ผลตอบแทนสูงกว่าพันธบัตรเพราะมีโอกาสผิดนัดชำระ "
        "และมักราคาลงพร้อมหุ้นในจังหวะที่ต้องการที่หลบ",
        "สินเชื่อ",
        ("credit-spread", "th-credit", "asia-credit", "private-credit",
         "illiquid", "financials"),
        "หุ้นกู้ให้ผลตอบแทนสูงกว่าพันธบัตรรัฐบาล "
        "เพราะผู้ออกอาจไม่ชำระหนี้ ผลตอบแทนส่วนเกินนั้นเรียกว่า credit "
        "spread",
        "spread จะกว้างขึ้นเร็วที่สุดในจังหวะเดียวกับที่หุ้นปรับลง "
        "กองทุนหุ้นกู้จึงป้องกันความเสี่ยงในภาวะวิกฤตได้น้อยกว่ากองทุนพันธบัตรรัฐบาล "
        "ในประเทศไทยเรื่องนี้สำคัญถึงระดับผู้ออกตราสารรายตัว "
        "เพราะการผิดนัดชำระหนี้เพียงรายเดียวก็ทำให้ NAV "
        "ของกองทุนเปลี่ยนได้",
        ("ข่าวการต่ออายุและการผิดนัดชำระหุ้นกู้ไทย", "ส่วนต่างอัตราผลตอบแทนหุ้นกู้ high yield สหรัฐฯ เป็นตัวชี้วัดระดับโลก", "การกระจุกตัวของผู้ออกตราสารในกองทุนตราสารหนี้ที่ถือ"),
        "หุ้นกู้ ผิดนัดชำระหนี้ credit spread ไทย",
        signal_key=None, base_severity=1,
    ),
    Theme(
        "energy", "ราคาพลังงานและการส่งผ่านไปสู่เงินเฟ้อ", "ราคาพลังงาน", 
        "ไทยนำเข้าพลังงาน น้ำมันแพงจึงฉุดเศรษฐกิจ แต่ดันหุ้นพลังงานขึ้น "
        "กองทุนกลุ่มนี้ช่วยเฉพาะตอนเงินเฟ้อ",
        "สินค้าโภคภัณฑ์",
        ("oil", "energy", "commodities", "agriculture"),
        "น้ำมันเป็นทั้งต้นทุนของระบบเศรษฐกิจ และเป็นรายได้ของหุ้นกลุ่มใหญ่ในดัชนีไทย",
        "ไทยเป็นผู้นำเข้าพลังงานสุทธิ "
        "ราคาน้ำมันที่พุ่งขึ้นต่อเนื่องจึงเป็นภาษีต่อการเติบโตของไทย "
        "แม้จะดันหุ้นกลุ่มพลังงานใน SET ขึ้นก็ตาม "
        "กองทุนพลังงานจึงเป็นเครื่องป้องกันความเสี่ยงเฉพาะสถานการณ์เงินเฟ้อ "
        "ไม่ใช่เครื่อง Diversify แบบใช้ได้ทุกกรณี",
        ("การตัดสินใจกำลังผลิตของ OPEC+", "นโยบายอุดหนุนราคาน้ำมันของไทย", "ความแตกต่างระหว่างเงินเฟ้อทั่วไปและเงินเฟ้อพื้นฐาน"),
        "ราคาน้ำมัน OPEC เงินเฟ้อ",
        signal_key=None, base_severity=1,
    ),
    Theme(
        "japan", "ญี่ปุ่น ค่าเงินเยน และ carry trade", "yen carry trade", 
        "เมื่อญี่ปุ่นขึ้นดอกเบี้ย เงินกู้ต้นทุนต่ำทั่วโลกถูกถอนกลับ "
        "สถานะที่คนถือกันเยอะจะถูกเทขายเร็วมาก",
        "ค่าเงิน",
        ("jp-equity", "jpy", "global-equity", "growth", "technology"),
        "หลายทศวรรษที่ผ่านมา "
        "เงินเยนเป็นแหล่งกู้เพื่อไปลงทุนที่อื่นแบบใช้เลเวอเรจ "
        "เพราะดอกเบี้ยญี่ปุ่นใกล้ศูนย์ "
        "เมื่อธนาคารกลางญี่ปุ่นเริ่มปรับนโยบายเข้าสู่ภาวะปกติ "
        "แหล่งเงินนั้นก็ถูกถอนออก",
        "การคลายสถานะในเดือนสิงหาคม 2024 "
        "แสดงให้เห็นว่าผลกระทบส่งผ่านเร็วเพียงใด สถานะ momentum "
        "ที่หนาแน่นทั่วโลกปรับลงเป็นเลขสองหลักภายในสามวันทำการ "
        "โดยที่ปัจจัยพื้นฐานของบริษัทไม่ได้เปลี่ยน "
        "นี่คือความเสี่ยงด้านสภาพคล่องที่ปรากฏขึ้นแม้ในกองทุนที่ไม่มีหุ้นญี่ปุ่นเลย",
        ("การประชุมนโยบายของธนาคารกลางญี่ปุ่น", "ระดับและความเร็วของการเคลื่อนไหว USD/JPY", "ข้อมูลสถานะการลงทุนในสัญญาล่วงหน้าเงินเยน"),
        "BOJ เงินเยน carry trade นโยบาย",
        signal_key=None, base_severity=1,
    ),
    Theme(
        "frontier", "อินเดีย เวียดนาม และการเข้าถึงตลาดชายขอบ", "การเข้าถึงตลาดชายขอบ", 
        "ตลาดเวียดนามและอินเดียสภาพคล่องต่ำ เวลาตลาดตกอาจขายออกไม่ได้ในราคาที่เป็นธรรม",
        "การเติบโต",
        ("india", "vietnam", "frontier", "em-equity", "asean", "small-cap"),
        "อินเดียและเวียดนามมีเรื่องราวการเติบโตจากโครงสร้างประชากรและภาคการผลิต "
        "แต่มีสภาพคล่องต่ำกว่าตลาดพัฒนาแล้วอย่างมีนัยสำคัญ",
        "ตลาดชายขอบอาจเคลื่อนไหวรุนแรงเพียงเพราะกระแสเงิน "
        "และกฎการเข้าถึงตลาดอาจเปลี่ยนได้ โอกาสรับผลตอบแทนมีจริง "
        "แต่การกำหนดขนาดสถานะควรสะท้อนว่าลูกค้าอาจขายออกในราคาที่เป็นธรรมไม่ได้อย่างรวดเร็ว",
        ("การจัดชั้นเวียดนามเข้าสู่ดัชนีตลาดเกิดใหม่", "กำไรของบริษัทอินเดียเทียบกับระดับมูลค่า", "เพดานการถือครองของนักลงทุนต่างชาติ"),
        "เวียดนาม อินเดีย ตลาดเกิดใหม่ หุ้น",
        signal_key=None, base_severity=1,
    ),
]

THEMES_BY_KEY: Dict[str, Theme] = {t.key: t for t in THEMES}


# --------------------------------------------------------------------------- #
# News and search trends
# --------------------------------------------------------------------------- #
@dataclass
class Headline:
    title: str
    source: str
    published: Optional[datetime]
    link: str
    theme: str = ""

    @property
    def age_hours(self) -> Optional[float]:
        if self.published is None:
            return None
        now = datetime.now(timezone.utc)
        pub = self.published
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        return (now - pub).total_seconds() / 3600.0


# Thai locale: the theme queries are in Thai, so ask Google News for Thai
# coverage. Thai outlets are what an RM will actually forward to a client.
_GOOGLE_NEWS = ("https://news.google.com/rss/search?q={query}+when:{days}d"
                "&hl=th&gl=TH&ceid=TH:th")
_GOOGLE_TRENDS = "https://trends.google.com/trending/rss?geo={geo}"


def _parse_feed(url: str):
    import feedparser

    parsed = feedparser.parse(url)
    if getattr(parsed, "bozo", 0) and not parsed.entries:
        raise RuntimeError(getattr(parsed, "bozo_exception", "feed parse failed"))
    return parsed.entries


def _entry_datetime(entry) -> Optional[datetime]:
    for key in ("published_parsed", "updated_parsed"):
        value = getattr(entry, key, None)
        if value:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def fetch_news(query: str, days: int = 7, limit: int = 6,
               theme_key: str = "") -> Tuple[List[Headline], str]:
    """Headlines for one query from Google News RSS."""
    url = _GOOGLE_NEWS.format(query=re.sub(r"\s+", "+", query.strip()), days=days)
    try:
        entries = _parse_feed(url)
    except Exception as exc:
        return [], f"offline ({exc})"

    out: List[Headline] = []
    for entry in entries[:limit]:
        title = getattr(entry, "title", "").strip()
        source = ""
        if " - " in title:
            title, source = title.rsplit(" - ", 1)
        if not source:
            source = getattr(getattr(entry, "source", None), "title", "") or "Google News"
        out.append(Headline(title.strip(), source.strip(), _entry_datetime(entry),
                            getattr(entry, "link", ""), theme_key))
    return out, "live"


@dataclass
class TrendItem:
    term: str
    traffic: str
    geo: str
    finance_related: bool


# Thai has no word boundaries, so every entry here has to be long enough to be
# unambiguous on its own. Bare "ทอง" (gold) is not: it sits inside the place
# name "บางบัวทอง", which is a district, not a commodity.
_FINANCE_TERMS_TH = [
    "หุ้น", "กองทุน", "ทองคำ", "ราคาทอง", "ดอกเบี้ย", "เงินบาท", "ค่าเงินบาท",
    "ตลาดหุ้น", "ลงทุน", "บิทคอยน์", "บิตคอยน์", "คริปโต", "เศรษฐกิจ",
    "ธนาคาร", "น้ำมัน", "ดัชนี", "เงินเฟ้อ", "ค่าเงิน", "ภาษี", "อสังหา",
    "พันธบัตร", "ตราสารหนี้", "ปันผล", "งบการเงิน", "ดอลลาร์",
]

# English matches on whole words, so "rate" does not fire on "corporate" and
# "oil" does not fire on "boil".
_FINANCE_TERMS_EN = [
    "stock", "stocks", "market", "markets", "gold", "rate", "rates", "fed",
    "inflation", "bitcoin", "crypto", "bond", "bonds", "yield", "yields",
    "dollar", "baht", "oil", "recession", "tariff", "tariffs", "earnings",
    "bank", "banks", "nasdaq", "economy", "invest", "investing", "investment",
    "etf", "fund", "funds", "dividend", "portfolio", "set50",
]
_EN_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _FINANCE_TERMS_EN) + r")\b",
    re.IGNORECASE,
)


def _is_finance(term: str) -> bool:
    return bool(_EN_PATTERN.search(term)) or any(t in term for t in _FINANCE_TERMS_TH)


def fetch_trends(geo: str = "TH", limit: int = 20) -> Tuple[List[TrendItem], str]:
    """What people are searching for right now, via Google Trends RSS."""
    try:
        entries = _parse_feed(_GOOGLE_TRENDS.format(geo=geo))
    except Exception as exc:
        return [], f"offline ({exc})"

    out: List[TrendItem] = []
    for entry in entries[:limit]:
        term = getattr(entry, "title", "").strip()
        if not term:
            continue
        traffic = (getattr(entry, "ht_approx_traffic", "")
                   or getattr(entry, "approx_traffic", "") or "")
        out.append(TrendItem(term, str(traffic), geo, _is_finance(term)))
    return out, "live"


# --------------------------------------------------------------------------- #
# Assembled view
# --------------------------------------------------------------------------- #
@dataclass
class ThemeView:
    theme: Theme
    severity: int
    signal: Optional[Signal]
    headlines: List[Headline] = field(default_factory=list)

    @property
    def severity_key(self) -> str:
        """English key, for colour and tone lookups."""
        return {0: "Background", 1: "Monitor", 2: "Elevated", 3: "Acute"}[
            int(np.clip(self.severity, 0, 3))]

    @property
    def severity_label(self) -> str:
        return {0: "พื้นหลัง", 1: "เฝ้าติดตาม", 2: "สูง", 3: "รุนแรง"}[
            int(np.clip(self.severity, 0, 3))]


@dataclass
class MarketView:
    macro: MacroSnapshot
    signals: List[Signal]
    themes: List[ThemeView]
    trends: List[TrendItem]
    trends_status: str
    news_status: str
    fetched_at: datetime

    @property
    def signal_map(self) -> Dict[str, Signal]:
        return {s.key: s for s in self.signals}

    @property
    def risk_temperature(self) -> Tuple[int, str]:
        """A single 0-3 read on the market environment, with a one-line reason."""
        if not self.signals:
            return 1, "ไม่มีข้อมูลตลาดแบบเรียลไทม์ — แสดงเฉพาะมุมมองเชิงโครงสร้าง"
        worst = max(self.signals, key=lambda s: s.severity)
        avg = float(np.mean([s.severity for s in self.signals]))
        level = int(round(max(avg, worst.severity - 0.5)))
        # The reason only — every caller renders the level's own label alongside
        # it, and returning the label here too printed it twice.
        return level, f"ปัจจัยนำคือ{worst.label} ซึ่งอยู่ในภาวะ{worst.state}"


def build_market_view(
    with_news: bool = True,
    news_days: int = 7,
    news_per_theme: int = 4,
    themes: Optional[Sequence[Theme]] = None,
) -> MarketView:
    """Fetch everything Part 3 needs, degrading gracefully on network failure."""
    macro = fetch_macro()
    signals = derive_signals(macro)
    signal_map = {s.key: s for s in signals}

    chosen = list(themes) if themes else THEMES
    news_status = "skipped"
    views: List[ThemeView] = []
    for theme in chosen:
        sig = signal_map.get(theme.signal_key) if theme.signal_key else None
        # A live reading beats the static baseline in both directions: a theme
        # whose signal says "calm" must be allowed to drop down the list, or
        # every theme sits permanently at "Elevated" and the ranking is noise.
        severity = max(1, sig.severity) if sig else theme.base_severity
        headlines: List[Headline] = []
        if with_news:
            headlines, news_status = fetch_news(
                theme.news_query, days=news_days, limit=news_per_theme,
                theme_key=theme.key)
        views.append(ThemeView(theme, severity, sig, headlines))

    views.sort(key=lambda v: (-v.severity, v.theme.title))
    trends, trends_status = fetch_trends("TH")

    return MarketView(macro, signals, views, trends, trends_status, news_status,
                      datetime.now(timezone.utc))
