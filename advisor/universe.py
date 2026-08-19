"""Fund universe metadata for the K-Asset investment universe.

Every fund in ``Fund Return.xlsx`` is tagged with:

* ``sec_bucket``   — the four suitability buckets used by the SEC / AIMC
                     "asset allocation by investor type" table (Cash & short
                     term debt, Fixed income > 1y, Equity, Alternative).
                     Allocation / balanced funds are *looked through* into
                     those buckets rather than sitting in one of them.
* ``risk_level``   — the 1-8 product risk level printed on the Thai fund fact
                     sheet ("ระดับความเสี่ยงของผลิตภัณฑ์").
* ``asset_class``  — a friendlier reporting class used in the UI.
* ``region``       — geography of the underlying assets.
* ``role``         — Core or Satellite, for core-satellite construction.
* ``tags``         — free-form exposure tags consumed by the market-caution
                     engine (``advisor.cautions``) to link macro themes to the
                     holdings that actually carry the exposure.

Classification is rule-based over the fund code, with an explicit override
table for the funds whose code does not give the answer away. Fund codes
follow K-Asset naming, e.g. ``K-USA-A(A)`` = accumulating class of K USA
Equity Fund.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping

# --------------------------------------------------------------------------- #
# Buckets
# --------------------------------------------------------------------------- #
CASH = "เงินฝากและตราสารหนี้ระยะสั้น"
FIXED = "ตราสารหนี้ที่มีอายุ > 1 ปี"
EQUITY = "ตราสารทุน"
ALT = "การลงทุนทางเลือก"

SEC_BUCKETS: List[str] = [CASH, FIXED, EQUITY, ALT]

# Reporting asset classes (finer than the suitability buckets)
AC_MONEY = "ตลาดเงิน"
AC_TH_FI = "ตราสารหนี้ไทย"
AC_GL_FI = "ตราสารหนี้ต่างประเทศ"
AC_TH_EQ = "หุ้นไทย"
AC_GL_EQ = "หุ้นต่างประเทศ"
AC_ASIA_EQ = "หุ้นเอเชีย / ตลาดเกิดใหม่"
AC_SECTOR = "หุ้นกลุ่มอุตสาหกรรม / ธีมการลงทุน"
AC_ALLOC = "กองทุนผสม"
AC_ALT = "สินทรัพย์ทางเลือก"

CORE = "Core"
SATELLITE = "Satellite"


@dataclass(frozen=True)
class Fund:
    code: str
    name: str
    asset_class: str
    region: str
    risk_level: int
    role: str
    lookthrough: Mapping[str, float]  # SEC bucket -> weight, sums to 1.0
    tags: tuple = field(default_factory=tuple)
    is_rmf: bool = False
    hedged: str = "n/a"  # "hedged" | "unhedged" | "partial" | "n/a"

    @property
    def sec_bucket(self) -> str:
        """The dominant suitability bucket (for single-bucket reporting)."""
        return max(self.lookthrough.items(), key=lambda kv: kv[1])[0]

    @property
    def is_multi_asset(self) -> bool:
        return sum(1 for v in self.lookthrough.values() if v > 0.02) > 1


def _lt(cash: float = 0.0, fixed: float = 0.0, equity: float = 0.0, alt: float = 0.0):
    total = cash + fixed + equity + alt
    if total <= 0:
        raise ValueError("look-through must be positive")
    return {CASH: cash / total, FIXED: fixed / total, EQUITY: equity / total, ALT: alt / total}


PURE_CASH = _lt(cash=1.0)
PURE_FIXED = _lt(fixed=1.0)
PURE_EQUITY = _lt(equity=1.0)
PURE_ALT = _lt(alt=1.0)


# --------------------------------------------------------------------------- #
# Explicit table for every fund family in the workbook.
#
# Keys are matched against the fund code with the share-class suffix stripped,
# so ``K-USA-A(A)``, ``K-USA-A(D)`` and ``K-USARMF`` all resolve to ``K-USA``.
# Fields: (display name, asset class, region, risk level, role, look-through, tags)
# --------------------------------------------------------------------------- #
_FAMILIES: Dict[str, tuple] = {
    # ---- Money market & short-term ----------------------------------------
    "K-CASH": ("K Cash Management", AC_MONEY, "ไทย", 1, CORE, PURE_CASH, ("thb-rates",)),
    "K-MONEY": ("K Money Market", AC_MONEY, "ไทย", 1, CORE, PURE_CASH, ("thb-rates",)),
    "K-TREASURY": ("K Treasury", AC_MONEY, "ไทย", 1, CORE, PURE_CASH, ("thb-rates", "th-govt")),
    "K-SF": ("K Short Term Fixed Income", AC_TH_FI, "ไทย", 4, CORE,
             _lt(cash=0.85, fixed=0.15), ("thb-rates", "th-credit")),
    "K-SFPLUS": ("K Short Term Fixed Income Plus", AC_TH_FI, "ไทย", 4, CORE,
                 _lt(cash=0.75, fixed=0.25), ("thb-rates", "th-credit")),

    # ---- Thai fixed income -------------------------------------------------
    "K-FIXED": ("K Fixed Income", AC_TH_FI, "ไทย", 4, CORE, PURE_FIXED,
                ("thb-rates", "th-govt", "th-credit", "duration")),
    "K-FIXEDPLUS": ("K Fixed Income Plus", AC_TH_FI, "ไทย", 4, CORE, PURE_FIXED,
                    ("thb-rates", "th-credit", "duration")),
    "K-FIXEDPRO": ("K Fixed Income Pro", AC_TH_FI, "ไทย", 4, CORE, PURE_FIXED,
                   ("thb-rates", "th-govt", "duration", "long-duration")),
    "K-CBOND": ("K Corporate Bond", AC_TH_FI, "ไทย", 4, CORE, PURE_FIXED,
                ("thb-rates", "th-credit", "credit-spread")),
    "K-FITM": ("K Fixed Income Term Medium", AC_TH_FI, "ไทย", 4, CORE, PURE_FIXED,
               ("thb-rates", "duration")),
    "K-FITL": ("K Fixed Income Term Long", AC_TH_FI, "ไทย", 4, CORE, PURE_FIXED,
               ("thb-rates", "duration", "long-duration")),
    "K-FITXL": ("K Fixed Income Term Extra Long", AC_TH_FI, "ไทย", 4, CORE, PURE_FIXED,
                ("thb-rates", "duration", "long-duration")),
    "K-FI": ("K Fixed Income RMF", AC_TH_FI, "ไทย", 4, CORE, PURE_FIXED,
             ("thb-rates", "duration")),
    "K-FL": ("K Flexible RMF", AC_ALLOC, "ไทย", 5, CORE,
             _lt(cash=0.10, fixed=0.35, equity=0.55), ("th-equity", "thb-rates")),
    "K-BL": ("K Balanced RMF", AC_ALLOC, "ไทย", 5, CORE,
             _lt(cash=0.10, fixed=0.55, equity=0.35), ("th-equity", "thb-rates")),

    # ---- Global fixed income ----------------------------------------------
    "K-GB": ("K Global Bond", AC_GL_FI, "ทั่วโลก", 4, CORE, PURE_FIXED,
             ("us-rates", "credit-spread", "fx-thb", "duration")),
    "K-GDBOND": ("K Global Dynamic Bond", AC_GL_FI, "ทั่วโลก", 5, CORE, PURE_FIXED,
                 ("us-rates", "credit-spread", "fx-thb", "duration")),
    "K-GINCOME": ("K Global Income", AC_ALLOC, "ทั่วโลก", 5, CORE,
                  _lt(fixed=0.55, equity=0.35, alt=0.10),
                  ("us-rates", "credit-spread", "global-equity", "fx-thb")),

    # ---- Thai equity -------------------------------------------------------
    "K-EQUITY": ("K Equity", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY, ("th-equity",)),
    "K-EQ": ("K Equity (Accum.)", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY, ("th-equity",)),
    "K-EQD": ("K Equity Dividend", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY, ("th-equity",)),
    "K-STAR": ("K Stock Advance Return", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY,
               ("th-equity", "th-growth")),
    "K-VALUE": ("K Value", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY,
                ("th-equity", "th-value")),
    "K-GROWTH": ("K Growth", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY,
                 ("th-equity", "th-growth")),
    "K-SELECT": ("K Select", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY, ("th-equity",)),
    "K-20SELECT": ("K 20 Select", AC_TH_EQ, "ไทย", 6, SATELLITE, PURE_EQUITY,
                   ("th-equity", "concentrated")),
    "K-MIDSMALL": ("K Mid Small Cap Equity", AC_TH_EQ, "ไทย", 6, SATELLITE, PURE_EQUITY,
                   ("th-equity", "small-cap")),
    "K-SET50": ("K SET50 Index", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY,
                ("th-equity", "index")),
    "K-S50": ("K SET50 Index (Class A)", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY,
              ("th-equity", "index")),
    "K-MV": ("K Minimum Volatility Equity", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY,
             ("th-equity", "low-vol")),
    "K-MS": ("K Multi Strategy Equity", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY, ("th-equity",)),
    "K-ESGSI": ("K ESG Sustainable Investing", AC_TH_EQ, "ไทย", 6, SATELLITE, PURE_EQUITY,
                ("th-equity", "esg")),
    "K-THAICG": ("K Thai CG RMF", AC_TH_EQ, "ไทย", 6, CORE, PURE_EQUITY,
                 ("th-equity", "esg")),
    "K-TNZ": ("K Thailand Net Zero", AC_TH_EQ, "ไทย", 6, SATELLITE, PURE_EQUITY,
              ("th-equity", "esg", "energy-transition")),
    "K-BANKING": ("K Banking Sector", AC_SECTOR, "ไทย", 7, SATELLITE, PURE_EQUITY,
                  ("th-equity", "financials", "credit-spread")),
    "K-ICT": ("K ICT Sector", AC_SECTOR, "ไทย", 7, SATELLITE, PURE_EQUITY,
              ("th-equity", "technology")),
    "K-ENERGY": ("K Energy Sector", AC_SECTOR, "ไทย", 7, SATELLITE, PURE_EQUITY,
                 ("th-equity", "energy", "oil")),

    # ---- Global / DM equity -------------------------------------------------
    "K-USA": ("K US Equity", AC_GL_EQ, "สหรัฐฯ", 6, CORE, PURE_EQUITY,
              ("us-equity", "global-equity", "fx-thb", "growth")),
    "K-US500X": ("K US 500 Index", AC_GL_EQ, "สหรัฐฯ", 6, CORE, PURE_EQUITY,
                 ("us-equity", "global-equity", "index", "fx-thb")),
    "K-USXNDQ": ("K US Nasdaq 100 Index", AC_SECTOR, "สหรัฐฯ", 7, SATELLITE, PURE_EQUITY,
                 ("us-equity", "technology", "growth", "fx-thb", "concentrated")),
    "K-EUROPE": ("K Europe Equity", AC_GL_EQ, "ยุโรป", 6, CORE, PURE_EQUITY,
                 ("eu-equity", "global-equity", "fx-thb")),
    "K-EUX": ("K Europe Index", AC_GL_EQ, "ยุโรป", 6, CORE, PURE_EQUITY,
              ("eu-equity", "global-equity", "index", "fx-thb")),
    "K-EU": ("K Europe RMF", AC_GL_EQ, "ยุโรป", 6, CORE, PURE_EQUITY,
             ("eu-equity", "global-equity", "fx-thb")),
    "K-EUSMALL": ("K Europe Small Cap", AC_GL_EQ, "ยุโรป", 6, SATELLITE, PURE_EQUITY,
                  ("eu-equity", "small-cap", "fx-thb")),
    "K-JP": ("K Japan Equity", AC_GL_EQ, "ญี่ปุ่น", 6, CORE, PURE_EQUITY,
             ("jp-equity", "global-equity", "fx-thb", "jpy")),
    "K-JPX": ("K Japan Index", AC_GL_EQ, "ญี่ปุ่น", 6, CORE, PURE_EQUITY,
              ("jp-equity", "global-equity", "index", "fx-thb", "jpy")),
    "K-WORLDX": ("K World Index", AC_GL_EQ, "ทั่วโลก", 6, CORE, PURE_EQUITY,
                 ("global-equity", "us-equity", "index", "fx-thb")),
    "K-GLOBE": ("K Global Equity", AC_GL_EQ, "ทั่วโลก", 6, CORE, PURE_EQUITY,
                ("global-equity", "fx-thb")),
    "K-GSELECT": ("K Global Select Equity", AC_GL_EQ, "ทั่วโลก", 6, CORE, PURE_EQUITY,
                  ("global-equity", "us-equity", "fx-thb")),
    "K-GSELECTU": ("K Global Select Equity (Unhedged)", AC_GL_EQ, "ทั่วโลก", 6, CORE, PURE_EQUITY,
                   ("global-equity", "us-equity", "fx-thb")),
    "K-GSF": ("K Global Select Fund (UH)", AC_GL_EQ, "ทั่วโลก", 6, CORE, PURE_EQUITY,
              ("global-equity", "fx-thb")),
    "K-GNEXT": ("K Global Next Generation", AC_SECTOR, "ทั่วโลก", 7, SATELLITE, PURE_EQUITY,
                ("global-equity", "technology", "growth", "fx-thb")),
    "K-GEMO": ("K Global Emerging Market Opportunity", AC_ASIA_EQ, "ตลาดเกิดใหม่", 6,
               SATELLITE, PURE_EQUITY, ("em-equity", "fx-thb", "usd")),

    # ---- Asia / EM equity ---------------------------------------------------
    "K-CHINA": ("K China Equity", AC_ASIA_EQ, "จีน", 6, SATELLITE, PURE_EQUITY,
                ("china", "em-equity", "fx-thb")),
    "K-CHX": ("K China Index", AC_ASIA_EQ, "จีน", 6, SATELLITE, PURE_EQUITY,
              ("china", "em-equity", "index", "fx-thb")),
    "K-CCTV": ("K China Continued Value", AC_ASIA_EQ, "จีน", 6, SATELLITE, PURE_EQUITY,
               ("china", "em-equity", "fx-thb")),
    "K-INDIA": ("K India Equity", AC_ASIA_EQ, "อินเดีย", 6, SATELLITE, PURE_EQUITY,
                ("india", "em-equity", "fx-thb")),
    "K-INDX": ("K India Index", AC_ASIA_EQ, "อินเดีย", 6, SATELLITE, PURE_EQUITY,
               ("india", "em-equity", "index", "fx-thb")),
    "K-VIETNAM": ("K Vietnam Equity", AC_ASIA_EQ, "เวียดนาม", 6, SATELLITE, PURE_EQUITY,
                  ("vietnam", "frontier", "em-equity", "fx-thb")),
    "K-ASIA": ("K Asia Pacific Equity", AC_ASIA_EQ, "เอเชียแปซิฟิก", 6, CORE, PURE_EQUITY,
               ("asia-equity", "em-equity", "fx-thb")),
    "K-ASIAX": ("K Asia ex-Japan Index", AC_ASIA_EQ, "เอเชีย (ไม่รวมญี่ปุ่น)", 6, CORE, PURE_EQUITY,
                ("asia-equity", "em-equity", "index", "fx-thb")),
    "K-ASIACV": ("K Asia Continued Value", AC_ASIA_EQ, "เอเชีย", 6, SATELLITE, PURE_EQUITY,
                 ("asia-equity", "em-equity", "fx-thb")),
    "K-AEC": ("K AEC Equity", AC_ASIA_EQ, "อาเซียน", 6, SATELLITE, PURE_EQUITY,
              ("asean", "em-equity", "fx-thb")),
    "K-APB": ("K Asia Pacific Bond", AC_GL_FI, "เอเชียแปซิฟิก", 5, SATELLITE, PURE_FIXED,
              ("asia-credit", "credit-spread", "fx-thb", "us-rates")),

    # ---- Global sector & thematic -------------------------------------------
    "K-ATECH": ("K Asia Technology", AC_SECTOR, "เอเชีย", 7, SATELLITE, PURE_EQUITY,
                ("technology", "semiconductor", "asia-equity", "china", "ai", "fx-thb")),
    "K-GTECH": ("K Global Technology", AC_SECTOR, "ทั่วโลก", 7, SATELLITE, PURE_EQUITY,
                ("technology", "semiconductor", "us-equity", "ai", "growth", "fx-thb")),
    "K-SEMQ": ("K Semiconductor Quality", AC_SECTOR, "ทั่วโลก", 7, SATELLITE, PURE_EQUITY,
               ("technology", "semiconductor", "ai", "us-equity", "fx-thb", "concentrated")),
    "K-GHEALTH": ("K Global Healthcare", AC_SECTOR, "ทั่วโลก", 7, SATELLITE, PURE_EQUITY,
                  ("healthcare", "us-equity", "fx-thb")),
    "K-GH": ("K Global Healthcare RMF", AC_SECTOR, "ทั่วโลก", 7, SATELLITE, PURE_EQUITY,
             ("healthcare", "us-equity", "fx-thb")),
    "K-CHANGE": ("K Positive Change Equity", AC_SECTOR, "ทั่วโลก", 7, SATELLITE, PURE_EQUITY,
                 ("global-equity", "esg", "growth", "fx-thb")),
    "K-PLANET": ("K Planetary Transition", AC_SECTOR, "ทั่วโลก", 7, SATELLITE, PURE_EQUITY,
                 ("esg", "energy-transition", "growth", "fx-thb")),
    "K-AGRI": ("K Agriculture", AC_SECTOR, "ทั่วโลก", 7, SATELLITE, PURE_EQUITY,
               ("commodities", "agriculture", "fx-thb")),

    # ---- Alternatives --------------------------------------------------------
    "K-GOLD": ("K Gold", AC_ALT, "ทั่วโลก", 8, SATELLITE, PURE_ALT,
               ("gold", "real-assets", "usd", "fx-thb", "safe-haven")),
    "K-GD": ("K Gold RMF", AC_ALT, "ทั่วโลก", 8, SATELLITE, PURE_ALT,
             ("gold", "real-assets", "usd", "fx-thb", "safe-haven")),
    "K-OIL": ("K Oil", AC_ALT, "ทั่วโลก", 8, SATELLITE, PURE_ALT,
              ("oil", "commodities", "energy", "usd")),
    "K-PROPI": ("K Property Infra Flexible", AC_ALT, "ไทย", 8, SATELLITE, PURE_ALT,
                ("property", "real-assets", "thb-rates", "duration")),
    "K-PROPI-RMF": ("K Property Infra RMF", AC_ALT, "ไทย", 8, SATELLITE, PURE_ALT,
                    ("property", "real-assets", "thb-rates")),
    "K-GPROP": ("K Global Property", AC_ALT, "ทั่วโลก", 8, SATELLITE, PURE_ALT,
                ("property", "real-assets", "us-rates", "duration", "fx-thb")),
    "K-GINFRA": ("K Global Infrastructure", AC_ALT, "ทั่วโลก", 8, SATELLITE, PURE_ALT,
                 ("infrastructure", "real-assets", "us-rates", "fx-thb")),

    # ---- Multi-asset / allocation -------------------------------------------
    "K-GA": ("K Global Allocation", AC_ALLOC, "ทั่วโลก", 5, CORE,
             _lt(cash=0.05, fixed=0.40, equity=0.50, alt=0.05),
             ("global-equity", "us-rates", "fx-thb")),
    "K-7030": ("K 70/30 Balanced", AC_ALLOC, "ไทย", 5, CORE,
               _lt(cash=0.05, fixed=0.65, equity=0.30),
               ("th-equity", "thb-rates")),
    "K-PLAN1": ("K Plan 1 (Conservative)", AC_ALLOC, "ไทย", 5, CORE,
                _lt(cash=0.15, fixed=0.75, equity=0.10), ("thb-rates", "th-equity")),
    "K-PLAN2": ("K Plan 2 (Moderate)", AC_ALLOC, "ไทย", 5, CORE,
                _lt(cash=0.10, fixed=0.60, equity=0.30), ("thb-rates", "th-equity")),
    "K-PLAN3": ("K Plan 3 (Aggressive)", AC_ALLOC, "ไทย", 5, CORE,
                _lt(cash=0.05, fixed=0.45, equity=0.50), ("thb-rates", "th-equity")),
    "K-2035": ("K Target Date 2035 RMF", AC_ALLOC, "ทั่วโลก", 5, CORE,
               _lt(cash=0.05, fixed=0.45, equity=0.45, alt=0.05),
               ("global-equity", "thb-rates")),
    "K-2040": ("K Target Date 2040 RMF", AC_ALLOC, "ทั่วโลก", 5, CORE,
               _lt(cash=0.05, fixed=0.30, equity=0.60, alt=0.05),
               ("global-equity", "thb-rates")),

    # ---- WealthPLUS glide-path series ---------------------------------------
    "K-WPLIGHT": ("K WealthPLUS Light", AC_ALLOC, "ทั่วโลก", 5, CORE,
                  _lt(cash=0.10, fixed=0.75, equity=0.13, alt=0.02),
                  ("global-equity", "us-rates", "thb-rates", "fx-thb")),
    "K-WPSPARK": ("K WealthPLUS Spark", AC_ALLOC, "ทั่วโลก", 5, CORE,
                  _lt(cash=0.07, fixed=0.63, equity=0.27, alt=0.03),
                  ("global-equity", "us-rates", "thb-rates", "fx-thb")),
    "K-WPBALANCED": ("K WealthPLUS Balanced", AC_ALLOC, "ทั่วโลก", 5, CORE,
                     _lt(cash=0.05, fixed=0.47, equity=0.44, alt=0.04),
                     ("global-equity", "us-equity", "us-rates", "fx-thb")),
    "K-WPBAL": ("K WealthPLUS Balanced RMF", AC_ALLOC, "ทั่วโลก", 5, CORE,
                _lt(cash=0.05, fixed=0.47, equity=0.44, alt=0.04),
                ("global-equity", "us-rates", "fx-thb")),
    "K-WPSPEEDUP": ("K WealthPLUS SpeedUp", AC_ALLOC, "ทั่วโลก", 5, CORE,
                    _lt(cash=0.03, fixed=0.26, equity=0.66, alt=0.05),
                    ("global-equity", "us-equity", "us-rates", "fx-thb")),
    "K-WPSPEED": ("K WealthPLUS SpeedUp RMF", AC_ALLOC, "ทั่วโลก", 5, CORE,
                  _lt(cash=0.03, fixed=0.26, equity=0.66, alt=0.05),
                  ("global-equity", "us-equity", "fx-thb")),
    "K-WPULTIMATE": ("K WealthPLUS Ultimate", AC_ALLOC, "ทั่วโลก", 5, CORE,
                     _lt(cash=0.02, fixed=0.08, equity=0.85, alt=0.05),
                     ("global-equity", "us-equity", "fx-thb")),
    "K-WPULTI": ("K WealthPLUS Ultimate RMF", AC_ALLOC, "ทั่วโลก", 5, CORE,
                 _lt(cash=0.02, fixed=0.08, equity=0.85, alt=0.05),
                 ("global-equity", "us-equity", "fx-thb")),

    # ---- Misc ----------------------------------------------------------------
    "K-GPIN": ("K Global Private Income", AC_ALT, "ทั่วโลก", 8, SATELLITE, PURE_ALT,
               ("private-credit", "credit-spread", "us-rates", "fx-thb", "illiquid")),
    "K-GPINUH": ("K Global Private Income (UH)", AC_ALT, "ทั่วโลก", 8, SATELLITE, PURE_ALT,
                 ("private-credit", "credit-spread", "us-rates", "fx-thb", "illiquid")),
    "K-GIF": ("K Global Income RMF", AC_ALLOC, "ทั่วโลก", 5, CORE,
              _lt(fixed=0.55, equity=0.35, alt=0.10),
              ("us-rates", "credit-spread", "global-equity", "fx-thb")),
    "K-GlNCOME": ("K Global Income RMF", AC_ALLOC, "ทั่วโลก", 5, CORE,
                  _lt(fixed=0.55, equity=0.35, alt=0.10),
                  ("us-rates", "credit-spread", "global-equity", "fx-thb")),
}

# Codes whose family key is not simply the prefix (odd names in the workbook).
_ALIASES: Dict[str, str] = {
    "K-GDBONDRMF": "K-GDBOND",
    "K-GDBONDUH": "K-GDBOND",
    "K-GDRMF": "K-GD",
    "K-GHRMF": "K-GH",
    "K-GHEALTH(UH)": "K-GHEALTH",
    "K-GSF(UH)": "K-GSF",
    "K-GlNCOMERMF": "K-GlNCOME",
    "K-GIFRMF": "K-GIF",
    "K-PROPIRMF": "K-PROPI-RMF",
    "K-FLRMF": "K-FL",
    "K-BLRMF": "K-BL",
    "K-FIRMF": "K-FI",
    "K-GBRMF": "K-GB",
    "K-GARMF": "K-GA",
    "K-EURMF": "K-EU",
    "K-SFRMF": "K-SF",
    "K-S50RMF": "K-S50",
    "K-MSRMF": "K-MS",
    "K-JPRMF": "K-JP",
    "K-CHANGERMF": "K-CHANGE",
    "K-CHINARMF": "K-CHINA",
    "K-INDIARMF": "K-INDIA",
    "K-VIETNAMRMF": "K-VIETNAM",
    "K-PLANETRMF": "K-PLANET",
    "K-STARRMF": "K-STAR",
    "K-THAICGRMF": "K-THAICG",
    "K-USARMF": "K-USA",
    "K-US500XRMF": "K-US500X",
    "K-US500XUH": "K-US500X",
    "K-USXNDQRMF": "K-USXNDQ",
    "K-USXNDQUH": "K-USXNDQ",
    "K-WORLDXRMF": "K-WORLDX",
    "K-GSELECTRMF": "K-GSELECT",
    "K-GTECHRMF": "K-GTECH",
    "K-WPBALRMF": "K-WPBAL",
    "K-WPLIGHTRMF": "K-WPLIGHT",
    "K-WPSPEEDRMF": "K-WPSPEED",
    "K-WPULTIRMF": "K-WPULTI",
    "K-2035RMF": "K-2035",
    "K-2040RMF": "K-2040",
    "K-PROPI-A(D)": "K-PROPI",
}

_SHARE_CLASS_RE = re.compile(r"-(?:A|T|R|C|I|SSF|SSFX)\(?[ADR]?\)?$", re.IGNORECASE)


def _family_key(code: str) -> str:
    """Strip the share-class suffix and resolve aliases to a family key."""
    if code in _ALIASES:
        return _ALIASES[code]
    base = _SHARE_CLASS_RE.sub("", code)
    if base in _FAMILIES:
        return base
    if base.endswith("RMF") and base[:-3] in _FAMILIES:
        return base[:-3]
    if code in _FAMILIES:
        return code
    # Longest-prefix fallback so an unseen share class still lands somewhere sane.
    candidates = [k for k in _FAMILIES if base.startswith(k)]
    if candidates:
        return max(candidates, key=len)
    return ""


def _share_class(code: str) -> str:
    """Thai fund tables label these สะสมมูลค่า (accumulating) / จ่ายปันผล (dividend)."""
    if code.endswith("(A)"):
        return "สะสมมูลค่า"
    if code.endswith("(D)"):
        return "จ่ายปันผล"
    return "Standard"


def _hedge_flag(code: str) -> str:
    if "UH" in code or "(UH)" in code:
        return "unhedged"
    return "n/a"


def build_fund(code: str) -> Fund:
    """Resolve a workbook column name into a :class:`Fund`."""
    key = _family_key(code)
    if not key:
        # Unknown fund: default to a diversified global equity satellite so it
        # can never sneak into a conservative mandate by accident.
        return Fund(code, code, AC_GL_EQ, "ทั่วโลก", 6, SATELLITE, PURE_EQUITY, ("unclassified",))

    name, ac, region, lvl, role, lt, tags = _FAMILIES[key]
    is_rmf = "RMF" in code.upper()
    display = name
    sc = _share_class(code)
    if sc != "Standard":
        display = f"{name} ({sc})"
    if is_rmf and "RMF" not in name:
        display = f"{name} RMF"
    return Fund(
        code=code,
        name=display,
        asset_class=ac,
        region=region,
        risk_level=lvl,
        role=role,
        lookthrough=lt,
        tags=tuple(tags),
        is_rmf=is_rmf,
        hedged=_hedge_flag(code),
    )


def build_universe(codes) -> Dict[str, Fund]:
    return {c: build_fund(c) for c in codes}


# --------------------------------------------------------------------------- #
# Short aliases used in the client portfolio definitions
# --------------------------------------------------------------------------- #
ALIAS_TO_CODE: Dict[str, str] = {
    "ksfplus": "K-SFPLUS-A",
    "kfixedpro": "K-FIXEDPRO",
    "kfixedplus": "K-FIXEDPLUS-A",
    "kfixed": "K-FIXED-A",
    "kwealthplus speedup": "K-WPSPEEDUP",
    "wealthplus balanced": "K-WPBALANCED",
    "kgtech": "K-GTECH",
    "katech": "K-ATECH",
    "kgold": "K-GOLD-A(A)",
    "kvalue": "K-VALUE-A(D)",
    "kchina": "K-CHINA-A(A)",
    "kstar": "K-STAR-A(A)",
    "kusa": "K-USA-A(A)",
    "kviet": "K-VIETNAM",
}


# --------------------------------------------------------------------------- #
# Factor proxies — long-history funds used to stand in for a whole asset class
# when a fund did not exist during a historical stress window.
# --------------------------------------------------------------------------- #
PROXY_BY_CLASS: Dict[str, str] = {
    AC_MONEY: "K-TREASURY",
    AC_TH_FI: "K-FIXED-A",
    AC_GL_FI: "K-GBRMF",
    AC_TH_EQ: "K-EQUITY",
    AC_GL_EQ: "K-GLOBE",
    AC_ASIA_EQ: "K-GLOBE",
    AC_SECTOR: "K-GLOBE",
    AC_ALLOC: "K-GA-A(D)",
    AC_ALT: "K-GOLD-A(D)",
}

# Factor model basis: (label, fund code used as the factor return series)
FACTOR_PROXIES: Dict[str, str] = {
    "หุ้นไทย": "K-EQUITY",
    "หุ้นต่างประเทศ": "K-GLOBE",
    "ดอกเบี้ยไทย (duration)": "K-FIXED-A",
    "ทองคำ": "K-GOLD-A(D)",
    "น้ำมัน": "K-OIL",
}
