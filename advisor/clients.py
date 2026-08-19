"""Client book and the SEC suitability framework that governs it.

Two regulatory tables drive everything downstream:

1. **Suitability score → acceptable product risk levels.** A client's risk
   profile number is the *highest product risk level* they may hold, so a
   profile-5 client may hold funds of level 1-5 and nothing above.

2. **Investor type → asset-allocation band.** For each investor type the SEC
   /AIMC suitability form prints a suggested allocation across four buckets:
   deposits & short-term debt, fixed income over one year, equity, and
   alternatives. Those bands are encoded here as hard constraints for the
   optimiser and as a compliance check for the existing book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from .universe import ALIAS_TO_CODE, ALT, CASH, EQUITY, FIXED, SEC_BUCKETS


# --------------------------------------------------------------------------- #
# Risk profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskProfile:
    """One row of the suitability table."""

    level: int                       # highest acceptable product risk level (1-8)
    name_en: str
    name_th: str
    score_range: str
    bands: Mapping[str, Tuple[float, float]]   # bucket -> (min, max) weight
    description: str

    def band(self, bucket: str) -> Tuple[float, float]:
        return self.bands.get(bucket, (0.0, 1.0))

    @property
    def acceptable_levels(self) -> str:
        return f"ระดับ 1 – {self.level}"


# Bands transcribed from the SEC suitability "asset allocation by investor
# type" table. Government and corporate bonds over one year share a single cap
# in that table, so they are combined into one FIXED bucket here.
#
# The form states caps ("< 20%") for most buckets and a floor only where it
# prints "> 60%". Floors are therefore left at zero except where the form gives
# one: the caps already imply a floor by arithmetic (profile 5 caps sum to
# 110%, so fixed income cannot fall below 50%), and encoding an invented floor
# would flag portfolios the regulator would not.
RISK_PROFILES: Dict[int, RiskProfile] = {
    1: RiskProfile(
        level=1,
        name_en="เสี่ยงต่ำ (Low risk)",
        name_th="เสี่ยงต่ำ",
        score_range="ต่ำกว่า 15",
        bands={CASH: (0.60, 1.00), FIXED: (0.00, 0.40), EQUITY: (0.00, 0.10), ALT: (0.00, 0.05)},
        description="รักษาเงินต้นเป็นหลัก น้ำหนักส่วนใหญ่อยู่ในเงินฝากและตราสารหนี้"
                    "ระยะสั้น หุ้นเป็นเพียงส่วนประกอบเล็กน้อย",
    ),
    4: RiskProfile(
        level=4,
        name_en="เสี่ยงปานกลางค่อนข้างต่ำ (Low to moderate)",
        name_th="เสี่ยงปานกลางค่อนข้างต่ำ",
        score_range="15 – 21",
        bands={CASH: (0.00, 0.20), FIXED: (0.00, 0.70), EQUITY: (0.00, 0.20), ALT: (0.00, 0.10)},
        description="เน้นรายได้สม่ำเสมอ ตราสารหนี้เป็นแกนของพอร์ต และมีสัดส่วนหุ้น"
                    "ไม่มากเพื่อเอาชนะเงินเฟ้อ",
    ),
    5: RiskProfile(
        level=5,
        name_en="เสี่ยงปานกลางค่อนข้างสูง (Moderate to moderately high)",
        name_th="เสี่ยงปานกลางค่อนข้างสูง",
        score_range="22 – 29",
        bands={CASH: (0.00, 0.10), FIXED: (0.00, 0.60), EQUITY: (0.00, 0.30), ALT: (0.00, 0.10)},
        description="สมดุล ยอมรับการขาดทุนระหว่างทางได้เพื่อการเติบโตที่แท้จริง "
                    "แต่ตราสารหนี้ยังเป็นฐานของพอร์ต",
    ),
    7: RiskProfile(
        level=7,
        name_en="เสี่ยงสูง (High risk)",
        name_th="เสี่ยงสูง",
        score_range="30 – 36",
        bands={CASH: (0.00, 0.10), FIXED: (0.00, 0.40), EQUITY: (0.00, 0.40), ALT: (0.00, 0.20)},
        description="เน้นการเติบโต ลงทุนกองทุนรายกลุ่มอุตสาหกรรมและธีมได้ "
                    "ส่วนตราสารหนี้มีไว้รองรับช่วงตลาดขาลง ไม่ใช่เพื่อสร้างผลตอบแทน",
    ),
    8: RiskProfile(
        level=8,
        name_en="เสี่ยงสูงมาก (Very high risk)",
        name_th="เสี่ยงสูงมาก",
        score_range="37 ขึ้นไป",
        bands={CASH: (0.00, 0.05), FIXED: (0.00, 0.30), EQUITY: (0.60, 1.00), ALT: (0.00, 0.30)},
        description="มุ่งการเติบโตสูงสุด ตราสารทุนต้องเป็นสัดส่วนหลักของพอร์ต "
                    "และสินทรัพย์ทางเลือกมีบทบาทเป็นส่วนเสริมขนาดใหญ่ได้",
    ),
}

# Levels 2, 3 and 6 are not used by the current book but the form defines them;
# they inherit the band of the nearest defined profile so the app never breaks
# if an RM types one in.
for _lvl, _src in ((2, 1), (3, 4), (6, 5)):
    _base = RISK_PROFILES[_src]
    RISK_PROFILES[_lvl] = RiskProfile(
        level=_lvl,
        name_en=_base.name_en,
        name_th=_base.name_th,
        score_range=_base.score_range,
        bands=_base.bands,
        description=_base.description,
    )


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
@dataclass
class Client:
    id: str
    name: str
    risk_profile: int
    holdings: Dict[str, float]         # fund code -> weight (sums to 1.0)
    aum_thb: float
    persona: str
    objective: str
    horizon_years: int
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def profile(self) -> RiskProfile:
        return RISK_PROFILES[self.risk_profile]

    @property
    def codes(self) -> List[str]:
        return list(self.holdings)

    @property
    def label(self) -> str:
        return f"{self.id} · {self.name}"


def _h(**kwargs: float) -> Dict[str, float]:
    """Build a holdings dict from short aliases, validating the weights sum."""
    out: Dict[str, float] = {}
    for alias, weight in kwargs.items():
        key = alias.replace("_", " ")
        code = ALIAS_TO_CODE.get(key)
        if code is None:
            raise KeyError(f"unknown fund alias: {alias!r}")
        out[code] = out.get(code, 0.0) + weight
    total = sum(out.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"holdings sum to {total:.4f}, expected 1.0 ({kwargs})")
    return out


CLIENTS: List[Client] = [
    # ---- Risk profile 4 ----------------------------------------------------
    Client(
        id="C01",
        name="คุณสมชาย ต.",
        risk_profile=4,
        holdings=_h(ksfplus=1.00),
        aum_thb=25_000_000,
        persona="เจ้าของธุรกิจเกษียณแล้ว อายุ 68",
        objective="รักษาเงินต้น ให้ผลตอบแทนชนะดอกเบี้ยเงินฝาก และคงสภาพคล่อง",
        horizon_years=3,
        notes="พักเงินทั้งหมดในกองทุนตราสารหนี้ระยะสั้นหลังขายธุรกิจครอบครัว "
              "เลือกความเสี่ยงต่ำด้วยตนเอง แต่พอร์ตทั้งหมดอยู่ในกองทุนเดียว",
        tags=["single-fund", "liquidity"],
    ),
    Client(
        id="C02",
        name="คุณวนิดา พ.",
        risk_profile=4,
        holdings=_h(kfixedpro=1.00),
        aum_thb=40_000_000,
        persona="ผู้บริหารระดับสูง อายุ 55",
        objective="ล็อกอัตราผลตอบแทนไว้ก่อนเกษียณในอีก 10 ปี",
        horizon_years=10,
        notes="ลงทุนเต็มจำนวนในพันธบัตรรัฐบาลไทยอายุยาว เป็นการเดิมพันบน duration "
              "เพียงอย่างเดียว ไม่มีการกระจายความเสี่ยงรองรับหากดอกเบี้ยขึ้นแรง",
        tags=["single-fund", "duration-risk"],
    ),

    # ---- Risk profile 5 ----------------------------------------------------
    Client(
        id="C03",
        name="คุณอนันต์ ก.",
        risk_profile=5,
        holdings=_h(kwealthplus_speedup=1.00),
        aum_thb=18_000_000,
        persona="ผู้ก่อตั้งบริษัทเทคโนโลยี อายุ 41",
        objective="การเติบโตระยะยาวผ่านกองทุนสำเร็จรูปกองเดียว",
        horizon_years=15,
        notes="กองทุนผสมทั่วโลกแบบครบในกองเดียว เรียบง่ายและกระจายความเสี่ยงดี "
              "แต่ลูกค้าไม่มีสิทธิกำหนดเส้นทางการปรับสัดส่วนของกองทุน",
        tags=["one-ticket"],
    ),
    Client(
        id="C04",
        name="คุณพิมพ์ชนก ส.",
        risk_profile=5,
        holdings=_h(wealthplus_balanced=0.70, kgtech=0.10, katech=0.10, kgold=0.10),
        aum_thb=60_000_000,
        persona="ผู้พัฒนาอสังหาริมทรัพย์ อายุ 48",
        objective="แกนพอร์ตแบบสมดุล เสริมด้วยธีมการลงทุนเพื่อเพิ่มโอกาสรับผลตอบแทน",
        horizon_years=10,
        notes="โครงสร้างเป็นแบบ core-satellite แต่ส่วนเสริมสองกองเป็นกองทุนรายกลุ่ม"
              "อุตสาหกรรมระดับ 7 และอีกหนึ่งกองระดับ 8 ซึ่งสูงกว่าระดับความเสี่ยง"
              "ที่ลูกค้ารับได้",
        tags=["core-satellite", "thematic-tilt"],
    ),
    Client(
        id="C05",
        name="คุณกฤต ว.",
        risk_profile=5,
        holdings=_h(ksfplus=0.50, kvalue=0.15, kgold=0.05, kfixed=0.20, kchina=0.10),
        aum_thb=32_000_000,
        persona="ประธานเจ้าหน้าที่การลงทุน Family Office อายุ 52",
        objective="การเติบโตระดับกลาง พร้อมกันชนสภาพคล่องขนาดใหญ่",
        horizon_years=8,
        notes="ครึ่งหนึ่งของพอร์ตอยู่ในสินทรัพย์คล้ายเงินสดอายุสั้น ซึ่งสูงกว่าเพดาน "
              "10% ของกลุ่มเงินฝากและตราสารหนี้ระยะสั้นสำหรับระดับความเสี่ยงนี้อย่างมาก",
        tags=["cash-heavy", "diversified"],
    ),

    # ---- Risk profile 7 ----------------------------------------------------
    Client(
        id="C06",
        name="คุณณัฐพงษ์ ร.",
        risk_profile=7,
        holdings=_h(kchina=0.50, katech=0.20, kfixedplus=0.20, kstar=0.10),
        aum_thb=85_000_000,
        persona="เจ้าของโรงงานผลิตเพื่อส่งออก อายุ 46",
        objective="จับจังหวะวัฏจักรเทคโนโลยีจีนและเอเชีย",
        horizon_years=7,
        notes="70% ของพอร์ตอยู่บนปัจจัยเดียวคือจีนและเทคโนโลยีเอเชีย ระดับความเสี่ยง"
              "ของกองทุนอยู่ในเกณฑ์ แต่การกระจุกตัวสูงมาก",
        tags=["concentrated", "china-risk"],
    ),
    Client(
        id="C07",
        name="คุณศิริพร ล.",
        risk_profile=7,
        holdings=_h(kstar=0.40, kvalue=0.20, kfixed=0.30, kgold=0.10),
        aum_thb=52_000_000,
        persona="หุ้นส่วนกลุ่มโรงพยาบาล อายุ 50",
        objective="การเติบโตจากหุ้นไทย ถ่วงดุลด้วยตราสารหนี้และทองคำ",
        horizon_years=10,
        notes="โครงสร้างสมเหตุสมผล แต่ความเสี่ยงด้านหุ้น 60% เป็นการเดิมพันในประเทศไทย"
              "ประเทศเดียว โดยไม่มีการกระจายไปต่างประเทศ",
        tags=["home-bias"],
    ),

    # ---- Risk profile 8 ----------------------------------------------------
    Client(
        id="C08",
        name="คุณธนพร ม.",
        risk_profile=8,
        holdings=_h(kgold=1.00),
        aum_thb=120_000_000,
        persona="ผู้ค้าสินค้าโภคภัณฑ์ อายุ 44",
        objective="ป้องกันความเสี่ยงจากค่าเงินและภาวะเงินเฟ้อ",
        horizon_years=5,
        notes="ลงทุน 100% ในสินทรัพย์ทางเลือกเพียงประเภทเดียว ขณะที่ระดับความเสี่ยงนี้"
              "กำหนดให้ตราสารทุนต้องเป็นสัดส่วนหลัก และจำกัดสินทรัพย์ทางเลือกไม่เกิน 30%",
        tags=["single-fund", "alt-concentration"],
    ),
    Client(
        id="C09",
        name="คุณรัชานนท์ ด.",
        risk_profile=8,
        holdings=_h(kstar=1.00),
        aum_thb=45_000_000,
        persona="ทายาทธุรกิจรุ่นที่สอง อายุ 33",
        objective="การเติบโตระยะยาวสูงสุด รับความผันผวนระหว่างทางได้",
        horizon_years=20,
        notes="ทุ่มทั้งพอร์ตให้ผู้จัดการกองทุนหุ้นไทยรายเดียว เป็นการซ้อนความเสี่ยง"
              "จากผู้จัดการกองทุนกับความเสี่ยงประเทศเข้าด้วยกัน",
        tags=["single-fund", "home-bias"],
    ),
    Client(
        id="C10",
        name="คุณอุบล ช.",
        risk_profile=8,
        holdings=_h(kusa=0.70, kgold=0.30),
        aum_thb=200_000_000,
        persona="ประธานกรรมการบริษัทจดทะเบียน อายุ 58",
        objective="การเติบโตจากหุ้นสหรัฐฯ คู่กับทองคำเพื่อป้องกันความเสี่ยง",
        horizon_years=12,
        notes="โครงสร้าง barbell สองสินทรัพย์ที่ชัดเจน แต่ทั้งสองขาอ้างอิงสกุลดอลลาร์ "
              "ทำให้พอร์ตทั้งหมดเป็นการเดิมพันค่าเงินบาทไปด้วย",
        tags=["barbell", "usd-exposure"],
    ),
    Client(
        id="C11",
        name="คุณจิรวัฒน์ น.",
        risk_profile=8,
        holdings=_h(kstar=0.70, kviet=0.20, kfixedplus=0.10),
        aum_thb=70_000_000,
        persona="เจ้าของธุรกิจค้าปลีกภูมิภาค อายุ 39",
        objective="การเติบโตจากตลาดชายขอบและหุ้นไทย",
        horizon_years=15,
        notes="หุ้นตลาดเกิดใหม่และตลาดชายขอบรวม 90% ขณะที่ส่วนตราสารหนี้ 10% "
              "เล็กเกินกว่าจะรองรับช่วงตลาดขาลงได้",
        tags=["frontier", "thin-ballast"],
    ),
]

CLIENTS_BY_ID: Dict[str, Client] = {c.id: c for c in CLIENTS}


def get_client(client_id: str) -> Client:
    return CLIENTS_BY_ID[client_id]


def clients_by_profile() -> Dict[int, List[Client]]:
    out: Dict[int, List[Client]] = {}
    for c in CLIENTS:
        out.setdefault(c.risk_profile, []).append(c)
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# Suitability checks
# --------------------------------------------------------------------------- #
@dataclass
class Breach:
    kind: str        # "fund-level" | "allocation-band"
    severity: str    # "breach" | "watch"
    subject: str
    detail: str
    actual: float
    limit: float


def bucket_exposure(holdings: Mapping[str, float], universe: Mapping) -> Dict[str, float]:
    """Look-through exposure of a holdings dict to the four SEC buckets."""
    out = {b: 0.0 for b in SEC_BUCKETS}
    for code, w in holdings.items():
        fund = universe.get(code)
        if fund is None:
            continue
        for bucket, share in fund.lookthrough.items():
            out[bucket] += w * share
    return out


def check_suitability(
    holdings: Mapping[str, float],
    profile: RiskProfile,
    universe: Mapping,
    watch_margin: float = 0.02,
    acknowledged: Optional[set] = None,
) -> List[Breach]:
    """Fund-level and allocation-band suitability findings, worst first.

    ``acknowledged`` names funds the client has signed a risk acknowledgement
    for. Those are still reported — an RM needs to see them — but as a watch
    item rather than a breach, because a disclosed and consented holding above
    the client's level is a different thing from an undisclosed one.
    """
    findings: List[Breach] = []
    signed = set(acknowledged or ())

    for code, w in sorted(holdings.items(), key=lambda kv: -kv[1]):
        fund = universe.get(code)
        if fund is None or w <= 0:
            continue
        if fund.risk_level > profile.level:
            is_signed = code in signed
            findings.append(
                Breach(
                    kind="acknowledged exception" if is_signed else "fund-level",
                    severity="watch" if is_signed else "breach",
                    subject=code,
                    detail=(
                        f"{fund.name} เป็นผลิตภัณฑ์ความเสี่ยงระดับ {fund.risk_level} "
                        f"ถือไว้ {w:.0%} สูงกว่าเพดานระดับ {profile.level} ของลูกค้า"
                        + (" โดยอยู่ในขอบเขตที่ลูกค้าลงนามรับทราบความเสี่ยงแล้ว"
                           if is_signed else "")
                    ),
                    actual=float(fund.risk_level),
                    limit=float(profile.level),
                )
            )

    exposure = bucket_exposure(holdings, universe)
    for bucket in SEC_BUCKETS:
        lo, hi = profile.band(bucket)
        actual = exposure[bucket]
        if actual > hi + 1e-6:
            findings.append(
                Breach("allocation-band", "breach", bucket,
                       f"{bucket} อยู่ที่ {actual:.0%} สูงกว่าเพดาน {hi:.0%} "
                       f"สำหรับผู้ลงทุน{profile.name_th}", actual, hi))
        elif actual < lo - 1e-6:
            findings.append(
                Breach("allocation-band", "breach", bucket,
                       f"{bucket} อยู่ที่ {actual:.0%} ต่ำกว่าขั้นต่ำ {lo:.0%} "
                       f"สำหรับผู้ลงทุน{profile.name_th}", actual, lo))
        elif hi < 1.0 and watch_margin > 0 and actual > hi - watch_margin:
            findings.append(
                Breach("allocation-band", "watch", bucket,
                       f"{bucket} อยู่ที่ {actual:.0%} ห่างจากเพดาน {hi:.0%} "
                       f"ไม่ถึง {watch_margin:.0%}", actual, hi))

    order = {"breach": 0, "watch": 1}
    return sorted(findings, key=lambda f: (order[f.severity], f.kind))


def suitability_status(findings: List[Breach]) -> str:
    if any(f.severity == "breach" for f in findings):
        return "Breach"
    if findings:
        return "Watch"
    return "Compliant"


def concentration_findings(
    holdings: Mapping[str, float],
    universe: Mapping,
    single_fund_cap: float = 0.40,
    single_class_cap: float = 0.70,
) -> List[Breach]:
    """House-rule concentration checks, separate from the regulatory bands."""
    out: List[Breach] = []
    for code, w in sorted(holdings.items(), key=lambda kv: -kv[1]):
        if w > single_fund_cap + 1e-9:
            fund = universe.get(code)
            label = fund.name if fund else code
            out.append(Breach("concentration", "breach", code,
                              f"{label} คิดเป็น {w:.0%} ของพอร์ต สูงกว่าเพดาน "
                              f"{single_fund_cap:.0%} ต่อกองทุนตามเกณฑ์ภายใน",
                              w, single_fund_cap))
    by_class: Dict[str, float] = {}
    for code, w in holdings.items():
        fund = universe.get(code)
        if fund is not None:
            by_class[fund.asset_class] = by_class.get(fund.asset_class, 0.0) + w
    for cls, w in sorted(by_class.items(), key=lambda kv: -kv[1]):
        if w > single_class_cap + 1e-9:
            out.append(Breach("concentration", "watch", cls,
                              f"{cls} คิดเป็น {w:.0%} ของพอร์ต สูงกว่าแนวปฏิบัติ "
                              f"{single_class_cap:.0%} ต่อกลุ่มสินทรัพย์",
                              w, single_class_cap))
    return out
