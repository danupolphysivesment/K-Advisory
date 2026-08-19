"""The bridge from Part 3 back into Parts 1 and 2.

Part 3 works out what is going on in the market. This module works out whether
*this client* should care, by intersecting each theme's exposure tags with the
tags carried by the funds the client actually holds.

The output is deliberately specific. "Watch US rates" is useless to an RM
sitting in front of a client. "62% of this book — K Fixed Income Pro and K
Global Bond — is long-duration, and the US 10-year has risen 40bp in three
months" is a conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .market import MarketView, Theme, ThemeView


@dataclass
class Caution:
    theme_key: str
    title: str
    category: str
    short: str                 # mid-sentence form of the theme name
    severity: int              # 0-3, from the theme
    exposure: float            # share of the portfolio carrying the tags
    holdings: List[Tuple[str, str, float]]   # (code, name, weight), largest first
    matched_tags: List[str]
    message: str
    why: str
    watch: Tuple[str, ...]
    score: float               # severity × exposure, used for ranking

    @property
    def severity_key(self) -> str:
        """English key, for colour and tone lookups."""
        return {0: "Background", 1: "Monitor", 2: "Elevated", 3: "Acute"}[
            int(np.clip(self.severity, 0, 3))]

    @property
    def severity_label(self) -> str:
        return {0: "พื้นหลัง", 1: "เฝ้าติดตาม", 2: "สูง", 3: "รุนแรง"}[
            int(np.clip(self.severity, 0, 3))]

    @property
    def exposure_label(self) -> str:
        if self.exposure >= 0.60:
            return "ครอบงำพอร์ต"
        if self.exposure >= 0.30:
            return "มีนัยสำคัญ"
        if self.exposure >= 0.10:
            return "ปานกลาง"
        return "เล็กน้อย"


def tag_exposure(
    weights: Mapping[str, float],
    universe: Mapping,
    tags: Sequence[str],
) -> Tuple[float, List[Tuple[str, str, float]], List[str]]:
    """Portfolio weight that carries any of ``tags``, plus the holdings that do.

    A holding is counted once no matter how many of the theme's tags it
    matches, so exposure can never exceed 100%.
    """
    wanted = set(tags)
    total = 0.0
    hits: List[Tuple[str, str, float]] = []
    matched: set = set()
    for code, w in weights.items():
        if w <= 0:
            continue
        fund = universe.get(code)
        if fund is None:
            continue
        overlap = wanted & set(fund.tags)
        if overlap:
            total += w
            hits.append((code, fund.name, w))
            matched |= overlap
    hits.sort(key=lambda h: -h[2])
    return total, hits, sorted(matched)


def build_cautions(
    weights: Mapping[str, float],
    universe: Mapping,
    view: MarketView,
    min_exposure: float = 0.05,
    min_severity: int = 1,
    limit: Optional[int] = None,
) -> List[Caution]:
    """Rank the market themes by how much this specific book is exposed to them."""
    out: List[Caution] = []
    for tv in view.themes:
        theme = tv.theme
        if tv.severity < min_severity:
            continue
        exposure, hits, matched = tag_exposure(weights, universe, theme.tags)
        if exposure < min_exposure or not hits:
            continue

        names = ", ".join(name for _, name, _ in hits[:3])
        if len(hits) > 3:
            names += f" และอีก {len(hits) - 3} กองทุน"

        driver = tv.signal.detail if tv.signal else theme.why.split(" ")[0:24]
        if isinstance(driver, list):
            driver = " ".join(driver)
        message = (f"{exposure:.0%} ของพอร์ตนี้ ({names}) มีความเสี่ยงต่อ"
                   f"{theme.short} — {driver}")

        out.append(Caution(
            theme_key=theme.key,
            title=theme.title,
            short=theme.short,
            category=theme.category,
            severity=tv.severity,
            exposure=exposure,
            holdings=hits,
            matched_tags=matched,
            message=message,
            why=theme.why,
            watch=theme.watch,
            score=tv.severity * exposure,
        ))

    out.sort(key=lambda c: -c.score)
    return out[:limit] if limit else out


def caution_frame(cautions: Sequence[Caution]) -> pd.DataFrame:
    if not cautions:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "ประเด็น": c.title,
            "หมวด": c.category,
            "ระดับ": c.severity_label,
            "สัดส่วนพอร์ตที่เกี่ยวข้อง": c.exposure,
            "จำนวนกองทุนที่กระทบ": len(c.holdings),
            "ลำดับความสำคัญ": c.score,
        }
        for c in cautions
    ]).set_index("ประเด็น")


def compare_cautions(
    current: Sequence[Caution],
    proposed: Sequence[Caution],
) -> pd.DataFrame:
    """Does the proposed portfolio reduce exposure to the live risks, or not?"""
    keys = {c.theme_key: c for c in current}
    props = {c.theme_key: c for c in proposed}
    rows = []
    for key in sorted(set(keys) | set(props)):
        a = keys.get(key)
        b = props.get(key)
        ref = a or b
        rows.append({
            "ประเด็น": ref.title,
            "ระดับ": ref.severity_label,
            "ปัจจุบัน": a.exposure if a else 0.0,
            "เสนอใหม่": b.exposure if b else 0.0,
            "เปลี่ยนแปลง": (b.exposure if b else 0.0) - (a.exposure if a else 0.0),
            "_sev": ref.severity,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["_sev", "ปัจจุบัน"], ascending=[False, False])
    return frame.drop(columns="_sev").set_index("ประเด็น")


def headline_risk_summary(cautions: Sequence[Caution]) -> str:
    """One sentence an RM can open the meeting with."""
    if not cautions:
        return ("ขณะนี้ไม่มีประเด็นตลาดใดที่กระทบสัดส่วนสำคัญของพอร์ตนี้")
    top = cautions[0]
    if len(cautions) == 1:
        return (f"ความเสี่ยงที่สำคัญที่สุดของพอร์ตนี้คือ{top.short} "
                f"ครอบคลุม {top.exposure:.0%} ของสินทรัพย์")
    second = cautions[1]
    return (f"สองประเด็นที่มีน้ำหนักที่สุดในพอร์ตนี้คือ {top.short} "
            f"({top.exposure:.0%} ของสินทรัพย์) และ {second.short} "
            f"({second.exposure:.0%})")


# --------------------------------------------------------------------------- #
# Structural (non-market) observations about a portfolio
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    kind: str          # "concentration" | "diversification" | "duration" | ...
    severity: str      # "high" | "medium" | "low"
    message: str


def structural_observations(
    weights: Mapping[str, float],
    universe: Mapping,
    effective_bets: float,
    diversification_ratio: float,
    top_risk_share: Optional[Tuple[str, float]] = None,
) -> List[Observation]:
    """Portfolio-shape findings that hold regardless of the market environment."""
    out: List[Observation] = []
    live = {c: w for c, w in weights.items() if w > 0}

    if len(live) == 1:
        code = next(iter(live))
        fund = universe.get(code)
        out.append(Observation(
            "concentration", "high",
            f"พอร์ตทั้งหมดอยู่ในกองทุนเดียว ({fund.name if fund else code}) "
            f"ลูกค้าแบกมุมมองของผู้จัดการกองทุนเพียงรายเดียว "
            f"โดยไม่มีอะไรมาถ่วงดุลหากมุมมองนั้นผิด"))
    elif effective_bets < 2.5:
        out.append(Observation(
            "concentration", "high",
            f"พอร์ตนี้ทำงานเสมือนมีสถานะอิสระเพียงประมาณ {effective_bets:.1f} สถานะ "
            f"แม้จะถือ {len(live)} กองทุน เพราะน้ำหนักเอียงมากเกินกว่าที่"
            f"จำนวนกองทุนจะมีความหมาย"))
    elif effective_bets < 4:
        out.append(Observation(
            "concentration", "medium",
            f"จำนวนสถานะที่มีผลจริงอยู่ที่ {effective_bets:.1f} "
            f"ยังมีช่องให้กระจายความเสี่ยงเพิ่มได้โดยไม่เปลี่ยนลักษณะของพอร์ต"))

    if diversification_ratio < 1.15:
        out.append(Observation(
            "diversification", "high",
            f"อัตราส่วนการกระจายความเสี่ยงอยู่ที่ {diversification_ratio:.2f} "
            f"กองทุนที่ถือเคลื่อนไหวไปด้วยกันมากพอที่การนำมารวมกัน"
            f"แทบไม่ช่วยลดความเสี่ยงเลย"))

    if top_risk_share is not None:
        code, share = top_risk_share
        fund = universe.get(code)
        if share > 0.60:
            out.append(Observation(
                "risk-concentration", "high",
                f"{fund.name if fund else code} เป็นแหล่งความเสี่ยง {share:.0%} "
                f"ของความเสี่ยงทั้งพอร์ต อะไรเกิดขึ้นกับกองทุนนี้ "
                f"ก็คือสิ่งที่จะเกิดขึ้นกับลูกค้า"))

    classes: Dict[str, float] = {}
    regions: Dict[str, float] = {}
    for code, w in live.items():
        fund = universe.get(code)
        if fund is None:
            continue
        classes[fund.asset_class] = classes.get(fund.asset_class, 0.0) + w
        regions[fund.region] = regions.get(fund.region, 0.0) + w

    for region, w in sorted(regions.items(), key=lambda kv: -kv[1])[:1]:
        if w > 0.70 and region != "ทั่วโลก":
            out.append(Observation(
                "geography", "medium",
                f"{w:.0%} ของพอร์ตอยู่ในตลาดเดียว ({region}) "
                f"เป็นการซ้อนความเสี่ยงประเทศกับความเสี่ยงค่าเงินเข้าด้วยกัน"))

    return out
