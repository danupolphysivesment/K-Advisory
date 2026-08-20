"""Thai display strings for values that stay in English internally.

Some strings do double duty as dictionary keys — suitability status, severity
level, core/satellite role — and translating the constant itself would mean
chasing every lookup that depends on it. Those stay English in the code and are
translated here at the point of display.

Terminology follows the SEC/AIMC suitability form the app implements, so the
wording on screen matches the wording on the paperwork an RM already uses.
"""

from __future__ import annotations

from typing import Dict

# --------------------------------------------------------------------------- #
# Suitability
# --------------------------------------------------------------------------- #
STATUS: Dict[str, str] = {
    "Breach": "ไม่ผ่านเกณฑ์",
    "Watch": "ต้องเฝ้าระวัง",
    "Compliant": "ผ่านเกณฑ์",
}

SEVERITY: Dict[str, str] = {
    "Acute": "รุนแรง",
    "Elevated": "สูง",
    "Monitor": "เฝ้าติดตาม",
    "Background": "พื้นหลัง",
}

FINDING_KIND: Dict[str, str] = {
    "fund-level": "ระดับความเสี่ยงกองทุน",
    "acknowledged exception": "ข้อยกเว้นที่ลูกค้ารับทราบ",
    "allocation-band": "Asset Allocation",
    "concentration": "การกระจุกตัว",
}

OBSERVATION_SEVERITY: Dict[str, str] = {
    "high": "สูง",
    "medium": "ปานกลาง",
    "low": "ต่ำ",
}

# --------------------------------------------------------------------------- #
# Portfolio roles
# --------------------------------------------------------------------------- #
# Core / Satellite stay English: they are the industry's own words and every
# Thai rendering ("แกนหลัก / ส่วนเสริม") reads like a translation exercise.
ROLE: Dict[str, str] = {
    "Core": "Core",
    "Satellite": "Satellite",
}

# --------------------------------------------------------------------------- #
# Market environment
# --------------------------------------------------------------------------- #
TEMPERATURE = ["สงบ", "ปกติ", "ตึงตัว", "กดดัน"]
TEMPERATURE_TONE = ["mint", "sky", "amber", "coral"]

# --------------------------------------------------------------------------- #
# Rebalancing frequency
# --------------------------------------------------------------------------- #
REBALANCE: Dict[str, str] = {
    "Q": "ทุกไตรมาส",
    "M": "ทุกเดือน",
    "A": "ทุกปี",
    "none": "Buy & Hold (ไม่ปรับ)",
}

# --------------------------------------------------------------------------- #
# Month abbreviations, for tables and date captions
# --------------------------------------------------------------------------- #
MONTH_ABBR: Dict[str, str] = {
    "Jan": "ม.ค.", "Feb": "ก.พ.", "Mar": "มี.ค.", "Apr": "เม.ย.",
    "May": "พ.ค.", "Jun": "มิ.ย.", "Jul": "ก.ค.", "Aug": "ส.ค.",
    "Sep": "ก.ย.", "Oct": "ต.ค.", "Nov": "พ.ย.", "Dec": "ธ.ค.",
}


def month(stamp) -> str:
    """``เม.ย. 2026`` for a pandas Timestamp."""
    if stamp is None:
        return "—"
    return f"{MONTH_ABBR.get(stamp.strftime('%b'), stamp.strftime('%b'))} {stamp.year}"


def date(stamp) -> str:
    """``15 ก.ค. 2026`` for a pandas Timestamp."""
    if stamp is None:
        return "—"
    return (f"{stamp.day} {MONTH_ABBR.get(stamp.strftime('%b'), stamp.strftime('%b'))} "
            f"{stamp.year}")


def status(value: str) -> str:
    return STATUS.get(value, value)


def severity(value: str) -> str:
    return SEVERITY.get(value, value)


def role(value: str) -> str:
    return ROLE.get(value, value)


def kind(value: str) -> str:
    return FINDING_KIND.get(value, value)
