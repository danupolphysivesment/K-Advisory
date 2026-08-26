"""Client-facing PDF, assembled from whichever sections the RM ticks.

The point of this module is that an RM leaves the screen with something they can
put in front of a client. So it is deliberately a *numbers* document — tables,
findings and cautions — rather than a screenshot of the app: the charts on
screen are interactive Plotly figures and rasterising them would need a headless
browser this app does not carry.

Thai text is the reason ``reportlab`` gets a bundled font rather than a built-in
one. Every Type 1 base font ships Latin only, so an unregistered Thai string
renders as a row of black boxes rather than failing loudly — which is exactly
the kind of error that reaches a client before it reaches anyone else. Sarabun
is embedded from ``assets/fonts`` under the OFL, the same face the web UI uses.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

# Palette for print. The screen theme is a dark canvas, which is wrong on paper:
# it drinks toner and reads as a slide, not a document.
INK = "#12312F"
INK_SOFT = "#4A6B68"
RULE = "#C8D8D4"
ACCENT = "#0F7A57"
NEGATIVE = "#B3453C"
BAND = "#EEF5F2"

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "fonts")
FONT_REGULAR = "Sarabun"
FONT_BOLD = "Sarabun-Bold"

_FONTS_READY: Optional[bool] = None


class ReportUnavailable(RuntimeError):
    """Raised when the PDF cannot be built, with a reason fit to show an RM."""


def available() -> tuple:
    """``(ok, reason)`` — whether a PDF can be produced in this environment."""
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return False, ("ยังไม่ได้ติดตั้งไลบรารี reportlab บนเครื่องที่รันแอปนี้ "
                       "เพิ่ม reportlab ลงใน requirements.txt แล้ว deploy ใหม่")
    if not os.path.isdir(FONT_DIR):
        return False, (f"ไม่พบโฟลเดอร์ฟอนต์ {FONT_DIR} — ต้อง push โฟลเดอร์ "
                       f"assets/fonts ขึ้นไปด้วย ไม่อย่างนั้นข้อความภาษาไทย"
                       f"จะกลายเป็นกล่องสี่เหลี่ยม")
    missing = [n for n in ("Sarabun-Regular.ttf", "Sarabun-Bold.ttf")
               if not os.path.isfile(os.path.join(FONT_DIR, n))]
    if missing:
        return False, f"ไฟล์ฟอนต์หายไป: {', '.join(missing)}"
    return True, ""


def _register_fonts() -> None:
    """Embed Sarabun once per process."""
    global _FONTS_READY
    if _FONTS_READY:
        return
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    ok, reason = available()
    if not ok:
        raise ReportUnavailable(reason)
    pdfmetrics.registerFont(TTFont(FONT_REGULAR,
                                   os.path.join(FONT_DIR, "Sarabun-Regular.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD,
                                   os.path.join(FONT_DIR, "Sarabun-Bold.ttf")))
    pdfmetrics.registerFontFamily(FONT_REGULAR, normal=FONT_REGULAR,
                                  bold=FONT_BOLD, italic=FONT_REGULAR,
                                  boldItalic=FONT_BOLD)
    _FONTS_READY = True


# --------------------------------------------------------------------------- #
# Charts, restyled for paper
# --------------------------------------------------------------------------- #
# The app's figures are drawn for a dark canvas: mint on near-black. Printed on
# white they range from washed out to invisible. Rather than keep a second set
# of chart builders in step with the first, every figure is passed through this
# map on its way out — one source of truth for what a chart *is*, one for what
# it looks like on paper.
PRINT_MAP: Dict[str, str] = {
    "#40F8A9": ACCENT,      # mint accent  -> deep emerald
    "#57FFB8": "#0B6446",
    "#2BC486": "#157C58",
    "#00868D": "#00686E",   # teal
    "#EEC295": "#A8763C",   # sand
    "#95887D": "#6E6459",   # taupe
    "#F2938A": NEGATIVE,    # coral / losses
    "#E5736A": NEGATIVE,
    "#E8B44A": "#9A6D08",   # amber / warnings
    "#6FC8E8": "#2E7FA8",   # sky / informational
    "#B7E86F": "#5F8C1C",
    "#D89A5C": "#96602A",
    "#5FA8AE": "#37727A",
    "#8FE9C4": "#358F6B",
    "#EAF3F1": INK,         # body text
    "#8FB5B2": INK_SOFT,    # muted text
    "#78A8A5": INK_SOFT,    # dim labels
    "#155055": RULE,        # hairlines
    "#003336": "#FFFFFF",   # canvas
    "#04282B": BAND,
    "#0A3A3E": "#FFFFFF",
    "#0E464A": BAND,
}

# The screen's correlation ramp is capped dark so light text reads on it. On
# white the logic inverts: the ramp has to stay light enough for dark text.
PRINT_CORRELATION_SCALE = [
    [0.00, "#7FB4CE"],
    [0.22, "#A8CCDD"],
    [0.42, "#DCE9EE"],
    [0.50, "#FFFFFF"],
    [0.58, "#DCEDE4"],
    [0.78, "#A5D4BD"],
    [1.00, "#6FBE9B"],
]

# Colours reach a figure as "#RRGGBB", "rgb(r, g, b)" or "rgba(r, g, b, a)" —
# theme.rgba() emits the last of these without spaces — so match on the numbers
# rather than on any one spelling.
_BY_RGB = {tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)): out
           for h, out in PRINT_MAP.items()}
_RGB_CALL = re.compile(r"^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*"
                       r"(?:,\s*([\d.]+)\s*)?\)$")


def _print_colour(value: str) -> str:
    """One screen colour translated to paper, alpha preserved."""
    if value.startswith("#") and len(value) == 7:
        return PRINT_MAP.get(value.upper(), value)

    match = _RGB_CALL.match(value.strip())
    if not match:
        return value
    try:
        rgb = tuple(int(round(float(match.group(i)))) for i in (1, 2, 3))
    except ValueError:
        return value
    replacement = _BY_RGB.get(rgb)
    if replacement is None:
        return value
    r, g, b = (int(replacement[i:i + 2], 16) for i in (1, 3, 5))
    alpha = match.group(4)
    return f"rgba({r}, {g}, {b}, {alpha})" if alpha else f"rgb({r}, {g}, {b})"


def _remap(node):
    """Walk a figure dict, translating every colour string it holds."""
    if isinstance(node, dict):
        return {k: (PRINT_CORRELATION_SCALE
                    if k == "colorscale" and _is_screen_scale(v)
                    else _remap(v))
                for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_remap(v) for v in node]
    if isinstance(node, str):
        return _print_colour(node)
    return node


def _is_screen_scale(value) -> bool:
    try:
        return any(str(stop[1]).upper() in ("#276C90", "#0F7A57")
                   for stop in value)
    except (TypeError, IndexError):
        return False


# A figure rendered 1100px wide and placed 178mm (≈504pt) wide on the page is
# shrunk by roughly 2.2×, so anything sized for the screen arrives unreadable.
# Type is scaled up by that factor before rasterising, and the margins with it,
# or Plotly clips the tick labels it no longer has room for.
PRINT_FIGURE_WIDTH = 1100
PRINT_TYPE_SCALE = 1100 / 504.0


def to_print(fig, width: int = PRINT_FIGURE_WIDTH):
    """A copy of ``fig`` styled and sized for white paper."""
    import plotly.graph_objects as go

    printed = go.Figure(_remap(fig.to_dict()))
    k = width / 504.0
    printed.update_layout(
        template=None,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Sarabun, sans-serif", size=round(8.2 * k), color=INK),
        title=dict(font=dict(size=round(9.5 * k), color=INK)),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=round(7.6 * k), color=INK_SOFT)),
        margin=dict(l=round(34 * k), r=round(14 * k),
                    t=round(26 * k), b=round(26 * k)),
    )
    axis = dict(gridcolor=RULE, zerolinecolor=RULE, linecolor=RULE,
                tickfont=dict(size=round(7.4 * k), color=INK_SOFT),
                title=dict(font=dict(size=round(7.8 * k), color=INK_SOFT)),
                automargin=True)
    printed.update_xaxes(**axis)
    printed.update_yaxes(**axis)
    # Annotations (the correlation grid's numbers, chart callouts) carry their
    # own sizes from the screen and need the same treatment.
    for note in printed.layout.annotations or ():
        if note.font is not None and note.font.size:
            note.font.size = round(note.font.size * k)
    for trace in printed.data:
        tf = getattr(trace, "textfont", None)
        if tf is not None and getattr(tf, "size", None):
            tf.size = round(tf.size * k)
    return printed


def charts_available() -> tuple:
    """``(ok, reason)`` — whether figures can be rasterised for the PDF."""
    try:
        import kaleido  # noqa: F401
    except ImportError:
        return False, ("ยังไม่ได้ติดตั้ง kaleido บนเครื่องที่รันแอปนี้ "
                       "เพิ่ม kaleido ลงใน requirements.txt แล้ว deploy ใหม่ "
                       "จึงจะใส่กราฟลงใน PDF ได้")
    return True, ""


def figure_png(fig, width: int = PRINT_FIGURE_WIDTH, height: int = 460,
               scale: float = 2.0) -> Optional[bytes]:
    """Rasterise a figure for print, or return ``None`` if that is not possible.

    Returning ``None`` rather than raising is deliberate: one chart that will
    not render should cost the RM that chart, not the whole document.
    """
    ok, _ = charts_available()
    if not ok or fig is None:
        return None
    try:
        import plotly.io as pio
        return pio.to_image(to_print(fig, width), format="png", width=width,
                            height=height, scale=scale)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# The pieces an RM can tick
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    """One switchable piece of the document.

    ``is_chart`` matters because charts need kaleido, which a deployment may not
    have. Those blocks are then shown disabled with a reason rather than silently
    dropped from a document the RM believes is complete.
    """
    key: str
    title: str
    note: str = ""
    is_chart: bool = False


@dataclass
class Section:
    """A rendered block: a heading, optional prose, table and/or figure."""
    title: str
    lead: str = ""
    columns: Sequence[str] = ()
    rows: Sequence[Sequence[str]] = ()
    bullets: Sequence[str] = ()
    align_right: Sequence[int] = field(default_factory=list)
    image: Optional[bytes] = None
    image_width_mm: float = 178.0
    image_height_mm: float = 0.0        # 0 keeps the PNG's own aspect ratio


CURRENT_BLOCKS: List[Block] = [
    Block("profile", "ข้อมูลลูกค้าและระดับความเสี่ยง",
          "ชื่อ ระดับความเสี่ยง วัตถุประสงค์ และมูลค่าพอร์ต"),
    Block("summary", "สรุปสถิติพอร์ต",
          "ผลตอบแทนต่อปี ความผันผวน Max Drawdown และ VaR"),
    Block("holdings", "รายการกองทุนที่ถืออยู่",
          "น้ำหนัก มูลค่า และ Risk Contribution รายกองทุน"),
    Block("allocation", "สัดส่วนตาม Asset Class",
          "และสัดส่วนตามกรอบความเหมาะสมของ ก.ล.ต."),
    Block("annual", "ผลตอบแทนรายปี",
          "ผลตอบแทน ความผันผวน และ Max Drawdown แต่ละปี"),
    Block("attribution", "Return Attribution",
          "กองทุนไหนสร้างผลตอบแทนให้พอร์ตเท่าไร"),
    Block("risk", "Risk Contribution",
          "น้ำหนักเงินลงทุนเทียบกับน้ำหนักความเสี่ยง"),
    Block("suitability", "ผลตรวจความเหมาะสม",
          "ข้อที่ผ่าน ข้อที่ต้องเฝ้าระวัง และข้อที่ไม่ผ่านเกณฑ์"),
    Block("stress", "Stress Test",
          "ผลกระทบถ้าเหตุการณ์ในอดีตเกิดขึ้นอีก"),
    Block("montecarlo", "Monte Carlo Simulation",
          "ช่วงมูลค่าพอร์ตที่เป็นไปได้ในอนาคต"),
    Block("cautions", "ข้อควรระวังจากภาวะตลาด",
          "ประเด็นตลาดที่กระทบพอร์ตนี้โดยตรง"),
    Block("notes", "บันทึกของผู้แนะนำการลงทุน",
          "ข้อความที่บันทึกไว้ในหน้าข้อมูลลูกค้า"),

    Block("chart_growth", "กราฟ · ผลการดำเนินงานย้อนหลัง",
          "มูลค่าเงินลงทุนเทียบกับตัวเปรียบเทียบ", is_chart=True),
    Block("chart_drawdown", "กราฟ · Drawdown",
          "ระยะที่พอร์ตอยู่ต่ำกว่าจุดสูงสุดเดิม", is_chart=True),
    Block("chart_calendar", "กราฟ · ผลตอบแทนรายปี", is_chart=True),
    Block("chart_allocation", "กราฟ · สัดส่วนการลงทุน",
          "แยกตามกองทุนและตาม Asset Class", is_chart=True),
    Block("chart_bands", "กราฟ · สัดส่วนเทียบกรอบความเหมาะสม", is_chart=True),
    Block("chart_attribution", "กราฟ · Return Attribution", is_chart=True),
    Block("chart_risk", "กราฟ · Risk Contribution",
          "น้ำหนักเงินลงทุนเทียบน้ำหนักความเสี่ยง", is_chart=True),
    Block("chart_correlation", "กราฟ · สหสัมพันธ์ระหว่างกองทุน", is_chart=True),
    Block("chart_stress", "กราฟ · Stress Test", is_chart=True),
    Block("chart_montecarlo", "กราฟ · Monte Carlo Fan Chart", is_chart=True),
]

PROPOSED_BLOCKS: List[Block] = [
    Block("mandate", "เงื่อนไขการจัดพอร์ต",
          "Objective เพดานน้ำหนัก และ Fund Universe ที่ใช้"),
    Block("summary", "สรุปสถิติพอร์ตที่แนะนำ",
          "Expected Return ความผันผวน และ Max Drawdown"),
    Block("holdings", "รายการกองทุนที่แนะนำ",
          "น้ำหนักและมูลค่าที่แนะนำรายกองทุน"),
    Block("allocation", "สัดส่วนตาม Asset Class และ Core / Satellite"),
    Block("comparison", "เปรียบเทียบกับพอร์ตปัจจุบัน",
          "สถิติของทั้งสองพอร์ตบนช่วงข้อมูลเดียวกัน"),
    Block("trades", "รายการซื้อขายที่ต้องทำ",
          "ซื้อ ขาย และมูลค่าต่อรายการ"),
    Block("suitability", "ผลตรวจความเหมาะสมของพอร์ตที่แนะนำ"),
    Block("stress", "Stress Test เทียบสองพอร์ต"),
    Block("montecarlo", "Monte Carlo Simulation ของพอร์ตที่แนะนำ"),
    Block("cautions", "ข้อควรระวังจากภาวะตลาด"),

    Block("chart_allocation", "กราฟ · สัดส่วนของพอร์ตที่แนะนำ", is_chart=True),
    Block("chart_weights", "กราฟ · เทียบน้ำหนักรายกองทุนสองพอร์ต",
          is_chart=True),
    Block("chart_frontier", "กราฟ · Efficient Frontier",
          "ตำแหน่งของพอร์ตที่แนะนำบนเส้นขอบประสิทธิภาพ", is_chart=True),
    Block("chart_stress", "กราฟ · Stress Test เทียบสองพอร์ต", is_chart=True),
    Block("chart_montecarlo", "กราฟ · Monte Carlo เทียบสองพอร์ต",
          is_chart=True),
    Block("chart_exposure", "กราฟ · Exposure ต่อประเด็นตลาด", is_chart=True),
]

# Ticked when the panel first opens: enough to be a usable handout, short
# enough to read in a meeting. Charts are in the default set because a client
# reads a picture faster than a table.
DEFAULT_CURRENT = ["profile", "summary", "holdings", "allocation",
                   "suitability", "cautions",
                   "chart_growth", "chart_allocation", "chart_risk"]
DEFAULT_PROPOSED = ["mandate", "summary", "holdings", "allocation",
                    "comparison", "trades", "suitability",
                    "chart_allocation", "chart_stress"]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def build_pdf(title: str, subtitle: str, sections: Sequence[Section],
              footer: str = "", disclaimer: str = "") -> bytes:
    """Lay the chosen sections out as A4 and return the file's bytes."""
    _register_fonts()

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                    PageTemplate, Paragraph, Spacer, Table,
                                    TableStyle)

    buffer = io.BytesIO()
    page_w, page_h = A4
    margin = 16 * mm

    body = ParagraphStyle("body", fontName=FONT_REGULAR, fontSize=8.4,
                          leading=13.5, textColor=colors.HexColor(INK),
                          alignment=TA_LEFT)
    lead = ParagraphStyle("lead", parent=body, fontSize=8.0, leading=13.0,
                          textColor=colors.HexColor(INK_SOFT),
                          spaceAfter=4)
    heading = ParagraphStyle("heading", fontName=FONT_BOLD, fontSize=10.5,
                             leading=15, textColor=colors.HexColor(ACCENT),
                             spaceBefore=8, spaceAfter=3)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=9,
                            bulletIndent=1, spaceAfter=2)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=11.4)
    cell_head = ParagraphStyle("cellhead", parent=cell, fontName=FONT_BOLD,
                               textColor=colors.HexColor(INK))

    def _decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT_BOLD, 13)
        canvas.setFillColor(colors.HexColor(ACCENT))
        canvas.drawString(margin, page_h - margin + 5 * mm, title)
        canvas.setFont(FONT_REGULAR, 7.6)
        canvas.setFillColor(colors.HexColor(INK_SOFT))
        canvas.drawString(margin, page_h - margin + 1.2 * mm, subtitle)
        canvas.setStrokeColor(colors.HexColor(RULE))
        canvas.setLineWidth(0.6)
        canvas.line(margin, page_h - margin - 1 * mm,
                    page_w - margin, page_h - margin - 1 * mm)
        canvas.line(margin, margin + 8 * mm, page_w - margin, margin + 8 * mm)
        canvas.setFont(FONT_REGULAR, 6.8)
        canvas.drawString(margin, margin + 4 * mm, footer)
        canvas.drawRightString(page_w - margin, margin + 4 * mm,
                               f"หน้า {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(buffer, pagesize=A4,
                          leftMargin=margin, rightMargin=margin,
                          topMargin=margin + 8 * mm, bottomMargin=margin + 12 * mm,
                          title=title, author="K-ADVISOR")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="body", showBoundary=0)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=_decorate)])

    story: List = []
    for section in sections:
        head = [Paragraph(section.title, heading)]
        if section.lead:
            head.append(Paragraph(section.lead, lead))

        if section.image:
            width = min(section.image_width_mm * mm, doc.width)
            if section.image_height_mm:
                height = section.image_height_mm * mm
            else:
                from reportlab.lib.utils import ImageReader
                px_w, px_h = ImageReader(io.BytesIO(section.image)).getSize()
                height = width * px_h / px_w
            # A chart and its heading travel together. Splitting them leaves a
            # title stranded at the foot of one page and an unlabelled picture
            # at the top of the next, which is worse than a short page.
            head.append(Image(io.BytesIO(section.image), width=width,
                              height=height))
            head.append(Spacer(1, 4))

        story.append(KeepTogether(head))

        for line in section.bullets:
            story.append(Paragraph(line, bullet, bulletText="•"))

        if section.columns and section.rows:
            data = [[Paragraph(str(c), cell_head) for c in section.columns]]
            for row in section.rows:
                data.append([Paragraph("" if v is None else str(v), cell)
                             for v in row])
            table = Table(data, repeatRows=1, hAlign="LEFT")
            style = [
                ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
                ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BAND)),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor(ACCENT)),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor(RULE)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
            for col in section.align_right:
                style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
            table.setStyle(TableStyle(style))
            story.append(table)

        story.append(Spacer(1, 5))

    if disclaimer:
        story.append(Spacer(1, 8))
        story.append(Paragraph(disclaimer, ParagraphStyle(
            "disc", parent=body, fontSize=6.9, leading=10.2,
            textColor=colors.HexColor(INK_SOFT))))

    if not story:
        story = [Paragraph("ไม่ได้เลือกหัวข้อใดไว้", body)]

    doc.build(story)
    return buffer.getvalue()
