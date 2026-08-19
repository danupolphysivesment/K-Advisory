"""Plotly figure builders, all sharing the ``kasset`` template from theme.py."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme as T


def _empty(message: str = "ไม่มีข้อมูลสำหรับตัวเลือกนี้") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(color=T.DIM, size=13, family=T.SANS))
    fig.update_layout(height=220, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# --------------------------------------------------------------------------- #
# Allocation
# --------------------------------------------------------------------------- #
def allocation_donut(weights: Mapping[str, float], labels: Mapping[str, str],
                     colors: Optional[Mapping[str, str]] = None,
                     centre: str = "", height: int = 300) -> go.Figure:
    items = [(labels.get(k, k), v) for k, v in weights.items() if v > 1e-6]
    if not items:
        return _empty()
    items.sort(key=lambda kv: -kv[1])
    names = [i[0] for i in items]
    values = [i[1] for i in items]
    if colors:
        marker = [colors.get(n, T.series_color(i)) for i, n in enumerate(names)]
    else:
        marker = [T.series_color(i) for i in range(len(names))]

    fig = go.Figure(go.Pie(
        labels=names, values=values, hole=0.62, sort=False,
        marker=dict(colors=marker, line=dict(color=T.BG, width=2)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    ))
    if centre:
        fig.add_annotation(text=centre, showarrow=False,
                           font=dict(family=T.SERIF, size=15, color=T.TEXT))
    fig.update_layout(height=height, showlegend=True,
                      legend=dict(orientation="v", x=1.0, y=0.5, yanchor="middle"),
                      margin=dict(l=0, r=0, t=10, b=0))
    return fig


def allocation_bars(weights: Mapping[str, float], labels: Mapping[str, str],
                    colors: Optional[Mapping[str, str]] = None,
                    height: int = 300, suffix: str = "") -> go.Figure:
    items = [(labels.get(k, k), v) for k, v in weights.items() if abs(v) > 1e-6]
    if not items:
        return _empty()
    items.sort(key=lambda kv: kv[1])
    names = [i[0] for i in items]
    values = [i[1] for i in items]
    marker = ([colors.get(n, T.MINT) for n in names] if colors
              else [T.series_color(i) for i in range(len(names))])

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=marker),
        text=[f"{v:.1%}{suffix}" for v in values],
        textposition="outside",
        textfont=dict(family=T.SANS, size=11, color=T.MUTED),
        hovertemplate="<b>%{y}</b><br>%{x:.2%}<extra></extra>",
    ))
    fig.update_layout(height=height, showlegend=False,
                      xaxis=dict(tickformat=".0%", range=[0, max(values) * 1.22]),
                      margin=dict(l=0, r=30, t=10, b=10))
    return fig


def band_compliance(exposure: Mapping[str, float],
                    bands: Mapping[str, tuple],
                    height: int = 250) -> go.Figure:
    """Actual bucket exposure against the suitability band for each bucket."""
    buckets = [b for b in exposure]
    if not buckets:
        return _empty()

    fig = go.Figure()
    for i, bucket in enumerate(buckets):
        lo, hi = bands.get(bucket, (0.0, 1.0))
        actual = exposure[bucket]
        # Permitted band as a light bar
        fig.add_trace(go.Bar(
            x=[hi - lo], y=[bucket], base=[lo], orientation="h",
            marker=dict(color=T.LINE), width=0.55,
            hovertemplate=f"อนุญาต {lo:.0%} – {hi:.0%}<extra></extra>",
            showlegend=i == 0, name="กรอบที่อนุญาต",
        ))
        breach = actual > hi + 1e-6 or actual < lo - 1e-6
        fig.add_trace(go.Scatter(
            x=[actual], y=[bucket], mode="markers",
            marker=dict(symbol="line-ns", size=26, line=dict(
                color=T.CORAL if breach else T.MINT, width=3)),
            hovertemplate=f"สัดส่วนจริง {actual:.1%}<extra></extra>",
            showlegend=i == 0, name="สัดส่วนจริง",
        ))
    fig.update_layout(
        height=height, barmode="overlay",
        xaxis=dict(tickformat=".0%", range=[0, 1.02], title=""),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=10, t=30, b=10),
    )
    return fig


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def growth_chart(series_map: Mapping[str, pd.Series], height: int = 340,
                 base: float = 100.0, log: bool = False) -> go.Figure:
    if not series_map:
        return _empty()
    fig = go.Figure()
    for i, (label, series) in enumerate(series_map.items()):
        s = series.dropna()
        if s.empty:
            continue
        curve = base * (1.0 + s).cumprod()
        colour = T.MINT if i == 0 else T.series_color(i + 1)
        fig.add_trace(go.Scatter(
            x=curve.index, y=curve.values, name=label, mode="lines",
            line=dict(color=colour, width=2.0 if i == 0 else 1.5),
            hovertemplate="<b>%{fullData.name}</b><br>%{x|%d %b %Y}<br>"
                          "%{y:,.1f}<extra></extra>",
        ))
    fig.update_layout(height=height,
                      yaxis=dict(title=f"มูลค่าจาก {base:,.0f}",
                                 type="log" if log else "linear"),
                      margin=dict(l=0, r=8, t=34, b=8))
    return fig


def drawdown_chart(series_map: Mapping[str, pd.Series], height: int = 250) -> go.Figure:
    if not series_map:
        return _empty()
    fig = go.Figure()
    for i, (label, series) in enumerate(series_map.items()):
        s = series.dropna()
        if s.empty:
            continue
        curve = (1.0 + s).cumprod()
        dd = curve / curve.cummax() - 1.0
        colour = T.CORAL if i == 0 else T.series_color(i + 2)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=label, mode="lines",
            line=dict(color=colour, width=1.4),
            fill="tozeroy" if i == 0 else None,
            fillcolor=T.rgba(T.CORAL, 0.16) if i == 0 else None,
            hovertemplate="<b>%{fullData.name}</b><br>%{x|%d %b %Y}<br>"
                          "%{y:.2%}<extra></extra>",
        ))
    fig.update_layout(height=height, yaxis=dict(tickformat=".0%", title="การขาดทุนจากจุดสูงสุด"),
                      margin=dict(l=0, r=8, t=30, b=8))
    return fig


def rolling_chart(series_map: Mapping[str, pd.Series], title: str,
                  percent: bool = True, height: int = 240) -> go.Figure:
    if not series_map:
        return _empty()
    fig = go.Figure()
    for i, (label, series) in enumerate(series_map.items()):
        s = series.dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=label, mode="lines",
            line=dict(color=T.MINT if i == 0 else T.series_color(i + 1), width=1.6),
            hovertemplate="%{x|%b %Y}<br>%{y:.2%}<extra></extra>"
            if percent else "%{x|%b %Y}<br>%{y:.2f}<extra></extra>",
        ))
    fig.update_layout(height=height, title=title,
                      yaxis=dict(tickformat=".0%" if percent else None),
                      margin=dict(l=0, r=8, t=36, b=8))
    return fig


def calendar_bars(returns_by_year: pd.Series, height: int = 230,
                  comparison: Optional[pd.Series] = None) -> go.Figure:
    if returns_by_year.empty:
        return _empty()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=returns_by_year.index.astype(str), y=returns_by_year.values,
        name="ปัจจุบัน",
        marker=dict(color=[T.MINT if v >= 0 else T.CORAL for v in returns_by_year]),
        hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
    ))
    if comparison is not None and not comparison.empty:
        aligned = comparison.reindex(returns_by_year.index)
        fig.add_trace(go.Bar(
            x=aligned.index.astype(str), y=aligned.values, name="เสนอใหม่",
            marker=dict(color=T.SAND, opacity=0.85),
            hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
        ))
    fig.update_layout(height=height, barmode="group",
                      yaxis=dict(tickformat=".0%"),
                      margin=dict(l=0, r=8, t=30, b=8))
    return fig


# --------------------------------------------------------------------------- #
# Attribution & risk
# --------------------------------------------------------------------------- #
def contribution_waterfall(contrib: pd.Series, total: float,
                           labels: Optional[Mapping[str, str]] = None,
                           height: int = 300) -> go.Figure:
    if contrib.empty:
        return _empty()
    ordered = contrib.sort_values(ascending=False)
    names = [(labels or {}).get(k, k) for k in ordered.index]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(ordered) + ["total"],
        x=names + ["รวมพอร์ต"],
        y=list(ordered.values) + [total],
        connector=dict(line=dict(color=T.LINE, width=1)),
        increasing=dict(marker=dict(color=T.MINT)),
        decreasing=dict(marker=dict(color=T.CORAL)),
        totals=dict(marker=dict(color=T.SAND)),
        text=[f"{v:+.1%}" for v in ordered.values] + [f"{total:+.1%}"],
        textposition="outside",
        textfont=dict(family=T.SANS, size=10, color=T.MUTED),
        hovertemplate="<b>%{x}</b><br>%{y:+.2%}<extra></extra>",
    ))
    fig.update_layout(height=height, showlegend=False,
                      yaxis=dict(tickformat=".0%"),
                      margin=dict(l=0, r=8, t=30, b=8))
    return fig


def risk_vs_weight(frame: pd.DataFrame, labels: Optional[Mapping[str, str]] = None,
                   height: int = 300) -> go.Figure:
    """Weight against share of risk — the gap is the story."""
    if frame.empty:
        return _empty()
    ordered = frame.sort_values("risk_share", ascending=True)
    names = [(labels or {}).get(k, k) for k in ordered.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ordered["weight"], y=names, orientation="h", name="น้ำหนักเงินลงทุน",
        marker=dict(color=T.TEAL), width=0.38, offset=-0.40,
        hovertemplate="<b>%{y}</b><br>น้ำหนัก %{x:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=ordered["risk_share"], y=names, orientation="h", name="สัดส่วนความเสี่ยง",
        marker=dict(color=T.MINT), width=0.38, offset=0.02,
        hovertemplate="<b>%{y}</b><br>ความเสี่ยง %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(height=height, barmode="overlay",
                      xaxis=dict(tickformat=".0%"),
                      margin=dict(l=0, r=8, t=34, b=8))
    return fig


def correlation_heatmap(corr: pd.DataFrame, labels: Optional[Mapping[str, str]] = None,
                        height: int = 340) -> go.Figure:
    if corr.empty:
        return _empty()
    names = [(labels or {}).get(c, c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=names, y=names,
        colorscale=[[0.0, T.SKY], [0.5, T.BG], [1.0, T.MINT]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(family=T.SANS, size=10, color=T.TEXT),
        hovertemplate="%{y} กับ %{x}<br>ρ = %{z:.2f}<extra></extra>",
        colorbar=dict(outlinewidth=0, tickfont=dict(color=T.MUTED, size=10),
                      thickness=10, len=0.8),
    ))
    fig.update_layout(height=height, margin=dict(l=0, r=8, t=30, b=8),
                      yaxis=dict(autorange="reversed"))
    return fig


def stress_bars(frame: pd.DataFrame, height: int = 340,
                comparison: Optional[pd.DataFrame] = None) -> go.Figure:
    if frame.empty:
        return _empty()
    ordered = frame.sort_values("ผลตอบแทน")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ordered["ผลตอบแทน"], y=ordered.index, orientation="h", name="ปัจจุบัน",
        marker=dict(color=[T.MINT if v >= 0 else T.CORAL for v in ordered["ผลตอบแทน"]]),
        text=[f"{v:.1%}" for v in ordered["ผลตอบแทน"]], textposition="outside",
        textfont=dict(family=T.SANS, size=10, color=T.MUTED),
        hovertemplate="<b>%{y}</b><br>ผลตอบแทน %{x:.1%}<extra></extra>",
    ))
    if comparison is not None and not comparison.empty:
        aligned = comparison.reindex(ordered.index)
        fig.add_trace(go.Scatter(
            x=aligned["ผลตอบแทน"], y=aligned.index, mode="markers", name="เสนอใหม่",
            marker=dict(symbol="diamond", size=9, color=T.SAND,
                        line=dict(color=T.BG, width=1)),
            hovertemplate="<b>%{y}</b><br>เสนอใหม่ %{x:.1%}<extra></extra>",
        ))
    span = float(ordered["ผลตอบแทน"].abs().max() or 0.1)
    fig.update_layout(height=height, xaxis=dict(tickformat=".0%",
                                                range=[-span * 1.35, span * 0.9]),
                      margin=dict(l=0, r=8, t=34, b=8))
    return fig


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
def fan_chart(percentiles: pd.DataFrame, period_label: str,
              initial_value: float = 1.0, height: int = 400,
              overlay: Optional[pd.DataFrame] = None,
              overlay_name: str = "เสนอใหม่") -> go.Figure:
    if percentiles.empty:
        return _empty()
    x = percentiles.index.to_numpy()
    scaled = percentiles * initial_value

    bands = [("p1", "p99", 0.06), ("p5", "p95", 0.10),
             ("p10", "p90", 0.14), ("p25", "p75", 0.22)]
    fig = go.Figure()
    for lo, hi, alpha in bands:
        if lo not in scaled or hi not in scaled:
            continue
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([scaled[hi].to_numpy(), scaled[lo].to_numpy()[::-1]]),
            fill="toself", fillcolor=T.rgba(T.MINT, alpha),
            line=dict(width=0), hoverinfo="skip",
            name=f"เปอร์เซ็นไทล์ {lo[1:]}–{hi[1:]}", showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=x, y=scaled["p50"], mode="lines", name="ค่ากลาง",
        line=dict(color=T.MINT, width=2.4),
        hovertemplate=f"{period_label} %{{x}}<br>%{{y:,.2f}}<extra></extra>",
    ))
    fig.add_hline(y=initial_value, line=dict(color=T.DIM, width=1, dash="dot"))

    if overlay is not None and not overlay.empty:
        ov = overlay * initial_value
        fig.add_trace(go.Scatter(
            x=ov.index.to_numpy(), y=ov["p50"], mode="lines",
            name=f"ค่ากลาง — {overlay_name}",
            line=dict(color=T.SAND, width=2, dash="dash"),
            hovertemplate=f"{overlay_name}<br>%{{y:,.2f}}<extra></extra>",
        ))
        for col, dash in (("p5", "dot"), ("p95", "dot")):
            if col in ov:
                fig.add_trace(go.Scatter(
                    x=ov.index.to_numpy(), y=ov[col], mode="lines",
                    name=f"{overlay_name} {col}", line=dict(color=T.SAND, width=1,
                                                            dash=dash),
                    opacity=0.6, showlegend=False, hoverinfo="skip"))

    fig.update_layout(height=height,
                      xaxis=dict(title=f"จำนวน{period_label}ข้างหน้า"),
                      yaxis=dict(title="มูลค่าพอร์ต"),
                      margin=dict(l=0, r=8, t=36, b=8))
    return fig


def terminal_distribution(terminal: np.ndarray, initial_value: float = 1.0,
                          height: int = 260, target: Optional[float] = None
                          ) -> go.Figure:
    if terminal is None or len(terminal) == 0:
        return _empty()
    values = terminal * initial_value
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=70, marker=dict(color=T.TEAL, line=dict(width=0)),
        hovertemplate="%{x:,.2f}<br>%{y} paths<extra></extra>",
    ))
    median = float(np.median(values))
    fig.add_vline(x=median, line=dict(color=T.MINT, width=2),
                  annotation_text=f"ค่ากลาง {median:,.2f}",
                  annotation_font=dict(color=T.MINT, size=11, family=T.SANS))
    fig.add_vline(x=initial_value, line=dict(color=T.DIM, width=1, dash="dot"),
                  annotation_text="เริ่มต้น",
                  annotation_font=dict(color=T.DIM, size=10, family=T.SANS))
    if target is not None:
        fig.add_vline(x=target, line=dict(color=T.SAND, width=1.5, dash="dash"),
                      annotation_text="เป้าหมาย",
                      annotation_font=dict(color=T.SAND, size=10, family=T.SANS))
    fig.update_layout(height=height, showlegend=False,
                      xaxis=dict(title="มูลค่าปลายทาง"),
                      yaxis=dict(title="จำนวนเส้นทาง"),
                      margin=dict(l=0, r=8, t=34, b=8))
    return fig


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #
def frontier_chart(frontier: pd.DataFrame,
                   points: Sequence[tuple] = (),
                   height: int = 360) -> go.Figure:
    """``points`` is a sequence of (label, vol, ret, colour, symbol)."""
    fig = go.Figure()
    if not frontier.empty:
        fig.add_trace(go.Scatter(
            x=frontier["volatility"], y=frontier["expected_return"],
            mode="lines", name="เส้นขอบประสิทธิภาพ",
            line=dict(color=T.TEAL, width=2),
            hovertemplate="ความผันผวน %{x:.2%}<br>ผลตอบแทน %{y:.2%}<extra></extra>",
        ))
    for label, vol, ret, colour, symbol in points:
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers+text", name=label,
            marker=dict(size=13, color=colour, symbol=symbol,
                        line=dict(color=T.BG, width=1.5)),
            text=[label], textposition="top center",
            textfont=dict(family=T.SANS, size=10, color=colour),
            hovertemplate=f"<b>{label}</b><br>ความผันผวน %{{x:.2%}}<br>"
                          f"ผลตอบแทน %{{y:.2%}}<extra></extra>",
        ))
    fig.update_layout(height=height, showlegend=False,
                      xaxis=dict(title="ความผันผวนคาดหวัง", tickformat=".0%"),
                      yaxis=dict(title="ผลตอบแทนคาดหวัง", tickformat=".0%"),
                      margin=dict(l=0, r=8, t=34, b=8))
    return fig


def weight_comparison(current: Mapping[str, float], proposed: Mapping[str, float],
                      labels: Optional[Mapping[str, str]] = None,
                      height: int = 340) -> go.Figure:
    codes = sorted(set(current) | set(proposed),
                   key=lambda c: -(max(current.get(c, 0), proposed.get(c, 0))))
    if not codes:
        return _empty()
    names = [(labels or {}).get(c, c) for c in codes]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=[current.get(c, 0.0) for c in codes], orientation="h",
        name="ปัจจุบัน", marker=dict(color=T.TEAL),
        hovertemplate="<b>%{y}</b><br>ปัจจุบัน %{x:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=names, x=[proposed.get(c, 0.0) for c in codes], orientation="h",
        name="เสนอใหม่", marker=dict(color=T.MINT),
        hovertemplate="<b>%{y}</b><br>เสนอใหม่ %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(height=height, barmode="group",
                      xaxis=dict(tickformat=".0%"),
                      yaxis=dict(autorange="reversed"),
                      margin=dict(l=0, r=8, t=34, b=8))
    return fig


def core_satellite_bar(split: Mapping[str, float], height: int = 120) -> go.Figure:
    core = split.get("Core", 0.0)
    sat = split.get("Satellite", 0.0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[core], y=[""], orientation="h", name="แกนหลัก",
                         marker=dict(color=T.TEAL),
                         text=[f"แกนหลัก {core:.0%}"], textposition="inside",
                         textfont=dict(family=T.SANS, size=12, color=T.TEXT),
                         hovertemplate="แกนหลัก %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(x=[sat], y=[""], orientation="h", name="ส่วนเสริม",
                         marker=dict(color=T.SAND),
                         text=[f"ส่วนเสริม {sat:.0%}"], textposition="inside",
                         textfont=dict(family=T.SANS, size=12, color=T.BG),
                         hovertemplate="ส่วนเสริม %{x:.1%}<extra></extra>"))
    fig.update_layout(height=height, barmode="stack", showlegend=False,
                      xaxis=dict(visible=False, range=[0, 1]),
                      yaxis=dict(visible=False),
                      margin=dict(l=0, r=0, t=8, b=8))
    return fig


# --------------------------------------------------------------------------- #
# Market
# --------------------------------------------------------------------------- #
def macro_sparkline(series: pd.Series, positive: bool = True,
                    height: int = 54) -> go.Figure:
    s = series.dropna().tail(180)
    if s.empty:
        return _empty("")
    colour = T.MINT if positive else T.CORAL
    fig = go.Figure(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(color=colour, width=1.4),
        fill="tozeroy", fillcolor=T.rgba(colour, 0.10),
        hoverinfo="skip",
    ))
    fig.update_layout(height=height, showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig


def theme_exposure_chart(frame: pd.DataFrame, height: int = 320) -> go.Figure:
    """Current vs proposed exposure to each live market theme."""
    if frame.empty:
        return _empty("ไม่มีประเด็นตลาดที่เกี่ยวข้องกับพอร์ตนี้")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=frame.index, x=frame["ปัจจุบัน"], orientation="h", name="ปัจจุบัน",
        marker=dict(color=T.CORAL),
        hovertemplate="<b>%{y}</b><br>ปัจจุบัน %{x:.0%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=frame.index, x=frame["เสนอใหม่"], orientation="h", name="เสนอใหม่",
        marker=dict(color=T.MINT),
        hovertemplate="<b>%{y}</b><br>เสนอใหม่ %{x:.0%}<extra></extra>",
    ))
    fig.update_layout(height=height, barmode="group",
                      xaxis=dict(tickformat=".0%"),
                      yaxis=dict(autorange="reversed"),
                      margin=dict(l=0, r=8, t=34, b=8))
    return fig
