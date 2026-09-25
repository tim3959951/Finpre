"""Plotly figures for the Streamlit app.

Colours: categorical slots from the validated reference palette (fixed order, never cycled).
Candles follow the market's convention: Taiwan 紅漲綠跌, US green-up/red-down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..features.indicators import add_indicators
from ..forecasting.base import ForecastResult

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
UP_RED, DOWN_GREEN = "#d03b3b", "#0ca30c"
MUTED = "rgba(128,128,128,0.55)"


def _candle_colors(market: str) -> tuple[str, str]:
    return (UP_RED, DOWN_GREEN) if market == "TW" else (DOWN_GREEN, UP_RED)


def future_dates(last: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last + pd.Timedelta(days=1), periods=n)


def price_chart(prices: pd.DataFrame, market: str, forecasts: dict[str, ForecastResult] | None = None,
                champion: str | None = None, lookback: int = 160, levels: dict | None = None) -> go.Figure:
    df = add_indicators(prices).tail(lookback)
    up, down = _candle_colors(market)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.60, 0.19, 0.21],
                        subplot_titles=("", "成交量", "MACD 柱狀體"))
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 name="K線", increasing=dict(line=dict(color=up, width=1), fillcolor=up),
                                 decreasing=dict(line=dict(color=down, width=1), fillcolor=down)), row=1, col=1)
    for i, (col, label) in enumerate([("ma20", "MA20 月線"), ("ma60", "MA60 季線"), ("ma240", "MA240 年線")]):
        if df[col].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=label, mode="lines",
                                     line=dict(color=SERIES[i + 1], width=1.5),
                                     hovertemplate=f"{label} %{{y:.2f}}<extra></extra>"), row=1, col=1)
    if forecasts:
        last_t, last_p = df.index[-1], float(df["close"].iloc[-1])
        names = [champion] + [m for m in forecasts if m != champion] if champion in forecasts else list(forecasts)
        for j, m in enumerate(names):
            fc = forecasts[m]
            xs = [last_t] + list(future_dates(last_t, fc.horizon))
            color = SERIES[0] if m == champion else SERIES[(j + 3) % len(SERIES)]
            if m == champion and fc.quantiles:
                lo, hi = [last_p] + list(fc.q(0.1)), [last_p] + list(fc.q(0.9))
                fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself", fillcolor="rgba(42,120,214,0.15)",
                                         line=dict(width=0), name=f"{m} 80% 區間", hoverinfo="skip"), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=[last_p] + list(fc.point), mode="lines+markers",
                                     name=f"{m} 預測", line=dict(color=color, width=2.5 if m == champion else 1.5,
                                                                   dash="solid" if m == champion else "dot"),
                                     marker=dict(size=8 if m == champion else 6),
                                     hovertemplate=f"{m} %{{x|%m/%d}} %{{y:.2f}}<extra></extra>"), row=1, col=1)
    for name, val in (levels or {}).items():
        if val:
            fig.add_hline(y=val, line=dict(color=MUTED, width=1, dash="dash"), annotation_text=f"{name} {val:,.2f}",
                          annotation_position="top left", annotation_font_size=11, row=1, col=1)
    vol_colors = [up if c >= o else down for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=vol_colors, name="成交量", showlegend=False,
                         hovertemplate="量 %{y:,.0f}<extra></extra>"), row=2, col=1)
    osc_colors = [up if v >= 0 else down for v in df["osc"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["osc"], marker_color=osc_colors, name="MACD OSC", showlegend=False,
                         hovertemplate="OSC %{y:.3f}<extra></extra>"), row=3, col=1)
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.04, x=0))
    holidays = pd.bdate_range(df.index[0], df.index[-1]).difference(df.index)   # 春節/國定假日 gaps
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=list(holidays))])
    return fig


def leaderboard_bar(lb: pd.DataFrame, metric: str = "mase") -> go.Figure:
    d = lb.sort_values(metric)
    colors = [SERIES[0] if m != "naive" else MUTED for m in d["model"]]
    fig = go.Figure(go.Bar(x=d[metric], y=d["model"], orientation="h", marker_color=colors,
                           text=[f"{v:.3f}" for v in d[metric]], textposition="outside",
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    if "naive" in set(d["model"]):
        fig.add_vline(x=float(d.loc[d["model"] == "naive", metric].iloc[0]), line=dict(color=MUTED, dash="dash"),
                      annotation_text="random walk")
    fig.update_layout(height=60 + 36 * len(d), margin=dict(l=10, r=40, t=20, b=10), xaxis_title=metric.upper(),
                      yaxis=dict(autorange="reversed"))
    return fig


def calibration_scatter(lb: pd.DataFrame) -> go.Figure:
    """Directional accuracy vs 80%-interval coverage: good models sit right of 50% and near 80%."""
    d = lb.dropna(subset=["dir_acc", "cov80"]) if "cov80" in lb else lb.iloc[0:0]
    fig = go.Figure(go.Scatter(x=d["cov80"] * 100, y=d["dir_acc"] * 100, mode="markers+text", text=d["model"],
                               textposition="top center", marker=dict(size=12, color=SERIES[0],
                                                                      line=dict(width=2, color="white")),
                               hovertemplate="%{text}<br>覆蓋率 %{x:.1f}%<br>方向準確率 %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=50, line=dict(color=MUTED, dash="dash"), annotation_text="擲硬幣 50%")
    fig.add_vline(x=80, line=dict(color=MUTED, dash="dash"), annotation_text="理想覆蓋 80%")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="80% 區間實際覆蓋率 (%)",
                      yaxis_title="方向準確率 (%)")
    return fig
