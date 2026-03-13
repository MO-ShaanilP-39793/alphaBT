"""Aggregate insight charts: Gantt timeline, P&L waterfall, return histogram."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


_CR = 1e7

_EXIT_COLOURS = {
    "TP": "#2ca02c",
    "SL": "#d62728",
    "Time": "#7f7f7f",
    "Regime": "#9467bd",
    "Open": "#1f77b4",
}


def _classify_exit(row: pd.Series) -> str:
    if row.get("TP_triggered"):
        return "TP"
    if row.get("SL_triggered"):
        return "SL"
    if row.get("regime_exit"):
        return "Regime"
    if pd.isna(row.get("exit_date")):
        return "Open"
    return "Time"


# ---------------------------------------------------------------------------
# 1. Holdings Gantt timeline
# ---------------------------------------------------------------------------

def holdings_gantt_chart(
    trades: pd.DataFrame,
    quarter_start: pd.Timestamp,
    quarter_end: pd.Timestamp,
    as_of_date: Optional[pd.Timestamp] = None,
) -> go.Figure:
    """Horizontal bar (Gantt-style) showing each stock's holding window.

    Parameters
    ----------
    as_of_date : Timestamp, optional
        For live/partial quarters, open positions extend to this date
        instead of the full quarter end.
    """
    df = trades.copy()
    df["exit_type"] = df.apply(_classify_exit, axis=1)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    fill_date = as_of_date if as_of_date is not None else quarter_end
    df["exit_date"] = df["exit_date"].fillna(fill_date)

    df = df.sort_values("entry_date", ascending=True)

    _MS_PER_DAY = 86_400_000  # Plotly date axes use milliseconds internally

    fig = go.Figure()

    has_sector = "sector" in df.columns

    for _, row in df.iterrows():
        colour = _EXIT_COLOURS.get(row["exit_type"], "#aaa")
        ret = row.get("stock_return")
        ret_str = f"{ret * 100:.1f}%" if pd.notna(ret) else "open"
        duration_ms = (row["exit_date"] - row["entry_date"]).total_seconds() * 1000
        duration_ms = max(duration_ms, _MS_PER_DAY)
        sector_line = f"Sector: {row['sector']}<br>" if has_sector and pd.notna(row.get("sector")) else ""
        fig.add_trace(
            go.Bar(
                x=[duration_ms],
                y=[row["co_name"]],
                base=[row["entry_date"]],
                orientation="h",
                marker_color=colour,
                name=row["exit_type"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['co_name']}</b><br>"
                    f"{sector_line}"
                    f"Entry: {row['entry_date'].strftime('%b %d')}<br>"
                    f"Exit: {row['exit_date'].strftime('%b %d')}<br>"
                    f"Return: {ret_str}<br>"
                    f"Type: {row['exit_type']}<extra></extra>"
                ),
                width=0.6,
            )
        )

    # Legend entries (one per type present)
    for exit_type, colour in _EXIT_COLOURS.items():
        if exit_type in df["exit_type"].values:
            fig.add_trace(
                go.Bar(
                    x=[None], y=[None], marker_color=colour,
                    name=exit_type, showlegend=True,
                )
            )

    fig.update_layout(
        title="Holdings Timeline",
        xaxis_title="Date",
        xaxis_type="date",
        barmode="overlay",
        height=max(350, len(df) * 24 + 80),
        margin=dict(l=140, r=20, t=50, b=30),
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    )
    fig.add_vrect(
        x0=quarter_start, x1=quarter_end,
        fillcolor="rgba(0,0,0,0.03)", line_width=0,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. P&L waterfall
# ---------------------------------------------------------------------------

def pnl_waterfall_chart(trades: pd.DataFrame, initial_capital: float) -> go.Figure:
    """Waterfall chart: starting capital -> per-stock P&L -> final value."""
    df = trades.copy()
    df["exit_type"] = df.apply(_classify_exit, axis=1)

    df["pnl"] = df.apply(
        lambda r: (r["stock_return"] * r["stock_weight"] * initial_capital)
        if pd.notna(r.get("stock_return")) and pd.notna(r.get("stock_weight"))
        else 0,
        axis=1,
    )
    df = df.sort_values("pnl", ascending=False)

    # Label open positions distinctly
    labels_list = []
    for _, r in df.iterrows():
        name = r["co_name"]
        if r["exit_type"] == "Open":
            name += " (open)"
        labels_list.append(name)

    labels = ["Start"] + labels_list + ["Final"]
    values = [initial_capital] + df["pnl"].tolist() + [0]
    measures = ["absolute"] + ["relative"] * len(df) + ["total"]

    text = []
    for i, v in enumerate(values):
        if measures[i] == "absolute" or measures[i] == "total":
            text.append(f"₹{v / _CR:,.1f} Cr")
        else:
            sign = "+" if v >= 0 else ""
            text.append(f"{sign}₹{v / _CR:,.2f} Cr")

    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            text=text,
            textposition="outside",
            connector_line_color="rgba(0,0,0,0.3)",
            increasing_marker_color="#2ca02c",
            decreasing_marker_color="#d62728",
            totals_marker_color="#1f77b4",
        )
    )

    fig.update_layout(
        title="P&L Contribution by Stock",
        yaxis_title="Portfolio Value (₹)",
        height=max(400, 50 + len(df) * 18),
        margin=dict(l=60, r=20, t=50, b=80),
        xaxis_tickangle=-45,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Return distribution histogram
# ---------------------------------------------------------------------------

def return_distribution_histogram(
    trades: pd.DataFrame,
    tp_pct: float | None = None,
    sl_pct: float | None = None,
) -> go.Figure:
    """Histogram of stock returns with optional TP/SL threshold lines."""
    returns = trades["stock_return"].dropna() * 100

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns, nbinsx=20,
            marker_color="rgba(31,119,180,0.6)",
            name="Stock Returns",
            hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        )
    )

    if tp_pct is not None:
        fig.add_vline(
            x=tp_pct * 100, line_dash="dot", line_color="#2ca02c",
            annotation_text=f"TP +{tp_pct*100:.0f}%",
            annotation_position="top right",
        )
    if sl_pct is not None:
        fig.add_vline(
            x=-sl_pct * 100, line_dash="dot", line_color="#d62728",
            annotation_text=f"SL -{sl_pct*100:.0f}%",
            annotation_position="top left",
        )
    fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)

    fig.update_layout(
        title="Distribution of Stock Returns",
        xaxis_title="Return (%)", yaxis_title="Count",
        height=380, margin=dict(l=50, r=20, t=50, b=30),
    )
    return fig
