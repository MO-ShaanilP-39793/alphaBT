"""Sector-level analysis charts: allocation donut, performance bar, win rate, exit breakdown."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


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
# 1. Sector Allocation Donut
# ---------------------------------------------------------------------------

def sector_allocation_donut(trades: pd.DataFrame) -> go.Figure:
    """Donut chart of capital weight allocation by sector."""
    alloc = trades.groupby("sector")["stock_weight"].sum().sort_values(ascending=False)

    fig = go.Figure(
        go.Pie(
            labels=alloc.index.tolist(),
            values=alloc.values.tolist(),
            hole=0.45,
            textinfo="label+percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Weight: %{value:.4f} (%{percent})<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Sector Allocation (by Weight)",
        height=400, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Sector Performance Bar
# ---------------------------------------------------------------------------

def sector_performance_bar(trades: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of average return per sector, sorted descending."""
    closed = trades.dropna(subset=["stock_return"])
    if closed.empty:
        fig = go.Figure()
        fig.update_layout(title="Sector Avg Return (no closed trades)", height=380)
        return fig

    avg_ret = closed.groupby("sector")["stock_return"].mean().sort_values()
    colours = ["#2ca02c" if v >= 0 else "#d62728" for v in avg_ret.values]

    fig = go.Figure(
        go.Bar(
            x=avg_ret.values * 100,
            y=avg_ret.index.tolist(),
            orientation="h",
            marker_color=colours,
            text=[f"{v * 100:+.2f}%" for v in avg_ret.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Avg Return: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Average Return by Sector",
        xaxis_title="Return (%)",
        height=max(350, len(avg_ret) * 36 + 80),
        margin=dict(l=140, r=60, t=50, b=30),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
    return fig


# ---------------------------------------------------------------------------
# 3. Sector Win Rate Bar
# ---------------------------------------------------------------------------

def sector_win_rate_bar(trades: pd.DataFrame) -> go.Figure:
    """Bar chart of TP hit rate per sector, annotated with trade count."""
    df = trades.copy()
    df["exit_type"] = df.apply(_classify_exit, axis=1)

    sector_stats = df.groupby("sector").agg(
        total=("exit_type", "size"),
        tp_count=("exit_type", lambda s: (s == "TP").sum()),
    )
    sector_stats["win_rate"] = sector_stats["tp_count"] / sector_stats["total"] * 100
    sector_stats = sector_stats.sort_values("win_rate", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=sector_stats["win_rate"],
            y=sector_stats.index.tolist(),
            orientation="h",
            marker_color="#2ca02c",
            text=[
                f"{wr:.0f}% ({n} trades)"
                for wr, n in zip(sector_stats["win_rate"], sector_stats["total"])
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Win Rate: %{x:.1f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="TP Hit Rate by Sector",
        xaxis_title="Win Rate (%)",
        xaxis_range=[0, max(110, sector_stats["win_rate"].max() + 15)],
        height=max(350, len(sector_stats) * 36 + 80),
        margin=dict(l=140, r=80, t=50, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Sector Exit Breakdown (Stacked Bar)
# ---------------------------------------------------------------------------

def sector_exit_breakdown(trades: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of exit-type counts per sector."""
    df = trades.copy()
    df["exit_type"] = df.apply(_classify_exit, axis=1)

    grouped = df.groupby(["sector", "exit_type"]).size().unstack(fill_value=0)

    fig = go.Figure()
    for exit_type in ["TP", "SL", "Time", "Regime", "Open"]:
        if exit_type not in grouped.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=grouped.index.tolist(),
                y=grouped[exit_type].tolist(),
                name=exit_type,
                marker_color=_EXIT_COLOURS.get(exit_type, "#aaa"),
                hovertemplate="%{x}<br>%{fullData.name}: %{y}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Exit Type Breakdown by Sector",
        xaxis_title="Sector", yaxis_title="Count",
        height=420, margin=dict(l=50, r=20, t=50, b=80),
        xaxis_tickangle=-30,
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
    )
    return fig
