"""Portfolio-level Plotly charts: value vs index, cash, invested-vs-cash."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_CR = 1e7  # 1 crore


def _fmt_cr(v: float) -> str:
    return f"₹{v / _CR:,.1f} Cr"


# ---------------------------------------------------------------------------
# 1. Portfolio value vs Index
# ---------------------------------------------------------------------------

def portfolio_vs_index_chart(
    daily_pf: pd.DataFrame,
    comparison_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Dual-line chart of portfolio value and index fund value.

    Parameters
    ----------
    daily_pf : DataFrame
        Must contain columns ``date`` and ``portfolio_value``.
    comparison_df : DataFrame, optional
        If provided, must contain ``date`` and ``index_fund_value``.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.75, 0.25],
        subplot_titles=("Portfolio vs Index", "Daily Alpha (%)"),
    )

    dates = pd.to_datetime(daily_pf["date"])
    pf_val = daily_pf["portfolio_value"] if "portfolio_value" in daily_pf.columns else daily_pf.get("Total_Portfolio_Value")

    fig.add_trace(
        go.Scatter(
            x=dates, y=pf_val, name="Portfolio",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="%{x|%b %d}<br>Portfolio: %{customdata}<extra></extra>",
            customdata=[_fmt_cr(v) for v in pf_val],
        ),
        row=1, col=1,
    )

    if comparison_df is not None and "index_fund_value" in comparison_df.columns:
        idx_dates = pd.to_datetime(comparison_df["date"])
        idx_val = comparison_df["index_fund_value"]

        fig.add_trace(
            go.Scatter(
                x=idx_dates, y=idx_val, name="Index Fund",
                line=dict(color="#ff7f0e", width=2, dash="dot"),
                hovertemplate="%{x|%b %d}<br>Index: %{customdata}<extra></extra>",
                customdata=[_fmt_cr(v) for v in idx_val],
            ),
            row=1, col=1,
        )

        if "alpha" in comparison_df.columns:
            alpha = comparison_df["alpha"]
            colours = ["#2ca02c" if a >= 0 else "#d62728" for a in alpha]
            fig.add_trace(
                go.Bar(
                    x=idx_dates, y=alpha, name="Alpha %",
                    marker_color=colours, opacity=0.6,
                    hovertemplate="%{x|%b %d}<br>Alpha: %{y:.2f}%<extra></extra>",
                ),
                row=2, col=1,
            )

    fig.update_yaxes(title_text="Value (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Alpha %", row=2, col=1)
    fig.update_layout(
        height=560, legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=60, r=20, t=40, b=30), hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Cash in hand area chart
# ---------------------------------------------------------------------------

def cash_area_chart(
    daily_pf: pd.DataFrame,
    trade_results: pd.DataFrame | None = None,
) -> go.Figure:
    """Area chart of cash in hand with optional annotations at exit events.

    Parameters
    ----------
    daily_pf : DataFrame
        ``date``, ``cash_in_hand`` (or ``Cash_In_Hand``).
    trade_results : DataFrame, optional
        If provided, exit events are annotated on the chart.
    """
    dates = pd.to_datetime(daily_pf["date"])
    cash_col = "cash_in_hand" if "cash_in_hand" in daily_pf.columns else "Cash_In_Hand"
    cash = daily_pf[cash_col]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates, y=cash, fill="tozeroy", name="Cash in Hand",
            line=dict(color="#17becf", width=1.5),
            fillcolor="rgba(23,190,207,0.25)",
            hovertemplate="%{x|%b %d}<br>Cash: %{customdata}<extra></extra>",
            customdata=[_fmt_cr(v) for v in cash],
        )
    )

    if trade_results is not None:
        exits = trade_results.dropna(subset=["exit_date"]).copy()
        exits["exit_date"] = pd.to_datetime(exits["exit_date"])
        for _, row in exits.iterrows():
            exit_type = _exit_label(row)
            colour = {"TP": "#2ca02c", "SL": "#d62728"}.get(exit_type, "#7f7f7f")
            fig.add_trace(
                go.Scatter(
                    x=[row["exit_date"]], y=[None],
                    mode="markers", marker=dict(size=7, color=colour, symbol="triangle-up"),
                    name=f"{row['co_name']} ({exit_type})",
                    showlegend=False,
                    hovertemplate=(
                        f"{row['co_name']}<br>"
                        f"Exit: {exit_type}<br>"
                        f"Date: {row['exit_date'].strftime('%b %d')}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title="Cash in Hand Over the Quarter",
        yaxis_title="Cash (₹)", xaxis_title="Date",
        height=380, margin=dict(l=60, r=20, t=40, b=30),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Invested vs Cash stacked area
# ---------------------------------------------------------------------------

def invested_vs_cash_chart(daily_pf: pd.DataFrame) -> go.Figure:
    """Stacked area showing invested capital vs idle cash."""
    dates = pd.to_datetime(daily_pf["date"])

    pf_col = "portfolio_value" if "portfolio_value" in daily_pf.columns else "Total_Portfolio_Value"
    cash_col = "cash_in_hand" if "cash_in_hand" in daily_pf.columns else "Cash_In_Hand"

    total = daily_pf[pf_col]
    cash = daily_pf[cash_col]
    invested = total - cash

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates, y=invested, name="Invested",
            stackgroup="one", line=dict(width=0),
            fillcolor="rgba(31,119,180,0.5)",
            hovertemplate="%{x|%b %d}<br>Invested: %{customdata}<extra></extra>",
            customdata=[_fmt_cr(v) for v in invested],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=cash, name="Cash",
            stackgroup="one", line=dict(width=0),
            fillcolor="rgba(23,190,207,0.5)",
            hovertemplate="%{x|%b %d}<br>Cash: %{customdata}<extra></extra>",
            customdata=[_fmt_cr(v) for v in cash],
        )
    )

    fig.update_layout(
        title="Capital Split: Invested vs Cash",
        yaxis_title="Value (₹)", xaxis_title="Date",
        height=380, margin=dict(l=60, r=20, t=40, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exit_label(row: pd.Series) -> str:
    if row.get("TP_triggered"):
        return "TP"
    if row.get("SL_triggered"):
        return "SL"
    if row.get("regime_exit"):
        return "Regime"
    return "Time"
