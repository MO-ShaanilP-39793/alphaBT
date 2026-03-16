"""Cross-quarter Plotly charts for the alphaBT dashboard."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_CR = 1e7

_MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _fmt_cr(v: float) -> str:
    return f"₹{v / _CR:,.1f} Cr"


# ---------------------------------------------------------------------------
# 1. Full-period equity curve
# ---------------------------------------------------------------------------

def equity_curve_chart(
    daily_pf: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Full-period portfolio vs index equity curve with daily alpha bars."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.75, 0.25],
        subplot_titles=("Portfolio vs Index — Full Period", "Daily Alpha (%)"),
    )

    dates = pd.to_datetime(daily_pf["date"])
    pf_col = "portfolio_value" if "portfolio_value" in daily_pf.columns else "Total_Portfolio_Value"
    pf_val = daily_pf[pf_col]

    fig.add_trace(
        go.Scatter(
            x=dates, y=pf_val, name="Portfolio",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="%{x|%b %Y}<br>Portfolio: %{customdata}<extra></extra>",
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
                hovertemplate="%{x|%b %Y}<br>Index: %{customdata}<extra></extra>",
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
                    hovertemplate="%{x|%b %Y}<br>Alpha: %{y:.2f}%<extra></extra>",
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
# 2. Full-period drawdown chart
# ---------------------------------------------------------------------------

def drawdown_chart(
    drawdown_series: pd.DataFrame,
    index_drawdown_series: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Area chart of running drawdown percentage from peak (portfolio + index)."""
    dates = pd.to_datetime(drawdown_series["date"])
    dd_pct = drawdown_series["drawdown_pct"]

    fig = go.Figure()

    if index_drawdown_series is not None and not index_drawdown_series.empty:
        idx_dates = pd.to_datetime(index_drawdown_series["date"])
        idx_dd_pct = index_drawdown_series["drawdown_pct"]
        fig.add_trace(
            go.Scatter(
                x=idx_dates, y=idx_dd_pct, fill="tozeroy", name="Index",
                line=dict(color="#ff7f0e", width=1.2, dash="dot"),
                fillcolor="rgba(255,127,14,0.12)",
                hovertemplate="%{x|%b %Y}<br>Index DD: %{y:.2f}%<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dates, y=dd_pct, fill="tozeroy", name="Portfolio",
            line=dict(color="#d62728", width=1.5),
            fillcolor="rgba(214,39,40,0.25)",
            hovertemplate="%{x|%b %Y}<br>Portfolio DD: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Drawdown From Peak — Portfolio vs Index",
        yaxis_title="Drawdown (%)", xaxis_title="Date",
        height=380, margin=dict(l=60, r=20, t=50, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Quarter-over-quarter performance bar chart
# ---------------------------------------------------------------------------

def qoq_performance_chart(quarterly_alpha: Optional[pd.DataFrame]) -> go.Figure:
    """Grouped bar for portfolio/benchmark return per quarter + alpha line.

    Expects the ``quarterly_alpha`` DataFrame from DashboardData (columns:
    Quarter, Portfolio_Return, Benchmark_Return, Outperformance).  Values
    are in decimal form (0.05 = 5%).
    """
    if quarterly_alpha is None or quarterly_alpha.empty:
        fig = go.Figure()
        fig.add_annotation(text="No quarterly alpha data available", showarrow=False)
        fig.update_layout(height=380)
        return fig

    df = quarterly_alpha.copy()
    df["Quarter"] = df["Quarter"].astype(str)
    pf_pct = df["Portfolio_Return"] * 100
    has_bench = "Benchmark_Return" in df.columns and df["Benchmark_Return"].notna().any()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df["Quarter"], y=pf_pct, name="Portfolio Return",
            marker_color="#1f77b4", opacity=0.85,
            hovertemplate="Q%{x}<br>Portfolio: %{y:+.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )

    if has_bench:
        bm_pct = df["Benchmark_Return"] * 100
        fig.add_trace(
            go.Bar(
                x=df["Quarter"], y=bm_pct, name="Benchmark Return",
                marker_color="#ff7f0e", opacity=0.85,
                hovertemplate="Q%{x}<br>Benchmark: %{y:+.2f}%<extra></extra>",
            ),
            secondary_y=False,
        )

    if "Outperformance" in df.columns:
        alpha_pct = df["Outperformance"] * 100
        fig.add_trace(
            go.Scatter(
                x=df["Quarter"], y=alpha_pct, name="Alpha",
                mode="lines+markers",
                line=dict(color="#2ca02c", width=2),
                marker=dict(size=7),
                hovertemplate="Q%{x}<br>Alpha: %{y:+.2f}%<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Quarter-over-Quarter Performance",
        barmode="group", height=440,
        margin=dict(l=60, r=60, t=50, b=30),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Return (%)", secondary_y=False)
    fig.update_yaxes(title_text="Alpha (%)", secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# 4. Calendar year performance
# ---------------------------------------------------------------------------

def calendar_year_chart(cal_year_df: pd.DataFrame) -> go.Figure:
    """Horizontal grouped bar of annualized return per calendar year."""
    if cal_year_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for calendar-year analysis (need >=1 full year)",
            showarrow=False,
        )
        fig.update_layout(height=380)
        return fig

    years = [str(p) for p in cal_year_df.index]
    fig = go.Figure()

    colours = {"Portfolio": "#1f77b4", "Benchmark": "#ff7f0e"}
    for col in cal_year_df.columns:
        fig.add_trace(
            go.Bar(
                y=years, x=cal_year_df[col], name=col, orientation="h",
                marker_color=colours.get(col, "#17becf"),
                hovertemplate="%{y}<br>" + col + ": %{x:+.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Calendar Year Returns (%)",
        barmode="group", height=max(320, len(years) * 50 + 80),
        margin=dict(l=80, r=20, t=50, b=30),
        xaxis_title="Return (%)",
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Monthly returns heatmap
# ---------------------------------------------------------------------------

def monthly_returns_heatmap(monthly_returns: pd.DataFrame) -> go.Figure:
    """Year x Month heatmap of portfolio monthly returns (%).

    ``monthly_returns`` index should be date-like; expects a 'Portfolio'
    column with returns already scaled to percent (×100).
    """
    col = "Portfolio" if "Portfolio" in monthly_returns.columns else monthly_returns.columns[0]
    df = monthly_returns[[col]].copy()
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot_table(index="year", columns="month", values=col, aggfunc="sum")
    pivot = pivot.reindex(columns=range(1, 13))

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=_MONTH_LABELS,
            y=[str(y) for y in pivot.index],
            colorscale=[[0, "#d62728"], [0.5, "#f7f7f7"], [1, "#2ca02c"]],
            zmid=0,
            text=np.where(np.isnan(pivot.values), "", np.vectorize(lambda v: f"{v:+.1f}%")(pivot.values)),
            texttemplate="%{text}",
            hovertemplate="Year %{y}, %{x}<br>Return: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="Return %"),
        )
    )

    fig.update_layout(
        title="Monthly Returns Heatmap (%)",
        height=max(300, len(pivot) * 40 + 100),
        margin=dict(l=60, r=20, t=50, b=30),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Cash allocation trend across quarters
# ---------------------------------------------------------------------------

def cash_trend_chart(cash_by_quarter: Optional[pd.DataFrame]) -> go.Figure:
    """Line chart of mean cash % per quarter with optional snapshot lines."""
    if cash_by_quarter is None or cash_by_quarter.empty:
        fig = go.Figure()
        fig.add_annotation(text="Cash data not available for this run", showarrow=False)
        fig.update_layout(height=380)
        return fig

    df = cash_by_quarter.copy()
    qs = df["quarter"].astype(str)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=qs, y=df["mean_cash_pct"], name="Avg Cash %",
            mode="lines+markers",
            line=dict(color="#17becf", width=2), marker=dict(size=7),
            hovertemplate="Q%{x}<br>Avg Cash: %{y:.1f}%<extra></extra>",
        )
    )

    if "cash_pct_1m" in df.columns and df["cash_pct_1m"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=qs, y=df["cash_pct_1m"], name="Cash % @ 1 Month",
                mode="lines+markers", line=dict(dash="dash", color="#9467bd"),
                hovertemplate="Q%{x}<br>1M Cash: %{y:.1f}%<extra></extra>",
            )
        )
    if "cash_pct_2m" in df.columns and df["cash_pct_2m"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=qs, y=df["cash_pct_2m"], name="Cash % @ 2 Months",
                mode="lines+markers", line=dict(dash="dot", color="#8c564b"),
                hovertemplate="Q%{x}<br>2M Cash: %{y:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Cash Allocation Trend Across Quarters",
        yaxis_title="Cash (% of Portfolio)", xaxis_title="Quarter",
        height=400, margin=dict(l=60, r=20, t=50, b=30),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Portfolio churn (retention) chart
# ---------------------------------------------------------------------------

def churn_chart(churn_df: pd.DataFrame) -> go.Figure:
    """Stacked bar of retained/new stocks per quarter + retention % line."""
    if churn_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough quarters for churn analysis (need >=2)",
            showarrow=False,
        )
        fig.update_layout(height=380)
        return fig

    df = churn_df[churn_df["quarter"] != "Average"].copy()
    qs = df["quarter"].astype(str)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=qs, y=df["num_retained"], name="Retained",
            marker_color="#2ca02c", opacity=0.85,
            hovertemplate="Q%{x}<br>Retained: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=qs, y=df["num_new"], name="New",
            marker_color="#ff7f0e", opacity=0.85,
            hovertemplate="Q%{x}<br>New: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=qs, y=df["retention_pct"], name="Retention %",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2), marker=dict(size=7),
            hovertemplate="Q%{x}<br>Retention: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Portfolio Churn: Stock Retention Across Quarters",
        barmode="stack", height=440,
        margin=dict(l=60, r=60, t=50, b=30),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Stock Count", secondary_y=False)
    fig.update_yaxes(title_text="Retention %", range=[0, 105], secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# 8. Return distribution (histogram + box plot)
# ---------------------------------------------------------------------------

def return_distribution_charts(monthly_returns: pd.DataFrame) -> go.Figure:
    """Side-by-side histogram and box plot of monthly returns."""
    if monthly_returns.empty:
        fig = go.Figure()
        fig.add_annotation(text="No monthly return data available", showarrow=False)
        fig.update_layout(height=380)
        return fig

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False,
        subplot_titles=("Monthly Return Distribution", "Monthly Return Spread"),
        column_widths=[0.6, 0.4],
    )

    has_pf = "Portfolio" in monthly_returns.columns
    has_bm = "Benchmark" in monthly_returns.columns

    if has_pf:
        pf = monthly_returns["Portfolio"] * 100
        fig.add_trace(
            go.Histogram(
                x=pf, name="Portfolio", nbinsx=20,
                marker_color="rgba(31,119,180,0.6)",
                hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Box(y=pf, name="Portfolio", marker_color="#1f77b4", boxmean=True),
            row=1, col=2,
        )
    if has_bm:
        bm = monthly_returns["Benchmark"] * 100
        fig.add_trace(
            go.Histogram(
                x=bm, name="Benchmark", nbinsx=20,
                marker_color="rgba(255,127,14,0.5)",
                hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Box(y=bm, name="Benchmark", marker_color="#ff7f0e", boxmean=True),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Monthly Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Monthly Return (%)", row=1, col=2)
    fig.update_layout(
        height=420, barmode="overlay",
        margin=dict(l=50, r=20, t=50, b=30),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return fig
