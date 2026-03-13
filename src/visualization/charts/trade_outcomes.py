"""Trade-outcome Plotly charts: exit-type donut, category bar, scatter, table."""

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


def _add_exit_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["exit_type"] = out.apply(_classify_exit, axis=1)
    return out


# ---------------------------------------------------------------------------
# 1. Exit-type donut
# ---------------------------------------------------------------------------

def exit_type_donut(trades: pd.DataFrame) -> go.Figure:
    """Donut chart of exit-type counts for a quarter's trades."""
    df = _add_exit_type(trades)
    counts = df["exit_type"].value_counts()

    labels = counts.index.tolist()
    values = counts.values.tolist()
    colours = [_EXIT_COLOURS.get(l, "#aaa") for l in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels, values=values, hole=0.45,
            marker=dict(colors=colours),
            textinfo="label+value+percent",
            hovertemplate="%{label}: %{value} trades (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        title="Exit Type Breakdown",
        height=360, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Exit counts by category (grouped bar)
# ---------------------------------------------------------------------------

def exit_by_category_bar(trades: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: exit-type counts per stock category."""
    df = _add_exit_type(trades)

    cat_col = "cat" if "cat" in df.columns else None
    if cat_col is None:
        cat_col = "category" if "category" in df.columns else None
    if cat_col is None:
        df["_cat"] = "All"
        cat_col = "_cat"

    grouped = df.groupby([cat_col, "exit_type"]).size().unstack(fill_value=0)

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
        barmode="group",
        title="Exits by Category",
        xaxis_title="Category", yaxis_title="Count",
        height=380, margin=dict(l=50, r=20, t=50, b=30),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Return vs holding-period scatter
# ---------------------------------------------------------------------------

def return_vs_holding_scatter(trades: pd.DataFrame) -> go.Figure:
    """Scatter plot of stock return vs holding period, coloured by exit type."""
    df = _add_exit_type(trades)
    df = df.dropna(subset=["stock_return", "holding_period"])

    has_sector = "sector" in df.columns
    sector_hover = "Sector: %{customdata[0]}<br>" if has_sector else ""

    fig = go.Figure()

    for exit_type in ["TP", "SL", "Time", "Regime", "Open"]:
        subset = df[df["exit_type"] == exit_type]
        if subset.empty:
            continue
        trace_kwargs = dict(
            x=subset["holding_period"],
            y=subset["stock_return"] * 100,
            mode="markers",
            name=exit_type,
            marker=dict(
                size=9, color=_EXIT_COLOURS.get(exit_type, "#aaa"),
                line=dict(width=0.5, color="white"),
            ),
            text=subset["co_name"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                + sector_hover
                + "Return: %{y:.2f}%<br>"
                "Holding: %{x} days<extra></extra>"
            ),
        )
        if has_sector:
            trace_kwargs["customdata"] = subset[["sector"]].values
        fig.add_trace(go.Scatter(**trace_kwargs))

    fig.update_layout(
        title="Stock Return vs Holding Period",
        xaxis_title="Holding Period (days)",
        yaxis_title="Return (%)",
        height=420, margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    return fig


# ---------------------------------------------------------------------------
# 4. Prepare table data (for Streamlit st.dataframe)
# ---------------------------------------------------------------------------

def trade_summary_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready DataFrame for the quarter's trade summary."""
    df = _add_exit_type(trades).copy()

    display_cols = ["co_name"]
    if "sector" in df.columns:
        display_cols.append("sector")
    if "cat" in df.columns:
        display_cols.append("cat")
    display_cols += [
        "entry_price", "exit_price", "stock_return",
        "exit_type", "holding_period",
    ]
    for extra in ("tp_pct_used", "sl_pct_used", "stock_weight"):
        if extra in df.columns:
            display_cols.append(extra)

    available = [c for c in display_cols if c in df.columns]
    out = df[available].copy()

    rename_map = {
        "co_name": "Stock",
        "sector": "Sector",
        "cat": "Category",
        "entry_price": "Entry ₹",
        "exit_price": "Exit ₹",
        "stock_return": "Return",
        "exit_type": "Exit Type",
        "holding_period": "Days Held",
        "tp_pct_used": "TP %",
        "sl_pct_used": "SL %",
        "stock_weight": "Weight",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    if "Return" in out.columns:
        out["Return"] = out["Return"].apply(
            lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "—"
        )
    for col in ("Entry ₹", "Exit ₹"):
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
    for col in ("TP %", "SL %"):
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "—"
            )
    if "Weight" in out.columns:
        out["Weight"] = out["Weight"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

    return out.reset_index(drop=True)
