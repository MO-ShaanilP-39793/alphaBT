"""
alphaBT — Interactive Quarter Drill-Down Dashboard

Launch:
    cd src
    streamlit run visualization/dashboard.py -- --run-dir ../backtesting_results/run_XXX

Or from repo root:
    streamlit run src/visualization/dashboard.py -- --run-dir backtesting_results/run_XXX
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import pandas as pd
import streamlit as st

# Ensure src/ is on sys.path so local imports resolve when Streamlit
# launches from different working directories.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from visualization.data_loader import (
    DashboardData,
    load_from_run_dir,
    get_quarter_trades,
    get_quarter_daily,
    classify_exit,
)
from visualization.charts.portfolio import (
    portfolio_vs_index_chart,
    cash_area_chart,
    invested_vs_cash_chart,
)
from visualization.charts.trade_outcomes import (
    exit_type_donut,
    exit_by_category_bar,
    return_vs_holding_scatter,
    trade_summary_table,
)
from visualization.charts.stock_drilldown import (
    stock_candlestick_chart,
    stock_info_card,
)
from visualization.charts.aggregate import (
    holdings_gantt_chart,
    pnl_waterfall_chart,
    return_distribution_histogram,
)
from utils.quarter import get_quarter_dates
from config.defaults import INITIAL_CAPITAL

# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="alphaBT Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CR = 1e7

_QUARTER_LABELS = {
    2: "Feb 15 – May 30",
    5: "May 31 – Aug 14",
    8: "Aug 15 – Nov 14",
    11: "Nov 15 – Feb 14 (+1y)",
}


# ── Data loading (cached) ───────────────────────────────────────────────

@st.cache_data(show_spinner="Loading backtest data …")
def _load_data(run_dir: str) -> dict:
    """Load and return raw dicts (Streamlit caching needs serialisable types)."""
    data = load_from_run_dir(run_dir)
    return {
        "trade_results": data.trade_results,
        "daily_pf_values": data.daily_pf_values,
        "price_data": data.price_data,
        "index_data": data.index_data,
        "comparison_df": data.comparison_df,
        "config": data.config,
        "quarters": data.quarters,
    }


def _to_dashboard_data(raw: dict) -> DashboardData:
    return DashboardData(
        trade_results=raw["trade_results"],
        daily_pf_values=raw["daily_pf_values"],
        price_data=raw["price_data"],
        index_data=raw["index_data"],
        comparison_df=raw["comparison_df"],
        config=raw["config"],
    )


# ── Helpers ──────────────────────────────────────────────────────────────

def _quarter_label(q: int) -> str:
    year = int(str(q)[:4])
    mm = int(str(q)[4:])
    date_range = _QUARTER_LABELS.get(mm, "")
    return f"Q{q}  ({year} {date_range})"


def _fmt_cr(v: float) -> str:
    return f"₹{v / _CR:,.1f} Cr"


# ── Sidebar ──────────────────────────────────────────────────────────────

def _render_sidebar(data: DashboardData):
    st.sidebar.title("alphaBT Dashboard")
    st.sidebar.markdown("---")

    quarters = data.quarters
    selected = st.sidebar.selectbox(
        "Select Quarter",
        quarters,
        format_func=_quarter_label,
    )

    q_trades = get_quarter_trades(data, selected)
    q_daily = get_quarter_daily(data, selected)

    # Summary metrics
    n_stocks = len(q_trades)
    tp_count = int(q_trades["TP_triggered"].sum()) if "TP_triggered" in q_trades.columns else 0
    sl_count = int(q_trades["SL_triggered"].sum()) if "SL_triggered" in q_trades.columns else 0

    st.sidebar.markdown(f"**Stocks:** {n_stocks}")
    st.sidebar.markdown(f"**TP Hits:** {tp_count}  |  **SL Hits:** {sl_count}")

    pf_col = "portfolio_value" if "portfolio_value" in q_daily.columns else "Total_Portfolio_Value"
    if pf_col in q_daily.columns and len(q_daily) >= 2:
        start_val = q_daily[pf_col].iloc[0]
        end_val = q_daily[pf_col].iloc[-1]
        q_return = (end_val - start_val) / start_val * 100
        st.sidebar.metric("Quarter Return", f"{q_return:+.2f}%")
        st.sidebar.markdown(f"Start: {_fmt_cr(start_val)}  →  End: {_fmt_cr(end_val)}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Plotly + Streamlit")

    return selected


# ── Tab: Portfolio Performance ───────────────────────────────────────────

def _tab_portfolio(data: DashboardData, quarter: int):
    q_daily = get_quarter_daily(data, quarter)
    q_trades = get_quarter_trades(data, quarter)

    # Filter comparison_df to the quarter's date range
    q_start, q_end = get_quarter_dates(quarter)
    comp = None
    if data.comparison_df is not None:
        comp = data.comparison_df.copy()
        comp["date"] = pd.to_datetime(comp["date"])
        comp = comp[(comp["date"] >= q_start) & (comp["date"] <= q_end)]
        if comp.empty:
            comp = None

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(
            portfolio_vs_index_chart(q_daily, comp),
            use_container_width=True,
        )
    with col2:
        pf_col = "portfolio_value" if "portfolio_value" in q_daily.columns else "Total_Portfolio_Value"
        if pf_col in q_daily.columns and len(q_daily) >= 2:
            daily_returns = q_daily[pf_col].pct_change().dropna()
            vol = daily_returns.std() * np.sqrt(252) * 100
            max_dd = ((q_daily[pf_col] / q_daily[pf_col].cummax()) - 1).min() * 100
            st.metric("Annualised Volatility", f"{vol:.1f}%")
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
            st.metric("Total Trades", len(q_trades))

    st.plotly_chart(
        cash_area_chart(q_daily, q_trades),
        use_container_width=True,
    )
    st.plotly_chart(
        invested_vs_cash_chart(q_daily),
        use_container_width=True,
    )


# ── Tab: Trade Outcomes ──────────────────────────────────────────────────

def _tab_trade_outcomes(data: DashboardData, quarter: int):
    q_trades = get_quarter_trades(data, quarter)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(exit_type_donut(q_trades), use_container_width=True)
    with col2:
        st.plotly_chart(exit_by_category_bar(q_trades), use_container_width=True)

    st.plotly_chart(
        return_vs_holding_scatter(q_trades),
        use_container_width=True,
    )

    st.subheader("Trade Summary")
    table_df = trade_summary_table(q_trades)
    st.dataframe(table_df, use_container_width=True, height=min(600, 35 * len(table_df) + 40))


# ── Tab: Stock Drill-Down ────────────────────────────────────────────────

def _tab_stock_drilldown(data: DashboardData, quarter: int):
    q_trades = get_quarter_trades(data, quarter)
    q_start, q_end = get_quarter_dates(quarter)

    stock_names = sorted(q_trades["co_name"].unique().tolist())
    if not stock_names:
        st.info("No trades for this quarter.")
        return

    selected_stock = st.selectbox("Select Stock", stock_names)

    row = q_trades[q_trades["co_name"] == selected_stock].iloc[0]

    col_chart, col_info = st.columns([3, 1])

    with col_chart:
        fig = stock_candlestick_chart(
            selected_stock, row, data.price_data, q_start, q_end,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        info = stock_info_card(row)
        for label, value in info.items():
            st.markdown(f"**{label}:** {value}")


# ── Tab: Aggregate Insights ─────────────────────────────────────────────

def _tab_aggregate(data: DashboardData, quarter: int):
    q_trades = get_quarter_trades(data, quarter)
    q_start, q_end = get_quarter_dates(quarter)

    st.plotly_chart(
        holdings_gantt_chart(q_trades, q_start, q_end),
        use_container_width=True,
    )

    # Determine capital for this quarter (use config or default)
    capital = data.config.get("_quarter_capital", INITIAL_CAPITAL)
    st.plotly_chart(
        pnl_waterfall_chart(q_trades, capital),
        use_container_width=True,
    )

    # Try to infer TP/SL pcts for the histogram threshold lines
    tp_pct = q_trades["tp_pct_used"].dropna().median() if "tp_pct_used" in q_trades.columns else None
    sl_pct = q_trades["sl_pct_used"].dropna().median() if "sl_pct_used" in q_trades.columns else None

    st.plotly_chart(
        return_distribution_histogram(q_trades, tp_pct=tp_pct, sl_pct=sl_pct),
        use_container_width=True,
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="alphaBT Dashboard")
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to a backtest run directory (e.g. backtesting_results/run_XXX)",
    )
    # Streamlit passes its own args; parse only known ones.
    args, _ = parser.parse_known_args()

    raw = _load_data(args.run_dir)
    data = _to_dashboard_data(raw)

    quarter = _render_sidebar(data)

    tab_pf, tab_trades, tab_drill, tab_agg = st.tabs([
        "Portfolio Performance",
        "Trade Outcomes",
        "Stock Drill-Down",
        "Aggregate Insights",
    ])

    with tab_pf:
        _tab_portfolio(data, quarter)
    with tab_trades:
        _tab_trade_outcomes(data, quarter)
    with tab_drill:
        _tab_stock_drilldown(data, quarter)
    with tab_agg:
        _tab_aggregate(data, quarter)


if __name__ == "__main__":
    main()
