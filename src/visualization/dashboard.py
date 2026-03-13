"""
alphaBT — Interactive Quarter Drill-Down Dashboard

Launch:
    cd src
    streamlit run visualization/dashboard.py -- --run-dir ../backtesting_results/run_XXX

Also works with get_portfolio --with-levels output:
    streamlit run visualization/dashboard.py -- --run-dir ../selected_stocks_data/strategy_config_XXX

Or from repo root:
    streamlit run src/visualization/dashboard.py -- --run-dir backtesting_results/run_XXX
"""

from __future__ import annotations

import argparse
import sys
import os

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
from visualization.charts.sector_analysis import (
    sector_allocation_donut,
    sector_performance_bar,
    sector_exit_breakdown,
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
        "is_live_quarter": data.is_live_quarter,
        "as_of_date": data.as_of_date,
    }


def _to_dashboard_data(raw: dict) -> DashboardData:
    return DashboardData(
        trade_results=raw["trade_results"],
        daily_pf_values=raw["daily_pf_values"],
        price_data=raw["price_data"],
        index_data=raw["index_data"],
        comparison_df=raw["comparison_df"],
        config=raw["config"],
        is_live_quarter=raw.get("is_live_quarter", False),
        as_of_date=raw.get("as_of_date"),
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

    if data.is_live_quarter:
        st.sidebar.markdown("**:blue[LIVE QUARTER]**")
        if data.as_of_date is not None:
            st.sidebar.caption(f"Data as of {data.as_of_date.strftime('%Y-%m-%d')}")

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
    open_count = int((q_trades["exit_date"].isna()).sum()) if "exit_date" in q_trades.columns else 0

    st.sidebar.markdown(f"**Stocks:** {n_stocks}")
    st.sidebar.markdown(f"**TP Hits:** {tp_count}  |  **SL Hits:** {sl_count}")
    if open_count > 0:
        st.sidebar.markdown(f"**Still Open:** {open_count}")

    pf_col = "portfolio_value" if "portfolio_value" in q_daily.columns else "Total_Portfolio_Value"
    if pf_col in q_daily.columns and len(q_daily) >= 2:
        start_val = q_daily[pf_col].iloc[0]
        end_val = q_daily[pf_col].iloc[-1]
        q_return = (end_val - start_val) / start_val * 100
        label = "Return (so far)" if data.is_live_quarter else "Quarter Return"
        st.sidebar.metric(label, f"{q_return:+.2f}%")
        st.sidebar.markdown(f"Start: {_fmt_cr(start_val)}  →  End: {_fmt_cr(end_val)}")

    # Sector filter (only when sector column exists)
    sector_filter = []
    if "sector" in q_trades.columns:
        all_sectors = sorted(q_trades["sector"].dropna().unique().tolist())
        if all_sectors:
            sector_filter = st.sidebar.multiselect(
                "Filter by Sector", all_sectors, default=[],
                help="Leave empty to show all sectors",
            )

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Plotly + Streamlit")

    return selected, sector_filter


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
            width="stretch",
        )
    with col2:
        pf_col = "portfolio_value" if "portfolio_value" in q_daily.columns else "Total_Portfolio_Value"
        if pf_col in q_daily.columns and len(q_daily) >= 2:
            pf_start = q_daily[pf_col].iloc[0]
            pf_end = q_daily[pf_col].iloc[-1]
            pf_ret = (pf_end - pf_start) / pf_start * 100
            st.metric("Portfolio Return", f"{pf_ret:+.2f}%")

        if comp is not None and "index_fund_value" in comp.columns and len(comp) >= 2:
            idx_start = comp["index_fund_value"].iloc[0]
            idx_end = comp["index_fund_value"].iloc[-1]
            idx_ret = (idx_end - idx_start) / idx_start * 100
            st.metric("Index Return", f"{idx_ret:+.2f}%")

    st.plotly_chart(
        cash_area_chart(q_daily, q_trades),
        width="stretch",
    )
    st.plotly_chart(
        invested_vs_cash_chart(q_daily),
        width="stretch",
    )


# ── Tab: Trade Outcomes ──────────────────────────────────────────────────

def _tab_trade_outcomes(data: DashboardData, quarter: int):
    q_trades = get_quarter_trades(data, quarter)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(exit_type_donut(q_trades), width="stretch")
    with col2:
        st.plotly_chart(exit_by_category_bar(q_trades), width="stretch")

    st.plotly_chart(
        return_vs_holding_scatter(q_trades),
        width="stretch",
    )

    st.subheader("Trade Summary")
    table_df = trade_summary_table(q_trades)
    st.dataframe(table_df, width="stretch", height=min(600, 35 * len(table_df) + 40))


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
        st.plotly_chart(fig, width="stretch")

    with col_info:
        info = stock_info_card(row)
        for label, value in info.items():
            st.markdown(f"**{label}:** {value}")


# ── Tab: Sector Analysis ─────────────────────────────────────────────────

def _tab_sector_analysis(data: DashboardData, quarter: int):
    q_trades = get_quarter_trades(data, quarter)

    if "sector" not in q_trades.columns or q_trades["sector"].dropna().empty:
        st.info("Sector data is not available for this run.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(sector_allocation_donut(q_trades), width="stretch")
    with col2:
        st.plotly_chart(sector_performance_bar(q_trades), width="stretch")

    st.plotly_chart(sector_exit_breakdown(q_trades), width="stretch")


# ── Tab: Aggregate Insights ─────────────────────────────────────────────

def _tab_aggregate(data: DashboardData, quarter: int):
    q_trades = get_quarter_trades(data, quarter)
    q_start, q_end = get_quarter_dates(quarter)

    st.plotly_chart(
        holdings_gantt_chart(q_trades, q_start, q_end, as_of_date=data.as_of_date),
        width="stretch",
    )

    # Determine capital for this quarter (use config or default)
    capital = data.config.get("_quarter_capital", INITIAL_CAPITAL)
    st.plotly_chart(
        pnl_waterfall_chart(q_trades, capital),
        width="stretch",
    )

    # Try to infer TP/SL pcts for the histogram threshold lines
    tp_pct = q_trades["tp_pct_used"].dropna().median() if "tp_pct_used" in q_trades.columns else None
    sl_pct = q_trades["sl_pct_used"].dropna().median() if "sl_pct_used" in q_trades.columns else None

    st.plotly_chart(
        return_distribution_histogram(q_trades, tp_pct=tp_pct, sl_pct=sl_pct),
        width="stretch",
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="alphaBT Dashboard")
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to a backtest run directory or get_portfolio --with-levels output folder",
    )
    # Streamlit passes its own args; parse only known ones.
    args, _ = parser.parse_known_args()

    raw = _load_data(args.run_dir)
    data = _to_dashboard_data(raw)

    quarter, sector_filter = _render_sidebar(data)

    if sector_filter:
        filtered_tr = data.trade_results[
            data.trade_results["sector"].isin(sector_filter)
        ]
        data = DashboardData(
            trade_results=filtered_tr,
            daily_pf_values=data.daily_pf_values,
            price_data=data.price_data,
            index_data=data.index_data,
            comparison_df=data.comparison_df,
            config=data.config,
            is_live_quarter=data.is_live_quarter,
            as_of_date=data.as_of_date,
        )

    tab_pf, tab_trades, tab_sector, tab_drill, tab_agg = st.tabs([
        "Portfolio Performance",
        "Trade Outcomes",
        "Sector Analysis",
        "Stock Drill-Down",
        "Aggregate Insights",
    ])

    with tab_pf:
        _tab_portfolio(data, quarter)
    with tab_trades:
        _tab_trade_outcomes(data, quarter)
    with tab_sector:
        _tab_sector_analysis(data, quarter)
    with tab_drill:
        _tab_stock_drilldown(data, quarter)
    with tab_agg:
        _tab_aggregate(data, quarter)


if __name__ == "__main__":
    main()
