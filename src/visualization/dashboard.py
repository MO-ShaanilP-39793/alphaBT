"""
alphaBT — Interactive Quarter Drill-Down Dashboard

Launch (single run):
    streamlit run src/visualization/dashboard.py -- --run-dir backtesting_results/run_XXX

Launch (hub — browse all runs in a directory):
    streamlit run src/visualization/hub.py

Also works with get_portfolio --with-levels output:
    streamlit run src/visualization/dashboard.py -- --run-dir selected_stocks_data/strategy_config_XXX
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
    CrossQuarterData,
    load_from_run_dir,
    get_quarter_trades,
    get_quarter_daily,
    classify_exit,
    compute_cross_quarter_data,
)
from visualization.charts.portfolio import (
    portfolio_vs_index_chart,
    cash_area_chart,
    invested_vs_cash_chart,
)
from visualization.charts.trade_outcomes import (
    exit_type_donut,
    exit_by_category_bar,
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
from visualization.charts.cross_quarter import (
    equity_curve_chart,
    drawdown_chart,
    qoq_performance_chart,
    calendar_year_chart,
    monthly_returns_heatmap,
    cash_trend_chart,
    churn_chart,
    return_distribution_charts,
)
from utils.quarter import get_quarter_dates
from config.defaults import INITIAL_CAPITAL

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
        "quarterly_alpha": data.quarterly_alpha,
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
        quarterly_alpha=raw.get("quarterly_alpha"),
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


def _lookup_quarterly_return(data: DashboardData, quarter: int):
    """Look up portfolio and benchmark returns from the quarterly_alpha table.

    Returns (portfolio_return_pct, benchmark_return_pct) or (None, None) when
    the table is unavailable (e.g. live-quarter / portfolio-only runs).
    Values are returned as percentages (e.g. 5.0 for 5%).
    """
    qa = data.quarterly_alpha
    if qa is None or qa.empty:
        return None, None
    row = qa.loc[qa["Quarter"] == quarter]
    if row.empty:
        return None, None
    r = row.iloc[0]
    pf = r.get("Portfolio_Return")
    bm = r.get("Benchmark_Return")
    pf_pct = pf * 100 if pd.notna(pf) else None
    bm_pct = bm * 100 if pd.notna(bm) else None
    return pf_pct, bm_pct


# ── Sidebar ──────────────────────────────────────────────────────────────

def _render_sidebar(data: DashboardData):
    st.sidebar.title("alphaBT Dashboard")

    if data.is_live_quarter:
        st.sidebar.markdown("**:blue[LIVE QUARTER]**")
        if data.as_of_date is not None:
            st.sidebar.caption(f"Data as of {data.as_of_date.strftime('%Y-%m-%d')}")

    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Dashboard Mode",
        ["Per Quarter", "Since Inception"],
        horizontal=True,
    )

    selected = None
    sector_filter = []

    if mode == "Per Quarter":
        quarters = data.quarters
        selected = st.sidebar.selectbox(
            "Select Quarter",
            quarters,
            format_func=_quarter_label,
        )

        q_trades = get_quarter_trades(data, selected)
        q_daily = get_quarter_daily(data, selected)

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

            qa_pf, _ = _lookup_quarterly_return(data, selected)
            if qa_pf is not None:
                q_return = qa_pf
            else:
                q_return = (end_val - start_val) / start_val * 100

            label = "Return (so far)" if data.is_live_quarter else "Quarter Return"
            st.sidebar.metric(label, f"{q_return:+.2f}%")
            st.sidebar.markdown(f"Start: {_fmt_cr(start_val)}  →  End: {_fmt_cr(end_val)}")

        if "sector" in q_trades.columns:
            all_sectors = sorted(q_trades["sector"].dropna().unique().tolist())
            if all_sectors:
                sector_filter = st.sidebar.multiselect(
                    "Filter by Sector", all_sectors, default=[],
                    help="Leave empty to show all sectors",
                )
    else:
        # Since Inception mode — show backtest date range summary
        daily_pf = data.daily_pf_values
        if "date" in daily_pf.columns and len(daily_pf) > 0:
            dates = pd.to_datetime(daily_pf["date"])
            st.sidebar.markdown(
                f"**Period:** {dates.min().strftime('%Y-%m-%d')} → "
                f"{dates.max().strftime('%Y-%m-%d')}"
            )
        st.sidebar.markdown(f"**Quarters:** {len(data.quarters)}")
        st.sidebar.markdown(f"**Total Trades:** {len(data.trade_results)}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Plotly + Streamlit")

    return mode, selected, sector_filter


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
        qa_pf, qa_bm = _lookup_quarterly_return(data, quarter)

        pf_col = "portfolio_value" if "portfolio_value" in q_daily.columns else "Total_Portfolio_Value"
        if pf_col in q_daily.columns and len(q_daily) >= 2:
            if qa_pf is not None:
                pf_ret = qa_pf
            else:
                pf_start = q_daily[pf_col].iloc[0]
                pf_end = q_daily[pf_col].iloc[-1]
                pf_ret = (pf_end - pf_start) / pf_start * 100
            st.metric("Portfolio Return", f"{pf_ret:+.2f}%")

        if qa_bm is not None:
            st.metric("Index Return", f"{qa_bm:+.2f}%")
        elif comp is not None and "index_fund_value" in comp.columns and len(comp) >= 2:
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


# ── Cross-Quarter Tabs ───────────────────────────────────────────────────

@st.cache_data(show_spinner="Computing since-inception analytics …")
def _cached_cross_quarter_data(raw: dict) -> CrossQuarterData:
    data = _to_dashboard_data(raw)
    return compute_cross_quarter_data(data)


def _tab_xq_overview(data: DashboardData, xq: CrossQuarterData):
    """Tab 1: Summary metrics, equity curve, drawdown."""
    pm = xq.portfolio_metrics
    bm = xq.benchmark_metrics

    st.subheader("Summary Metrics")
    cols = st.columns(4)
    _metrics = [
        ("CAGR", f"{pm.get('cagr_pct', 0):+.2f}%"),
        ("Sharpe (ex Rf)", f"{pm.get('sharpe_ratio_ex_Rf', 0):.3f}"),
        ("Sortino", f"{pm.get('sortino_ratio', 0):.3f}"),
        ("Max Drawdown", f"{pm.get('max_drawdown_pct', 0):.2f}%"),
    ]
    for col, (label, val) in zip(cols, _metrics):
        col.metric(label, val)

    cols2 = st.columns(4)
    _metrics2 = [
        ("Total Return", f"{pm.get('total_return_pct', 0):+.2f}%"),
        ("Calmar", f"{pm.get('calmar_ratio', 0):.3f}"),
        ("VaR 95%", f"{pm.get('var_95_pct', 0):.2f}%"),
        ("Positive Days", f"{pm.get('positive_days_pct', 0):.1f}%"),
    ]
    for col, (label, val) in zip(cols2, _metrics2):
        col.metric(label, val)

    if bm is not None:
        st.markdown("**Benchmark Relative**")
        cols3 = st.columns(4)
        _bm_metrics = [
            ("Alpha", f"{bm.get('alpha_pct', 0):+.2f}%"),
            ("Beta", f"{bm.get('beta', 0):.3f}"),
            ("Info Ratio", f"{bm.get('information_ratio', 0):.3f}"),
            ("Up / Down Capture",
             f"{bm.get('up_capture_pct', 0):.0f}% / {bm.get('down_capture_pct', 0):.0f}%"),
        ]
        for col, (label, val) in zip(cols3, _bm_metrics):
            col.metric(label, val)

    st.markdown("---")
    st.plotly_chart(
        equity_curve_chart(data.daily_pf_values, data.comparison_df),
        width='stretch',
    )
    st.plotly_chart(
        drawdown_chart(xq.drawdown_series, xq.index_drawdown_series),
        width='stretch',
    )


def _tab_xq_performance(data: DashboardData, xq: CrossQuarterData):
    """Tab 2: QoQ bar chart, calendar year performance, monthly heatmap."""
    st.plotly_chart(
        qoq_performance_chart(data.quarterly_alpha),
        width='stretch',
    )
    st.plotly_chart(
        calendar_year_chart(xq.calendar_year_df),
        width='stretch',
    )
    st.plotly_chart(
        monthly_returns_heatmap(xq.monthly_returns),
        width='stretch',
    )


def _tab_xq_dynamics(xq: CrossQuarterData):
    """Tab 3: Cash trend, churn, return distribution."""
    st.plotly_chart(
        cash_trend_chart(xq.cash_by_quarter),
        width='stretch',
    )
    st.plotly_chart(
        churn_chart(xq.churn_df),
        width='stretch',
    )
    st.plotly_chart(
        return_distribution_charts(xq.monthly_returns),
        width='stretch',
    )


# ── Render (importable by hub.py) ─────────────────────────────────────────

def render_dashboard(run_dir: str) -> None:
    """Render the full dashboard for a single run directory.

    Callable from both ``main()`` (standalone mode) and ``hub.py``.
    The caller is responsible for calling ``st.set_page_config`` beforehand.
    """
    raw = _load_data(run_dir)
    data = _to_dashboard_data(raw)

    mode, quarter, sector_filter = _render_sidebar(data)

    if mode == "Since Inception":
        xq = _cached_cross_quarter_data(raw)

        tab_overview, tab_perf, tab_dynamics = st.tabs([
            "Overview",
            "Performance Analysis",
            "Portfolio Dynamics",
        ])

        with tab_overview:
            _tab_xq_overview(data, xq)
        with tab_perf:
            _tab_xq_performance(data, xq)
        with tab_dynamics:
            _tab_xq_dynamics(xq)

    else:
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
                quarterly_alpha=data.quarterly_alpha,
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


# ── Main (standalone entry point) ─────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="alphaBT Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    parser = argparse.ArgumentParser(description="alphaBT Dashboard")
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to a backtest run directory or get_portfolio --with-levels output folder",
    )
    args, _ = parser.parse_known_args()

    render_dashboard(args.run_dir)


if __name__ == "__main__":
    main()
