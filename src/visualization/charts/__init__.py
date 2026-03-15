"""Plotly chart builders for the alphaBT dashboard."""

from .portfolio import (
    portfolio_vs_index_chart,
    cash_area_chart,
    invested_vs_cash_chart,
)
from .trade_outcomes import (
    exit_type_donut,
    exit_by_category_bar,
    return_vs_holding_scatter,
)
from .stock_drilldown import stock_candlestick_chart
from .aggregate import (
    holdings_gantt_chart,
    pnl_waterfall_chart,
    return_distribution_histogram,
)
from .sector_analysis import (
    sector_allocation_donut,
    sector_performance_bar,
    sector_win_rate_bar,
    sector_exit_breakdown,
)
from .cross_quarter import (
    equity_curve_chart,
    drawdown_chart,
    qoq_performance_chart,
    calendar_year_chart,
    monthly_returns_heatmap,
    cash_trend_chart,
    churn_chart,
    return_distribution_charts,
)
