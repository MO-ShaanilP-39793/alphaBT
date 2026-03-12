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
