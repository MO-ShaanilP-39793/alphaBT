"""Core backtesting functionality."""

from backtest.tpsl import simulate_trades
from backtest.simulation import (
    compute_pf_value_over_quarters,
    plot_pf_vs_index,
)
from backtest.analytics import (
    compute_portfolio_metrics,
    compute_monthly_returns,
    compute_benchmark_metrics,
    get_comprehensive_quarter_analysis,
    plot_drawdown,
    plot_monthly_returns_heatmap,
    plot_return_distribution,
    plot_category_performance_summary,
)
from backtest.regime_dates import CRISIS_REGIMES, MARKET_REGIMES
