"""Core backtesting functionality."""

from backtest.tpsl import simulate_trades
from backtest.simulation import (
    compute_portfolio_value_over_quarters,
    compute_portfolio_vs_index,
)
from config.regime_dates import CRISIS_REGIMES, MARKET_REGIMES
