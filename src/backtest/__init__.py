"""Core backtesting functionality."""

from backtest.tpsl import simulate_trades
from backtest.simulation import (
    compute_pf_value_over_quarters,
    compute_pf_vs_index,
)
from backtest.regime_dates import CRISIS_REGIMES, MARKET_REGIMES
