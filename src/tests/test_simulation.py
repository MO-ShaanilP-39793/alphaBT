"""
Phase 4: Tests for backtest/simulation.py.

Covers calculate_position_sizes, generate_quarter_equity_curve,
_generate_quarter_sequence, compute_portfolio_value_over_quarter.
"""

import pytest
import pandas as pd
import numpy as np

from tests.conftest import Q1, Q2, STOCK_NAMES, MCAP_CATEGORIES


# =============================================================================
# Helpers
# =============================================================================

def _make_trades(quarter, stocks, weight_mode="stock_weight"):
    """Create a minimal trades DataFrame for simulation."""
    exit_dates = pd.bdate_range(start="2023-04-01", periods=len(stocks))
    rows = []
    for i, name in enumerate(stocks):
        entry_price = 100.0 + i * 10
        exit_price = entry_price * 1.02
        row = {
            "quarter": quarter,
            "co_name": name,
            "cat": MCAP_CATEGORIES.get(name, "largecap"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_date": exit_dates[i],
            "stock_weight": 1.0 / len(stocks),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# calculate_position_sizes
# =============================================================================

class TestCalculatePositionSizes:

    def test_stock_weight_mode(self):
        from backtest.simulation import calculate_position_sizes
        trades = _make_trades(Q1, STOCK_NAMES[:3], weight_mode="stock_weight")
        result = calculate_position_sizes(trades, total_capital=1_000_000_000)
        assert "allocated_capital" in result.columns
        assert "shares" in result.columns
        # Each stock gets 1/3 of capital
        for _, row in result.iterrows():
            assert abs(row["allocated_capital"] - 1_000_000_000 / 3) < 1.0

    def test_stock_weight_sums_to_one(self):
        from backtest.simulation import calculate_position_sizes
        # Use stocks from all 3 categories so weights sum properly
        stocks = ["STOCK_A", "STOCK_B", "STOCK_D", "STOCK_E", "STOCK_G", "STOCK_H"]
        trades = _make_trades(Q1, stocks, weight_mode="stock_weight")
        result = calculate_position_sizes(trades, total_capital=1_000_000_000)
        assert "allocated_capital" in result.columns
        # Total allocated should match sum of stock_weights * capital
        # Each stock gets 1/6 of capital, so total = 1.0
        total_allocated = result["allocated_capital"].sum()
        assert abs(total_allocated - 1_000_000_000) < 100

    def test_shares_positive(self):
        from backtest.simulation import calculate_position_sizes
        trades = _make_trades(Q1, STOCK_NAMES[:3], weight_mode="stock_weight")
        result = calculate_position_sizes(trades, total_capital=1_000_000_000)
        assert all(result["shares"] > 0)


# =============================================================================
# _generate_quarter_sequence
# =============================================================================

class TestGenerateQuarterSequence:

    def test_single_quarter(self):
        from backtest.simulation import _generate_quarter_sequence
        result = _generate_quarter_sequence(202302, 202302)
        assert result == [202302]

    def test_two_quarter_span(self):
        from backtest.simulation import _generate_quarter_sequence
        result = _generate_quarter_sequence(202302, 202305)
        assert result == [202302, 202305]

    def test_year_wrap(self):
        from backtest.simulation import _generate_quarter_sequence
        result = _generate_quarter_sequence(202311, 202402)
        assert result == [202311, 202402]

    def test_full_year(self):
        from backtest.simulation import _generate_quarter_sequence
        result = _generate_quarter_sequence(202302, 202311)
        assert result == [202302, 202305, 202308, 202311]

    def test_reversed_returns_empty(self):
        from backtest.simulation import _generate_quarter_sequence
        result = _generate_quarter_sequence(202305, 202302)
        assert result == []


# =============================================================================
# generate_quarter_equity_curve
# =============================================================================

class TestGenerateQuarterEquityCurve:

    def test_equity_curve_has_total_column(self, sized_trades, sample_price_data):
        from backtest.simulation import generate_quarter_equity_curve
        result = generate_quarter_equity_curve(sized_trades, sample_price_data, Q1)
        assert "Total_Portfolio_Value" in result.columns
        assert len(result) > 0

    def test_equity_curve_no_nan(self, sized_trades, sample_price_data):
        from backtest.simulation import generate_quarter_equity_curve
        result = generate_quarter_equity_curve(sized_trades, sample_price_data, Q1)
        assert not result["Total_Portfolio_Value"].isna().any()

    def test_equity_curve_first_value_reasonable(self, sized_trades, sample_price_data):
        from backtest.simulation import generate_quarter_equity_curve
        result = generate_quarter_equity_curve(sized_trades, sample_price_data, Q1)
        initial = result["Total_Portfolio_Value"].iloc[0]
        # Should be close to total allocated capital
        total_allocated = sized_trades["allocated_capital"].sum()
        assert abs(initial - total_allocated) / total_allocated < 0.05


# =============================================================================
# compute_portfolio_value_over_quarter
# =============================================================================

class TestComputePfValueOverQuarter:

    def test_single_quarter_computation(self, sample_price_data):
        from backtest.simulation import compute_portfolio_value_over_quarter
        trades = _make_trades(Q1, STOCK_NAMES[:3], weight_mode="stock_weight")
        result = compute_portfolio_value_over_quarter(
            trades, sample_price_data, Q1, initial_capital=1_000_000_000
        )
        assert result is not None
        assert "Total_Portfolio_Value" in result.columns
        assert len(result) > 0

    def test_empty_trades_returns_none(self, sample_price_data):
        from backtest.simulation import compute_portfolio_value_over_quarter
        trades = pd.DataFrame(columns=["quarter", "co_name", "cat", "stock_weight",
                                        "entry_price", "exit_price", "exit_date"])
        result = compute_portfolio_value_over_quarter(trades, sample_price_data, Q1)
        assert result is None
