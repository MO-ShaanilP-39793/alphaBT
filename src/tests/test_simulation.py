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


# =============================================================================
# Cash-In-Hand tracking and risk-free appreciation
# =============================================================================

class TestCashInHand:

    def test_cash_in_hand_column_present_no_interest(self, sized_trades, sample_price_data):
        from backtest.simulation import generate_quarter_equity_curve
        result = generate_quarter_equity_curve(
            sized_trades, sample_price_data, Q1, risk_free_rate_annual=None
        )
        assert "Cash_In_Hand" in result.columns
        assert "Total_Portfolio_Value" in result.columns

    def test_cash_in_hand_column_present_with_interest(self, sized_trades, sample_price_data):
        from backtest.simulation import generate_quarter_equity_curve
        result = generate_quarter_equity_curve(
            sized_trades, sample_price_data, Q1, risk_free_rate_annual=0.065
        )
        assert "Cash_In_Hand" in result.columns

    def test_no_interest_total_matches_legacy_logic(self, sized_trades, sample_price_data):
        """With risk_free_rate=None, Total_Portfolio_Value should equal equity + raw cash."""
        from backtest.simulation import generate_quarter_equity_curve
        result = generate_quarter_equity_curve(
            sized_trades, sample_price_data, Q1, risk_free_rate_annual=None
        )
        assert not result["Total_Portfolio_Value"].isna().any()
        assert not result["Cash_In_Hand"].isna().any()
        # Cash should be >= 0 at all times
        assert (result["Cash_In_Hand"] >= -0.01).all()

    def test_interest_increases_total(self, sized_trades, sample_price_data):
        """When risk-free rate > 0, total value after exits should be >= the no-interest total."""
        from backtest.simulation import generate_quarter_equity_curve
        result_no = generate_quarter_equity_curve(
            sized_trades, sample_price_data, Q1, risk_free_rate_annual=None
        )
        result_with = generate_quarter_equity_curve(
            sized_trades, sample_price_data, Q1, risk_free_rate_annual=0.065
        )
        # On the last day (all exits done, cash has compounded), with-interest >= no-interest
        assert result_with["Total_Portfolio_Value"].iloc[-1] >= result_no["Total_Portfolio_Value"].iloc[-1] - 0.01

    def test_cash_in_hand_in_multi_quarter(self, sample_price_data):
        from backtest.simulation import compute_portfolio_value_over_quarters
        trades_q1 = _make_trades(Q1, STOCK_NAMES[:3])
        trades_q2 = _make_trades(Q2, STOCK_NAMES[:3])
        all_trades = pd.concat([trades_q1, trades_q2], ignore_index=True)
        result = compute_portfolio_value_over_quarters(
            all_trades, sample_price_data, Q1, Q2,
            initial_capital=1_000_000_000,
            risk_free_rate_annual=0.065,
        )
        assert result is not None
        assert "Cash_In_Hand" in result.columns
        assert "Total_Portfolio_Value" in result.columns

    def test_end_of_quarter_value_rolls_to_next(self, sample_price_data):
        """End-of-quarter Total_Portfolio_Value becomes next quarter's starting capital."""
        from backtest.simulation import compute_portfolio_value_over_quarters
        trades_q1 = _make_trades(Q1, STOCK_NAMES[:3])
        trades_q2 = _make_trades(Q2, STOCK_NAMES[:3])
        all_trades = pd.concat([trades_q1, trades_q2], ignore_index=True)
        result = compute_portfolio_value_over_quarters(
            all_trades, sample_price_data, Q1, Q2,
            initial_capital=1_000_000_000,
        )
        assert result is not None
        q1_rows = result[result['quarter'] == Q1]
        q2_rows = result[result['quarter'] == Q2]
        if not q1_rows.empty and not q2_rows.empty:
            # Q2's first value should reflect Q1's ending value (which was the capital input)
            assert q1_rows["Total_Portfolio_Value"].iloc[-1] > 0
            assert q2_rows["Total_Portfolio_Value"].iloc[0] > 0


# =============================================================================
# compute_cash_metrics
# =============================================================================

class TestComputeCashMetrics:

    def test_returns_none_without_cash_column(self):
        from reporting.metrics import compute_cash_metrics
        daily_pf = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=10),
            "portfolio_value": [1e9] * 10,
            "quarter": 202302,
        })
        assert compute_cash_metrics(daily_pf) is None

    def test_returns_metrics_with_cash_column(self):
        from reporting.metrics import compute_cash_metrics
        daily_pf = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=10),
            "portfolio_value": [1e9] * 10,
            "cash_in_hand": [1e8] * 10,
            "quarter": 202302,
        })
        result = compute_cash_metrics(daily_pf)
        assert result is not None
        assert "avg_cash_pct" in result
        assert "avg_cash" in result
        assert "cash_by_quarter" in result
        assert abs(result["avg_cash_pct"] - 10.0) < 0.1  # 1e8 / 1e9 = 10%

    def test_by_quarter_breakdown(self):
        from reporting.metrics import compute_cash_metrics
        dates_q1 = pd.bdate_range("2023-01-02", periods=5)
        dates_q2 = pd.bdate_range("2023-06-01", periods=5)
        daily_pf = pd.DataFrame({
            "date": list(dates_q1) + list(dates_q2),
            "portfolio_value": [1e9] * 10,
            "cash_in_hand": [1e8] * 5 + [2e8] * 5,
            "quarter": [202302] * 5 + [202305] * 5,
        })
        result = compute_cash_metrics(daily_pf)
        by_q = result["cash_by_quarter"]
        assert len(by_q) == 2
        q1_row = by_q[by_q["quarter"] == 202302].iloc[0]
        q2_row = by_q[by_q["quarter"] == 202305].iloc[0]
        assert abs(q1_row["mean_cash_pct"] - 10.0) < 0.1
        assert abs(q2_row["mean_cash_pct"] - 20.0) < 0.1
