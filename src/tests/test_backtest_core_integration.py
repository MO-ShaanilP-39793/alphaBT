"""
Integration tests for backtest_core() — no mocking.

Run the full pipeline (selection → simulate_trades → equity curve) on small
synthetic data.  These tests verify end-to-end correctness, not just wiring.
"""

import pytest
import pandas as pd
import numpy as np
from tests.conftest import Q1, Q2, STOCK_NAMES

from backtest_strategy import backtest_core


# =============================================================================
# HELPERS
# =============================================================================

def _assert_valid_result(result):
    """Common assertions on a non-None backtest result."""
    assert result is not None, "backtest_core returned None"
    assert "trade_results" in result
    assert "daily_pf_values" in result

    tr = result["trade_results"]
    assert not tr.empty, "trade_results is empty"

    required_trade_cols = {
        "entry_date", "exit_date", "entry_price", "exit_price",
        "stock_return", "SL_triggered", "TP_triggered",
    }
    missing = required_trade_cols - set(tr.columns)
    assert not missing, f"Missing trade result columns: {missing}"

    eq = result["daily_pf_values"]
    assert not eq.empty, "daily_pf_values is empty"
    assert "Total_Portfolio_Value" in eq.columns


# =============================================================================
# FLAT MODE — category_based selection
# =============================================================================

class TestFlatModeEndToEnd:

    def test_flat_mode_category_based(self, base_config,
                                       sample_input_data_volatility,
                                       sample_price_data, sample_index_data):
        """Full pipeline: volatility category_based selection + flat TP/SL."""
        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
            verbose=True,
        )
        _assert_valid_result(result)

        tr = result["trade_results"]
        # With 2 stocks per category × 3 categories × 2 quarters = 12 trades
        assert len(tr) == 12, f"Expected 12 trades, got {len(tr)}"

        # All trades should have valid entry_price (no NaN)
        assert tr["entry_price"].notna().all(), "Some trades have NaN entry_price"

    def test_flat_mode_single_quarter(self, base_config,
                                       sample_input_data_volatility,
                                       sample_price_data, sample_index_data):
        """Single quarter run."""
        base_config["first_quarter"] = Q1
        base_config["last_quarter"] = Q1

        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        _assert_valid_result(result)

        tr = result["trade_results"]
        assert (tr["quarter"] == Q1).all()


# =============================================================================
# TP/SL DISABLED
# =============================================================================

class TestTpSlDisabled:

    def test_both_disabled_no_triggers(self, base_config,
                                        sample_input_data_volatility,
                                        sample_price_data, sample_index_data):
        """TP and SL both disabled → no early exits, all trades run to quarter end."""
        base_config["tp_enabled"] = False
        base_config["sl_enabled"] = False

        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        _assert_valid_result(result)

        tr = result["trade_results"]
        # No TP or SL should trigger
        assert not tr["TP_triggered"].any(), "TP triggered with tp_enabled=False"
        assert not tr["SL_triggered"].any(), "SL triggered with sl_enabled=False"


# =============================================================================
# TOP-K SELECTION
# =============================================================================

class TestTopKSelection:

    def test_top_k_flat_mode(self, base_config,
                              sample_input_data_volatility,
                              sample_price_data, sample_index_data):
        """top_k selection with equal weighting."""
        base_config["selection_type"] = "top_k"
        base_config["top_k_config"] = {"k": 5, "weighting_scheme": "equal"}

        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        _assert_valid_result(result)

        tr = result["trade_results"]
        assert "stock_weight" in tr.columns
        # 5 stocks × 2 quarters = 10 trades
        assert len(tr) == 10, f"Expected 10 trades, got {len(tr)}"
        # Equal weights
        assert tr["stock_weight"].nunique() == 1
        assert abs(tr["stock_weight"].iloc[0] - 0.2) < 1e-8


# =============================================================================
# PRESELECTED PORTFOLIO
# =============================================================================

class TestPreselectedPortfolio:

    def test_preselected_flat_mode(self, base_config,
                                    sample_preselected_data,
                                    sample_price_data, sample_index_data):
        """Preselected input with category."""
        base_config["run_stock_selection"] = False

        result = backtest_core(
            config=base_config,
            input_data=sample_preselected_data,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        _assert_valid_result(result)

        tr = result["trade_results"]
        assert "stock_weight" in tr.columns


# =============================================================================
# ATR AND PIVOT MODES
# =============================================================================

class TestDynamicModes:

    def test_atr_mode(self, base_config,
                       sample_input_data_volatility,
                       sample_price_data, sample_index_data):
        """ATR mode with config. May fall back to flat on short data — that's fine."""
        base_config["tp_mode"] = "atr"
        base_config["sl_mode"] = "atr"
        base_config["atr_config"] = {
            "period": 14,
            "tp_multiplier": 2.0,
            "sl_multiplier": 1.5,
        }

        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        _assert_valid_result(result)

    def test_pivot_mode(self, base_config,
                         sample_input_data_volatility,
                         sample_price_data, sample_index_data):
        """Pivot mode with config."""
        base_config["tp_mode"] = "pivot"
        base_config["sl_mode"] = "pivot"
        base_config["pivot_config"] = {
            "lookback_days": 30,
            "tp_level": "R1",
            "sl_level": "S1",
        }

        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        _assert_valid_result(result)


# =============================================================================
# CAPITAL CONTINUITY
# =============================================================================

class TestCapitalContinuity:

    def test_capital_continuity_across_quarters(self, base_config,
                                                  sample_input_data_volatility,
                                                  sample_price_data, sample_index_data):
        """
        The ending portfolio value of Q1 should equal starting capital of Q2.
        We verify via the daily_pf_values having a continuous equity curve.
        """
        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        _assert_valid_result(result)

        eq = result["daily_pf_values"]
        quarters = eq["quarter"].unique()
        if len(quarters) >= 2:
            q1_data = eq[eq["quarter"] == quarters[0]]
            q2_data = eq[eq["quarter"] == quarters[1]]
            q1_end = q1_data["Total_Portfolio_Value"].iloc[-1]
            q2_start = q2_data["Total_Portfolio_Value"].iloc[0]
            # They should be reasonably close (Q2 starts with Q1's ending capital,
            # but daily value on first day may differ due to new positions)
            # Allow 5% tolerance for position re-entry effects
            assert abs(q2_start - q1_end) / q1_end < 0.05, (
                f"Capital discontinuity: Q1 end={q1_end:.0f}, Q2 start={q2_start:.0f}"
            )


# =============================================================================
# DETERMINISM
# =============================================================================

class TestDeterminism:

    def test_deterministic_results(self, base_config,
                                    sample_input_data_volatility,
                                    sample_price_data, sample_index_data):
        """Two runs with same inputs → identical results."""
        result1 = backtest_core(
            config=base_config.copy(),
            input_data=sample_input_data_volatility.copy(),
            price_data=sample_price_data.copy(),
            index_data=sample_index_data.copy(),
        )
        result2 = backtest_core(
            config=base_config.copy(),
            input_data=sample_input_data_volatility.copy(),
            price_data=sample_price_data.copy(),
            index_data=sample_index_data.copy(),
        )
        _assert_valid_result(result1)
        _assert_valid_result(result2)

        pd.testing.assert_frame_equal(
            result1["trade_results"].reset_index(drop=True),
            result2["trade_results"].reset_index(drop=True),
        )

    def test_no_nan_entry_prices(self, base_config,
                                  sample_input_data_volatility,
                                  sample_price_data, sample_index_data):
        """Synthetic data is sufficient — no NaN entry prices."""
        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        _assert_valid_result(result)
        assert result["trade_results"]["entry_price"].notna().all()
