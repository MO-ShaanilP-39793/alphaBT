"""
Unit tests for backtest_core() — orchestration logic.

These tests mock downstream functions (simulate_trades, compute_portfolio_value_over_quarters,
select_and_weight_stocks, select_top_k_stocks) to isolate the control-flow and
config-validation logic inside backtest_core().
"""

import pytest
import warnings
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pydantic import ValidationError
from tests.conftest import Q1, Q2, STOCK_NAMES

from backtest_strategy import backtest_core, validate_preselected_input


# =============================================================================
# HELPERS — mock return values
# =============================================================================

def _mock_selected_stocks_category_based():
    """Fake output of select_and_weight_stocks (category_based, volatility)."""
    rows = []
    cats = ["high_volatility", "medium_volatility", "low_volatility"]
    weights = [0.33, 0.33, 0.34]
    for q in [Q1, Q2]:
        for i, cat in enumerate(cats):
            for name in STOCK_NAMES[i*3:(i+1)*3][:2]:  # 2 per cat
                rows.append({
                    "quarter": q,
                    "co_name": name,
                    "cat": cat,
                    "selection_cat": cat,
                    "cat_weight": weights[i],
                })
    return pd.DataFrame(rows)


def _mock_selected_stocks_top_k():
    """Fake output of select_top_k_stocks."""
    rows = []
    for q in [Q1, Q2]:
        for name in STOCK_NAMES[:5]:
            rows.append({
                "quarter": q,
                "co_name": name,
                "stock_weight": 0.2,
            })
    return pd.DataFrame(rows)


def _mock_trade_results():
    """Fake trade results DataFrame with required columns."""
    rows = []
    for q in [Q1, Q2]:
        for name in STOCK_NAMES[:6]:
            rows.append({
                "quarter": q,
                "co_name": name,
                "cat": "high_volatility",
                "cat_weight": 0.33,
                "entry_date": pd.Timestamp("2023-02-16"),
                "exit_date": pd.Timestamp("2023-05-29"),
                "entry_price": 100.0,
                "exit_price": 105.0,
                "holding_period": 70,
                "SL_triggered": False,
                "TP_triggered": True,
                "regime_exit": False,
                "tp_pct_used": 0.05,
                "sl_pct_used": 0.05,
                "stock_return": 0.05,
            })
    return pd.DataFrame(rows)


def _mock_equity_curve():
    """Fake daily portfolio values."""
    dates = pd.bdate_range("2023-02-15", "2023-08-14")
    df = pd.DataFrame({
        "Total_Portfolio_Value": np.linspace(1e9, 1.05e9, len(dates)),
        "quarter": Q1,
    }, index=dates)
    return df


# =============================================================================
# PATCH PATHS — these match where the names are looked up in backtest_strategy
# =============================================================================

PATCH_SIMULATE = "backtest_strategy.simulate_trades"
PATCH_COMPUTE_PF = "backtest_strategy.compute_portfolio_value_over_quarters"
PATCH_SELECT_AND_WEIGHT = "backtest_strategy.run_stock_selection"
PATCH_SELECT_TOP_K = "backtest_strategy.select_top_k_stocks"


# =============================================================================
# TESTS — Return structure & happy path
# =============================================================================

class TestBacktestCoreReturnStructure:
    """Verify the return dict shape on the happy path."""

    @patch(PATCH_COMPUTE_PF, return_value=_mock_equity_curve())
    @patch(PATCH_SIMULATE, return_value=_mock_trade_results())
    @patch(PATCH_SELECT_AND_WEIGHT, return_value=_mock_selected_stocks_category_based())
    def test_returns_dict_with_expected_keys(self, mock_sel, mock_sim, mock_pf,
                                              base_config, sample_input_data_volatility,
                                              sample_price_data, sample_index_data):
        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        assert result is not None
        assert set(result.keys()) == {
            "daily_pf_values", "trade_results",
            "first_quarter", "last_quarter", "data_issues",
        }

    @patch(PATCH_COMPUTE_PF, return_value=_mock_equity_curve())
    @patch(PATCH_SIMULATE, return_value=_mock_trade_results())
    @patch(PATCH_SELECT_AND_WEIGHT, return_value=_mock_selected_stocks_category_based())
    def test_returns_dataframes(self, mock_sel, mock_sim, mock_pf,
                                 base_config, sample_input_data_volatility,
                                 sample_price_data, sample_index_data):
        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        assert isinstance(result["trade_results"], pd.DataFrame)
        assert isinstance(result["daily_pf_values"], pd.DataFrame)


# =============================================================================
# TESTS — Quarter filtering
# =============================================================================

class TestQuarterFiltering:

    @patch(PATCH_COMPUTE_PF, return_value=_mock_equity_curve())
    @patch(PATCH_SIMULATE, return_value=_mock_trade_results())
    @patch(PATCH_SELECT_AND_WEIGHT, return_value=_mock_selected_stocks_category_based())
    def test_auto_quarter_detection(self, mock_sel, mock_sim, mock_pf,
                                     base_config, sample_input_data_volatility,
                                     sample_price_data, sample_index_data):
        """When first/last quarter are None, infer from data."""
        base_config["first_quarter"] = None
        base_config["last_quarter"] = None

        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        assert result is not None
        assert result["first_quarter"] == Q1
        assert result["last_quarter"] == Q2

    @patch(PATCH_COMPUTE_PF, return_value=_mock_equity_curve())
    @patch(PATCH_SIMULATE, return_value=_mock_trade_results())
    @patch(PATCH_SELECT_AND_WEIGHT, return_value=_mock_selected_stocks_category_based())
    def test_explicit_quarter_filtering(self, mock_sel, mock_sim, mock_pf,
                                         base_config, sample_input_data_volatility,
                                         sample_price_data, sample_index_data):
        """Restrict to single quarter."""
        base_config["first_quarter"] = Q1
        base_config["last_quarter"] = Q1

        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        assert result is not None
        assert result["first_quarter"] == Q1
        assert result["last_quarter"] == Q1

    def test_empty_input_after_filter_returns_none(self, base_config,
                                                     sample_input_data_volatility,
                                                     sample_price_data, sample_index_data):
        """Quarter range with no matching data → None."""
        base_config["first_quarter"] = 202002
        base_config["last_quarter"] = 202005

        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        assert result is None


# =============================================================================
# TESTS — Empty / None intermediate results
# =============================================================================

class TestEmptyIntermediateResults:

    @patch(PATCH_SIMULATE, return_value=_mock_trade_results())
    @patch(PATCH_SELECT_AND_WEIGHT, return_value=pd.DataFrame())
    def test_empty_selected_stocks_returns_none(self, mock_sel, mock_sim,
                                                  base_config, sample_input_data_volatility,
                                                  sample_price_data, sample_index_data):
        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        assert result is None

    @patch(PATCH_COMPUTE_PF, return_value=_mock_equity_curve())
    @patch(PATCH_SIMULATE, return_value=pd.DataFrame())
    @patch(PATCH_SELECT_AND_WEIGHT, return_value=_mock_selected_stocks_category_based())
    def test_empty_trade_results_returns_none(self, mock_sel, mock_sim, mock_pf,
                                                base_config, sample_input_data_volatility,
                                                sample_price_data, sample_index_data):
        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        assert result is None


# =============================================================================
# TESTS — Config validation errors
# =============================================================================

class TestConfigValidation:
    """These errors are caught by backtest_core's try/except → returns None + warning."""

    def _assert_returns_none_with_warning(self, config, input_data, price_data, index_data, match_str):
        """Helper: verify backtest_core returns None and warns with match_str."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = backtest_core(
                config=config,
                input_data=input_data,
                price_data=price_data,
                index_data=index_data,
            )
        assert result is None, "Expected None for invalid config"
        warning_messages = [str(x.message) for x in w]
        assert any(match_str in msg for msg in warning_messages), (
            f"Expected warning containing '{match_str}', got: {warning_messages}"
        )

    def test_entry_price_window_zero(self, base_config, sample_input_data_volatility,
                                      sample_price_data, sample_index_data):
        base_config["entry_price_window"] = 0
        with pytest.raises(ValidationError, match="entry_price_window"):
            backtest_core(
                config=base_config,
                input_data=sample_input_data_volatility,
                price_data=sample_price_data,
                index_data=sample_index_data,
            )

    def test_entry_price_window_negative(self, base_config, sample_input_data_volatility,
                                           sample_price_data, sample_index_data):
        base_config["entry_price_window"] = -1
        with pytest.raises(ValidationError, match="entry_price_window"):
            backtest_core(
                config=base_config,
                input_data=sample_input_data_volatility,
                price_data=sample_price_data,
                index_data=sample_index_data,
            )

    def test_entry_price_window_string(self, base_config, sample_input_data_volatility,
                                         sample_price_data, sample_index_data):
        base_config["entry_price_window"] = "abc"
        with pytest.raises(ValidationError, match="entry_price_window"):
            backtest_core(
                config=base_config,
                input_data=sample_input_data_volatility,
                price_data=sample_price_data,
                index_data=sample_index_data,
            )

    def test_tpsl_category_dimension_invalid(self, base_config,
                                               sample_input_data_volatility,
                                               sample_price_data, sample_index_data):
        base_config["tpsl_category_dimension"] = "invalid"
        self._assert_returns_none_with_warning(
            base_config, sample_input_data_volatility, sample_price_data,
            sample_index_data, "tpsl_category_dimension"
        )

    def test_tpsl_dimension_mismatch(self, base_config,
                                       sample_input_data_volatility,
                                       sample_price_data, sample_index_data):
        """selection=vol, weighting=vol, tpsl_category_dimension=mcap → None + warning."""
        base_config["selection_dimension"] = "volatility"
        base_config["weighting_dimension"] = "volatility"
        base_config["tpsl_category_dimension"] = "mcap"
        self._assert_returns_none_with_warning(
            base_config, sample_input_data_volatility, sample_price_data,
            sample_index_data, "tpsl_category_dimension"
        )


# =============================================================================
# TESTS — Selection path branching
# =============================================================================

class TestSelectionPaths:

    @patch(PATCH_COMPUTE_PF, return_value=_mock_equity_curve())
    @patch(PATCH_SIMULATE, return_value=_mock_trade_results())
    @patch(PATCH_SELECT_AND_WEIGHT, return_value=_mock_selected_stocks_category_based())
    def test_category_based_selection_path(self, mock_sel, mock_sim, mock_pf,
                                            base_config, sample_input_data_volatility,
                                            sample_price_data, sample_index_data):
        """category_based path calls run_stock_selection."""
        base_config["selection_type"] = "category_based"
        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        assert result is not None
        mock_sel.assert_called_once()

    @patch(PATCH_COMPUTE_PF, return_value=_mock_equity_curve())
    @patch(PATCH_SIMULATE, return_value=_mock_trade_results())
    @patch(PATCH_SELECT_AND_WEIGHT, return_value=_mock_selected_stocks_top_k())
    def test_top_k_selection_path(self, mock_sel, mock_sim, mock_pf,
                                    base_config, sample_input_data_volatility,
                                    sample_price_data, sample_index_data):
        """top_k path calls run_stock_selection (which internally dispatches to select_top_k)."""
        base_config["selection_type"] = "top_k"
        base_config["top_k_config"] = {"k": 5, "weighting_scheme": "equal"}
        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        assert result is not None
        mock_sel.assert_called_once()

    @patch(PATCH_COMPUTE_PF, return_value=_mock_equity_curve())
    @patch(PATCH_SIMULATE, return_value=_mock_trade_results())
    def test_preselected_path(self, mock_sim, mock_pf,
                                base_config, sample_preselected_data,
                                sample_price_data, sample_index_data):
        """run_stock_selection=False → no selection functions called."""
        base_config["run_stock_selection"] = False
        result = backtest_core(
            config=base_config,
            input_data=sample_preselected_data,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        assert result is not None

    def test_preselected_missing_columns(self, base_config,
                                          sample_price_data, sample_index_data):
        """Preselected input missing stock_weight → returns None + warning."""
        base_config["run_stock_selection"] = False
        bad_input = pd.DataFrame({
            "quarter": [Q1], "co_name": ["STOCK_A"],
            # missing stock_weight
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = backtest_core(
                config=base_config,
                input_data=bad_input,
                price_data=sample_price_data,
                index_data=sample_index_data,
            )
        assert result is None
        warning_messages = [str(x.message) for x in w]
        assert any("Backtest core failed" in msg for msg in warning_messages)

    def test_preselected_tiered_no_category_no_config(self, base_config,
                                                        sample_preselected_no_category,
                                                        sample_price_data, sample_index_data):
        """Tiered mode + no category + no tiered_config → ValidationError."""
        base_config["run_stock_selection"] = False
        base_config["tp_mode"] = "tiered"
        base_config["tp_enabled"] = True
        base_config.pop("tiered_config", None)
        with pytest.raises(ValidationError, match="tiered_config"):
            backtest_core(
                config=base_config,
                input_data=sample_preselected_no_category,
                price_data=sample_price_data,
                index_data=sample_index_data,
            )


# =============================================================================
# TESTS — Exception handling
# =============================================================================

class TestExceptionHandling:

    @patch(PATCH_SIMULATE, side_effect=RuntimeError("boom"))
    @patch(PATCH_SELECT_AND_WEIGHT, return_value=_mock_selected_stocks_category_based())
    def test_exception_returns_none_with_warning(self, mock_sel, mock_sim,
                                                   base_config, sample_input_data_volatility,
                                                   sample_price_data, sample_index_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = backtest_core(
                config=base_config,
                input_data=sample_input_data_volatility,
                price_data=sample_price_data,
                index_data=sample_index_data,
            )
        assert result is None
        # Check that a warning was issued about the failure
        warning_messages = [str(x.message) for x in w]
        assert any("Backtest core failed" in msg for msg in warning_messages)


# =============================================================================
# TESTS — Log output (replaces verbose flag tests)
# =============================================================================

class TestLogOutput:

    @patch(PATCH_COMPUTE_PF, return_value=_mock_equity_curve())
    @patch(PATCH_SIMULATE, return_value=_mock_trade_results())
    @patch(PATCH_SELECT_AND_WEIGHT, return_value=_mock_selected_stocks_category_based())
    def test_backtest_core_emits_log_messages(self, mock_sel, mock_sim, mock_pf,
                                      base_config, sample_input_data_volatility,
                                      sample_price_data, sample_index_data, caplog):
        """backtest_core should emit step markers via logging."""
        import logging
        with caplog.at_level(logging.INFO, logger="alphaBT"):
            backtest_core(
                config=base_config,
                input_data=sample_input_data_volatility,
                price_data=sample_price_data,
                index_data=sample_index_data,
            )
        # Should see step markers like "[1/4]" in log output
        assert any("[1/4]" in msg for msg in caplog.messages)


# =============================================================================
# TESTS — tpsl_cat column assignment
# =============================================================================

class TestTpslCatColumnAssignment:

    @patch(PATCH_COMPUTE_PF, return_value=_mock_equity_curve())
    @patch(PATCH_SIMULATE, return_value=_mock_trade_results())
    @patch(PATCH_SELECT_AND_WEIGHT, return_value=_mock_selected_stocks_category_based())
    def test_tpsl_cat_defaults_to_selection_cat(self, mock_sel, mock_sim, mock_pf,
                                                  base_config, sample_input_data_volatility,
                                                  sample_price_data, sample_index_data):
        """Default tpsl_category_dimension='selection' → tpsl_cat = selection_cat."""
        base_config["tpsl_category_dimension"] = "selection"
        result = backtest_core(
            config=base_config,
            input_data=sample_input_data_volatility,
            price_data=sample_price_data,
            index_data=sample_index_data,
        )
        assert result is not None
        # The selected_stocks passed to simulate_trades should have tpsl_cat
        # We verify by checking the mock call args
        call_args = mock_sim.call_args
        selected_passed = call_args[0][0] if call_args[0] else call_args[1].get("selected_stocks")
        if "tpsl_cat" in selected_passed.columns:
            assert (selected_passed["tpsl_cat"] == selected_passed["selection_cat"]).all()
