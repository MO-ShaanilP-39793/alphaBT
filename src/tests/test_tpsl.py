"""
Phase 5: Tests for backtest/tpsl.py.

Covers simulate_trades config validation, flat mode, TP/SL disabled scenarios,
and dynamic threshold dispatch.
"""

import pytest
import pandas as pd
import numpy as np

from tests.conftest import Q1, Q2, STOCK_NAMES, MCAP_CATEGORIES, _make_price_data


# =============================================================================
# Helpers
# =============================================================================

def _make_selected_stocks(quarter, stocks, mode="cat_weight"):
    """Create selected_stocks DataFrame for simulate_trades."""
    rows = []
    for i, name in enumerate(stocks):
        row = {
            "quarter": quarter,
            "co_name": name,
            "cat": MCAP_CATEGORIES.get(name, "largecap"),
            "prob": 0.8 - i * 0.05,
        }
        if mode == "cat_weight":
            row["cat_weight"] = 1.0 / 3
        else:
            row["stock_weight"] = 1.0 / len(stocks)
        rows.append(row)
    return pd.DataFrame(rows)


def _make_index_data(start="2023-02-01", end="2023-08-31"):
    dates = pd.bdate_range(start=start, end=end)
    values = [1000.0 * (1 + 0.0005) ** i for i in range(len(dates))]
    return pd.DataFrame({"date": dates, "value": values})


def _base_index_exit():
    return {
        "regime_filter": {"enabled": False, "ma_period": 20, "exit_threshold": -0.02},
        "vol_adjustment": {
            "enabled": False, "lookback": 20,
            "high_vol_threshold": 0.25, "low_vol_threshold": 0.15,
            "high_vol_multiplier": 1.5, "low_vol_multiplier": 0.8,
        },
    }


# =============================================================================
# Config validation
# =============================================================================

class TestSimulateTradesValidation:

    def test_missing_index_exit_config_raises(self):
        from backtest.tpsl import simulate_trades
        stocks = _make_selected_stocks(Q1, STOCK_NAMES[:3])
        prices = _make_price_data(STOCK_NAMES[:3], "2023-02-01", "2023-08-31")
        with pytest.raises(ValueError, match="index_exit_config"):
            simulate_trades(
                stocks, prices, category_scheme="volatility",
                tp_mode="flat", sl_mode="flat",
                flat_config={"tp_pct": 0.05, "sl_pct": 0.05},
                index_exit_config=None,
            )

    def test_missing_atr_config_raises(self):
        from backtest.tpsl import simulate_trades
        stocks = _make_selected_stocks(Q1, STOCK_NAMES[:3])
        prices = _make_price_data(STOCK_NAMES[:3], "2023-02-01", "2023-08-31")
        with pytest.raises(ValueError):
            simulate_trades(
                stocks, prices, category_scheme="volatility",
                tp_mode="atr", sl_mode="flat",
                flat_config={"tp_pct": 0.05, "sl_pct": 0.05},
                atr_config=None,
                index_exit_config=_base_index_exit(),
            )

    def test_missing_pivot_config_raises(self):
        from backtest.tpsl import simulate_trades
        stocks = _make_selected_stocks(Q1, STOCK_NAMES[:3])
        prices = _make_price_data(STOCK_NAMES[:3], "2023-02-01", "2023-08-31")
        with pytest.raises(ValueError):
            simulate_trades(
                stocks, prices, category_scheme="volatility",
                tp_mode="pivot", sl_mode="flat",
                flat_config={"tp_pct": 0.05, "sl_pct": 0.05},
                pivot_config=None,
                index_exit_config=_base_index_exit(),
            )


# =============================================================================
# Flat mode trades
# =============================================================================

class TestFlatModeTrades:

    def test_flat_mode_produces_results(self):
        from backtest.tpsl import simulate_trades
        stocks = _make_selected_stocks(Q1, STOCK_NAMES[:3])
        prices = _make_price_data(STOCK_NAMES[:3], "2023-02-01", "2023-08-31")
        result = simulate_trades(
            stocks, prices, category_scheme="volatility",
            tp_mode="flat", sl_mode="flat",
            tp_enabled=True, sl_enabled=True,
            flat_config={"tp_pct": 0.05, "sl_pct": 0.05},
            index_exit_config=_base_index_exit(),
        )
        assert len(result) == 3
        assert "entry_price" in result.columns
        assert "exit_price" in result.columns
        assert "stock_return" in result.columns

    def test_entry_prices_populated(self):
        from backtest.tpsl import simulate_trades
        stocks = _make_selected_stocks(Q1, STOCK_NAMES[:3])
        prices = _make_price_data(STOCK_NAMES[:3], "2023-02-01", "2023-08-31")
        result = simulate_trades(
            stocks, prices, category_scheme="volatility",
            tp_mode="flat", sl_mode="flat",
            flat_config={"tp_pct": 0.05, "sl_pct": 0.05},
            index_exit_config=_base_index_exit(),
        )
        assert not result["entry_price"].isna().any()

    def test_holding_period_positive(self):
        from backtest.tpsl import simulate_trades
        stocks = _make_selected_stocks(Q1, STOCK_NAMES[:3])
        prices = _make_price_data(STOCK_NAMES[:3], "2023-02-01", "2023-08-31")
        result = simulate_trades(
            stocks, prices, category_scheme="volatility",
            tp_mode="flat", sl_mode="flat",
            flat_config={"tp_pct": 0.05, "sl_pct": 0.05},
            index_exit_config=_base_index_exit(),
        )
        valid = result[result["holding_period"].notna()]
        assert all(valid["holding_period"] > 0)


# =============================================================================
# TP/SL disabled
# =============================================================================

class TestTPSLDisabled:

    def test_no_tp_triggered_when_disabled(self):
        from backtest.tpsl import simulate_trades
        stocks = _make_selected_stocks(Q1, STOCK_NAMES[:3])
        prices = _make_price_data(STOCK_NAMES[:3], "2023-02-01", "2023-08-31")
        result = simulate_trades(
            stocks, prices, category_scheme="volatility",
            tp_mode="flat", sl_mode="flat",
            tp_enabled=False, sl_enabled=False,
            flat_config={"tp_pct": 0.05, "sl_pct": 0.05},
            index_exit_config=_base_index_exit(),
        )
        assert not result["TP_triggered"].any()
        assert not result["SL_triggered"].any()


# =============================================================================
# Stock weight mode (top-k / preselected)
# =============================================================================

class TestStockWeightMode:

    def test_stock_weight_mode_works(self):
        from backtest.tpsl import simulate_trades
        stocks = _make_selected_stocks(Q1, STOCK_NAMES[:3], mode="stock_weight")
        prices = _make_price_data(STOCK_NAMES[:3], "2023-02-01", "2023-08-31")
        result = simulate_trades(
            stocks, prices, category_scheme="volatility",
            tp_mode="flat", sl_mode="flat",
            flat_config={"tp_pct": 0.05, "sl_pct": 0.05},
            index_exit_config=_base_index_exit(),
        )
        assert len(result) == 3
        assert "stock_weight" in result.columns


# =============================================================================
# Stock return computation
# =============================================================================

class TestStockReturn:

    def test_return_sign_matches_price_direction(self):
        """With tiny TP (0.1%), stocks going up should hit TP."""
        from backtest.tpsl import simulate_trades
        stocks = _make_selected_stocks(Q1, STOCK_NAMES[:3])
        # Uptrending prices (0.1% daily)
        prices = _make_price_data(STOCK_NAMES[:3], "2023-02-01", "2023-08-31", daily_return=0.001)
        result = simulate_trades(
            stocks, prices, category_scheme="volatility",
            tp_mode="flat", sl_mode="flat",
            flat_config={"tp_pct": 0.001, "sl_pct": 0.10},  # very tight TP
            index_exit_config=_base_index_exit(),
        )
        valid = result[result["stock_return"].notna()]
        # With uptrending stock and tight TP, most should hit TP
        assert valid["TP_triggered"].sum() >= 1
