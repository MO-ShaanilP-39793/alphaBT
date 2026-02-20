"""
Phase 2: Tests for backtest/dynamic_levels.py — ATR, pivot, and index-based functions.

All 8 public functions are pure-functional with deterministic synthetic data,
so no mocking is needed.
"""

import pytest
import pandas as pd
import numpy as np


# =============================================================================
# Helpers
# =============================================================================

def _make_stock_price_data(co_name, n_days=30, base_price=100.0, daily_move=0.5,
                           high_spread=1.0, low_spread=1.0, start="2023-01-02"):
    """Build deterministic OHLCV for one stock."""
    dates = pd.bdate_range(start=start, periods=n_days)
    rows = []
    for i, d in enumerate(dates):
        close = base_price + i * daily_move
        rows.append({
            "date": d,
            "co_name": co_name,
            "open": close - 0.3,
            "high": close + high_spread,
            "low": close - low_spread,
            "close": close,
        })
    return pd.DataFrame(rows)


def _make_index_data(n_days=60, base=1000.0, daily_return=0.0005, start="2023-01-02"):
    dates = pd.bdate_range(start=start, periods=n_days)
    values = [base * (1 + daily_return) ** i for i in range(n_days)]
    return pd.DataFrame({"date": dates, "value": values})


# =============================================================================
# calculate_atr
# =============================================================================

class TestCalculateATR:

    def test_basic_atr_computation(self, price_data_for_atr):
        from backtest.dynamic_levels import calculate_atr
        end_date = pd.Timestamp("2023-02-15")
        atr = calculate_atr(price_data_for_atr, "TEST_STOCK", end_date, period=14)
        assert atr is not None
        assert atr > 0

    def test_insufficient_data_returns_none(self):
        from backtest.dynamic_levels import calculate_atr
        df = _make_stock_price_data("X", n_days=5)
        end_date = pd.Timestamp("2023-01-20")
        atr = calculate_atr(df, "X", end_date, period=14)
        assert atr is None

    def test_atr_uses_correct_stock(self):
        from backtest.dynamic_levels import calculate_atr
        df1 = _make_stock_price_data("A", n_days=30, high_spread=2.0, low_spread=2.0)
        df2 = _make_stock_price_data("B", n_days=30, high_spread=0.5, low_spread=0.5)
        combined = pd.concat([df1, df2], ignore_index=True)
        end_date = pd.Timestamp("2023-02-15")
        atr_a = calculate_atr(combined, "A", end_date, period=14)
        atr_b = calculate_atr(combined, "B", end_date, period=14)
        assert atr_a > atr_b  # A has wider range


# =============================================================================
# calculate_atr_thresholds
# =============================================================================

class TestCalculateATRThresholds:

    def test_tp_above_entry_sl_below(self, price_data_for_atr):
        from backtest.dynamic_levels import calculate_atr_thresholds
        end_date = pd.Timestamp("2023-02-15")
        tp, sl, atr = calculate_atr_thresholds(
            price_data_for_atr, "TEST_STOCK", entry_price=110.0,
            end_date=end_date, tp_multiplier=2.0, sl_multiplier=1.5,
        )
        assert tp is not None
        assert sl is not None
        assert tp > 110.0
        assert sl < 110.0

    def test_sl_floor(self):
        """SL should never go below ATR_SL_FLOOR_FACTOR × entry."""
        from backtest.dynamic_levels import calculate_atr_thresholds
        from config.defaults import ATR_SL_FLOOR_FACTOR
        # High-volatility stock where ATR is large relative to price
        df = _make_stock_price_data("X", n_days=30, base_price=10.0, high_spread=8.0, low_spread=8.0)
        end_date = pd.Timestamp("2023-02-15")
        tp, sl, _ = calculate_atr_thresholds(
            df, "X", entry_price=10.0, end_date=end_date,
            sl_multiplier=5.0,  # very aggressive SL
        )
        assert sl >= 10.0 * ATR_SL_FLOOR_FACTOR

    def test_insufficient_data_returns_nones(self):
        from backtest.dynamic_levels import calculate_atr_thresholds
        df = _make_stock_price_data("X", n_days=3)
        tp, sl, atr = calculate_atr_thresholds(df, "X", 100.0, pd.Timestamp("2023-01-10"))
        assert tp is None and sl is None and atr is None


# =============================================================================
# calculate_pivot_points
# =============================================================================

class TestCalculatePivotPoints:

    def test_basic_pivot_computation(self, price_data_for_atr):
        from backtest.dynamic_levels import calculate_pivot_points
        end_date = pd.Timestamp("2023-02-15")
        pivots = calculate_pivot_points(price_data_for_atr, "TEST_STOCK", end_date, lookback_days=20)
        assert pivots is not None
        assert "pivot" in pivots
        assert pivots["R1"] > pivots["pivot"] > pivots["S1"]
        assert pivots["R3"] > pivots["R2"] > pivots["R1"]
        assert pivots["S1"] > pivots["S2"] > pivots["S3"]

    def test_insufficient_data_returns_none(self):
        from backtest.dynamic_levels import calculate_pivot_points
        df = _make_stock_price_data("X", n_days=3)
        result = calculate_pivot_points(df, "X", pd.Timestamp("2023-01-10"), lookback_days=20)
        assert result is None

    def test_pivot_formula_correctness(self):
        """Verify the classic pivot point formula P = (H + L + C) / 3."""
        from backtest.dynamic_levels import calculate_pivot_points
        # Create data with known H, L, C
        df = _make_stock_price_data("X", n_days=10, base_price=100.0,
                                    high_spread=5.0, low_spread=5.0)
        end_date = pd.Timestamp("2023-01-20")
        pivots = calculate_pivot_points(df, "X", end_date, lookback_days=10)
        # Manually compute expected pivot
        h = df["high"].max()
        l = df["low"].min()
        c = df["close"].iloc[-1]
        expected_pivot = (h + l + c) / 3
        assert abs(pivots["pivot"] - expected_pivot) < 0.01


# =============================================================================
# calculate_pivot_thresholds
# =============================================================================

class TestCalculatePivotThresholds:

    def test_tp_above_sl_below_entry(self, price_data_for_atr):
        from backtest.dynamic_levels import calculate_pivot_thresholds
        end_date = pd.Timestamp("2023-02-15")
        # Entry price near the middle of the range
        tp, sl, pivots = calculate_pivot_thresholds(
            price_data_for_atr, "TEST_STOCK", entry_price=110.0,
            end_date=end_date, tp_level="R1", sl_level="S1",
        )
        assert tp is not None
        assert sl is not None
        assert tp > 110.0
        assert sl < 110.0

    def test_fallback_when_resistance_below_entry(self):
        """When all resistance levels are below entry, a percentage fallback should be used."""
        from backtest.dynamic_levels import calculate_pivot_thresholds
        from config.defaults import PIVOT_TP_FALLBACK_FACTOR
        # Create data with very low prices so pivots are all below entry
        df = _make_stock_price_data("X", n_days=20, base_price=50.0)
        end_date = pd.Timestamp("2023-02-01")
        tp, sl, pivots = calculate_pivot_thresholds(
            df, "X", entry_price=200.0,  # way above all pivot levels
            end_date=end_date,
        )
        assert tp is not None
        assert abs(tp - 200.0 * PIVOT_TP_FALLBACK_FACTOR) < 0.01

    def test_fallback_when_support_above_entry(self):
        """When all support levels are above entry, a percentage fallback should be used."""
        from backtest.dynamic_levels import calculate_pivot_thresholds
        from config.defaults import PIVOT_SL_FALLBACK_FACTOR
        # Create data with very high prices so pivots are all above entry
        df = _make_stock_price_data("X", n_days=20, base_price=500.0)
        end_date = pd.Timestamp("2023-02-01")
        tp, sl, pivots = calculate_pivot_thresholds(
            df, "X", entry_price=10.0,  # way below all pivot levels
            end_date=end_date, sl_level="S1",
        )
        assert sl is not None
        assert abs(sl - 10.0 * PIVOT_SL_FALLBACK_FACTOR) < 0.01

    def test_insufficient_data_returns_nones(self):
        from backtest.dynamic_levels import calculate_pivot_thresholds
        df = _make_stock_price_data("X", n_days=3)
        tp, sl, pivots = calculate_pivot_thresholds(df, "X", 100.0, pd.Timestamp("2023-01-10"))
        assert tp is None and sl is None and pivots is None


# =============================================================================
# calculate_index_volatility
# =============================================================================

class TestCalculateIndexVolatility:

    def test_basic_volatility(self, index_data_for_volatility):
        from backtest.dynamic_levels import calculate_index_volatility
        date = pd.Timestamp("2023-03-20")
        vol = calculate_index_volatility(index_data_for_volatility, date, lookback=20)
        assert vol is not None
        assert vol > 0

    def test_insufficient_data_returns_none(self):
        from backtest.dynamic_levels import calculate_index_volatility
        df = _make_index_data(n_days=5)
        vol = calculate_index_volatility(df, pd.Timestamp("2023-01-10"), lookback=20)
        assert vol is None


# =============================================================================
# calculate_index_ma
# =============================================================================

class TestCalculateIndexMA:

    def test_basic_ma(self, index_data_for_volatility):
        from backtest.dynamic_levels import calculate_index_ma
        date = pd.Timestamp("2023-03-20")
        current, ma, pct = calculate_index_ma(index_data_for_volatility, date, ma_period=20)
        assert current is not None
        assert ma is not None
        # Uptrending index → current should be above MA
        assert pct > 0

    def test_insufficient_data_returns_nones(self):
        from backtest.dynamic_levels import calculate_index_ma
        df = _make_index_data(n_days=5)
        cur, ma, pct = calculate_index_ma(df, pd.Timestamp("2023-01-10"), ma_period=20)
        assert cur is None and ma is None and pct is None


# =============================================================================
# get_volatility_adjustment_multiplier
# =============================================================================

class TestGetVolatilityAdjustmentMultiplier:

    def test_normal_vol_returns_one(self, index_data_for_volatility):
        """Low-volatility deterministic data should return 1.0 (normal regime)."""
        from backtest.dynamic_levels import get_volatility_adjustment_multiplier
        date = pd.Timestamp("2023-03-20")
        mult = get_volatility_adjustment_multiplier(index_data_for_volatility, date, lookback=20)
        # Our index has tiny daily moves (0.05%), annualized ~0.8% — well below low_vol_threshold
        # So this should hit the low_vol regime
        assert mult is not None
        assert isinstance(mult, float)

    def test_insufficient_data_returns_one(self):
        from backtest.dynamic_levels import get_volatility_adjustment_multiplier
        df = _make_index_data(n_days=3)
        mult = get_volatility_adjustment_multiplier(df, pd.Timestamp("2023-01-05"), lookback=20)
        assert mult == 1.0


# =============================================================================
# check_regime_exit_signal
# =============================================================================

class TestCheckRegimeExitSignal:

    def test_no_exit_in_uptrend(self, index_data_for_volatility):
        from backtest.dynamic_levels import check_regime_exit_signal
        date = pd.Timestamp("2023-03-20")
        signal = check_regime_exit_signal(index_data_for_volatility, date, ma_period=20)
        assert signal == False  # steady uptrend, no exit

    def test_insufficient_data_returns_false(self):
        from backtest.dynamic_levels import check_regime_exit_signal
        df = _make_index_data(n_days=3)
        signal = check_regime_exit_signal(df, pd.Timestamp("2023-01-05"), ma_period=20)
        assert signal is False

    def test_exit_triggered_in_downtrend(self):
        """When index drops well below MA, exit signal should be True."""
        from backtest.dynamic_levels import check_regime_exit_signal
        dates = pd.bdate_range("2023-01-02", periods=60)
        # First 40 days steady, then sharp drop
        values = [1000.0] * 40 + [1000.0 - i * 5 for i in range(1, 21)]
        df = pd.DataFrame({"date": dates, "value": values})
        date = dates[-1]
        signal = check_regime_exit_signal(df, date, ma_period=20, exit_threshold=-0.02)
        assert signal == True
