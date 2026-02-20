"""
Shared pytest fixtures for alphaBT backtesting tests.

All synthetic data is deterministic — no randomness, no dependency on real data files.
"""

import pytest
import pandas as pd
import numpy as np


# =============================================================================
# QUARTER CONSTANTS
# =============================================================================

# Quarter 202302: Feb 15 → May 30, 2023
# Quarter 202305: May 31 → Aug 14, 2023
Q1 = 202302
Q2 = 202305
Q1_START = pd.Timestamp("2023-02-15")
Q1_END = pd.Timestamp("2023-05-30")
Q2_START = pd.Timestamp("2023-05-31")
Q2_END = pd.Timestamp("2023-08-14")

# Stocks used across fixtures
STOCK_NAMES = [
    "STOCK_A", "STOCK_B", "STOCK_C",  # high_volatility / largecap
    "STOCK_D", "STOCK_E", "STOCK_F",  # medium_volatility / midcap
    "STOCK_G", "STOCK_H", "STOCK_I",  # low_volatility / smallcap
]

VOL_CATEGORIES = {
    "STOCK_A": "high_volatility", "STOCK_B": "high_volatility", "STOCK_C": "high_volatility",
    "STOCK_D": "medium_volatility", "STOCK_E": "medium_volatility", "STOCK_F": "medium_volatility",
    "STOCK_G": "low_volatility", "STOCK_H": "low_volatility", "STOCK_I": "low_volatility",
}

MCAP_CATEGORIES = {
    "STOCK_A": "largecap", "STOCK_B": "largecap", "STOCK_C": "largecap",
    "STOCK_D": "midcap", "STOCK_E": "midcap", "STOCK_F": "midcap",
    "STOCK_G": "smallcap", "STOCK_H": "smallcap", "STOCK_I": "smallcap",
}

VOLATILITIES = {
    "STOCK_A": 0.035, "STOCK_B": 0.030, "STOCK_C": 0.028,
    "STOCK_D": 0.018, "STOCK_E": 0.016, "STOCK_F": 0.015,
    "STOCK_G": 0.008, "STOCK_H": 0.007, "STOCK_I": 0.006,
}

PROBABILITIES = {
    "STOCK_A": 0.85, "STOCK_B": 0.80, "STOCK_C": 0.75,
    "STOCK_D": 0.70, "STOCK_E": 0.65, "STOCK_F": 0.60,
    "STOCK_G": 0.55, "STOCK_H": 0.50, "STOCK_I": 0.45,
}


# =============================================================================
# INPUT DATA FIXTURE
# =============================================================================

@pytest.fixture
def sample_input_data():
    """
    Inference input data: 9 stocks × 2 quarters = 18 rows.
    Columns: [quarter, co_name, category, prob, volatility]
    """
    rows = []
    for q in [Q1, Q2]:
        for name in STOCK_NAMES:
            rows.append({
                "quarter": q,
                "co_name": name,
                "category": MCAP_CATEGORIES[name],
                "prob": PROBABILITIES[name],
                "volatility": VOLATILITIES[name],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_input_data_volatility():
    """
    Same as sample_input_data but with volatility-based category labels.
    """
    rows = []
    for q in [Q1, Q2]:
        for name in STOCK_NAMES:
            rows.append({
                "quarter": q,
                "co_name": name,
                "category": VOL_CATEGORIES[name],
                "prob": PROBABILITIES[name],
                "volatility": VOLATILITIES[name],
            })
    return pd.DataFrame(rows)


# =============================================================================
# PRICE DATA FIXTURE
# =============================================================================

def _make_price_data(stocks, start_date, end_date, base_price=100.0, daily_return=0.001):
    """
    Generate deterministic OHLCV data for a list of stocks over a date range.
    
    Each stock starts at base_price and grows by daily_return each day.
    O/H/L/C are derived deterministically:
      close = base_price * (1 + daily_return)^day_index
      open  = close * 0.999
      high  = close * 1.005
      low   = close * 0.995
    """
    dates = pd.bdate_range(start=start_date, end=end_date)
    rows = []
    for name in stocks:
        for i, d in enumerate(dates):
            close = base_price * (1 + daily_return) ** i
            rows.append({
                "date": d,
                "co_name": name,
                "open": round(close * 0.999, 2),
                "high": round(close * 1.005, 2),
                "low": round(close * 0.995, 2),
                "close": round(close, 2),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_price_data():
    """
    OHLCV data for all 9 stocks covering both quarters (Feb 15 → Aug 14, 2023).
    Deterministic: ~0.1% daily growth from base price 100.
    """
    return _make_price_data(
        STOCK_NAMES,
        start_date="2023-02-01",   # start a bit early so entry window always has data
        end_date="2023-08-31",     # end a bit late for safety
        base_price=100.0,
        daily_return=0.001,
    )


# =============================================================================
# INDEX DATA FIXTURE
# =============================================================================

@pytest.fixture
def sample_index_data():
    """
    Index data with columns [date, value]. Deterministic ~0.05% daily growth from 1000.
    """
    dates = pd.bdate_range(start="2023-02-01", end="2023-08-31")
    values = [1000.0 * (1 + 0.0005) ** i for i in range(len(dates))]
    return pd.DataFrame({"date": dates, "value": values})


# =============================================================================
# PRESELECTED PORTFOLIO FIXTURE
# =============================================================================

@pytest.fixture
def sample_preselected_data():
    """
    Pre-built portfolio: 6 stocks per quarter, equal weights summing to 1.0.
    Columns: [quarter, co_name, stock_weight, category]
    """
    stocks = STOCK_NAMES[:6]
    rows = []
    for q in [Q1, Q2]:
        for name in stocks:
            rows.append({
                "quarter": q,
                "co_name": name,
                "stock_weight": 1.0 / len(stocks),
                "category": MCAP_CATEGORIES[name],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_preselected_no_category():
    """
    Pre-built portfolio without category column.
    """
    stocks = STOCK_NAMES[:6]
    rows = []
    for q in [Q1, Q2]:
        for name in stocks:
            rows.append({
                "quarter": q,
                "co_name": name,
                "stock_weight": 1.0 / len(stocks),
            })
    return pd.DataFrame(rows)


# =============================================================================
# CONFIG FIXTURES
# =============================================================================

@pytest.fixture
def minimal_index_exit_config():
    """Minimal index_exit config with everything disabled."""
    return {
        "regime_filter": {
            "enabled": False,
            "ma_period": 20,
            "exit_threshold": -0.02,
        },
        "vol_adjustment": {
            "enabled": False,
            "lookback": 20,
            "high_vol_threshold": 0.25,
            "low_vol_threshold": 0.15,
            "high_vol_multiplier": 1.5,
            "low_vol_multiplier": 0.8,
        },
    }


@pytest.fixture
def base_config(minimal_index_exit_config):
    """
    Minimal valid config dict for backtest_core().
    Uses flat TP/SL, category_based selection, volatility scheme.
    """
    return {
        "first_quarter": Q1,
        "last_quarter": Q2,
        "category_scheme": "volatility",
        "run_stock_selection": True,
        "selection_type": "category_based",
        "category_counts": [2, 2, 2],
        "category_weights": [0.33, 0.33, 0.34],
        "selection_method": "probability",
        "min_prob_threshold": None,
        "category_based_selection_weighting_scheme": "use_category_weights",
        "tp_mode": "flat",
        "sl_mode": "flat",
        "tp_enabled": True,
        "sl_enabled": True,
        "flat_config": {"tp_pct": 0.05, "sl_pct": 0.05},
        "entry_price_window": 3,
        "index_exit": minimal_index_exit_config,
        "generate_report": False,
    }


# =============================================================================
# REPORTING FIXTURES — Synthetic portfolios for metric verification
# =============================================================================

def _make_daily_pf(values, start_date="2023-01-02", quarter=202302):
    """Helper: create a daily_pf DataFrame from a list of portfolio values."""
    dates = pd.bdate_range(start=start_date, periods=len(values))
    return pd.DataFrame({
        "date": dates,
        "portfolio_value": values,
        "quarter": quarter,
    })


@pytest.fixture
def constant_portfolio():
    """Portfolio that stays at ₹1B for 252 trading days (zero return)."""
    return _make_daily_pf([1_000_000_000.0] * 252)


@pytest.fixture
def linear_growth_portfolio():
    """
    Portfolio growing linearly from ₹1B to ₹1.1B over 252 trading days.
    Total return = 10%.
    """
    values = np.linspace(1_000_000_000.0, 1_100_000_000.0, 252)
    return _make_daily_pf(values.tolist())


@pytest.fixture
def known_cagr_portfolio():
    """
    Portfolio growing from ₹1B to ₹1.5B over ~2 years (504 trading days).
    CAGR ≈ 22.47%.
    """
    n_days = 504
    initial = 1_000_000_000.0
    final = 1_500_000_000.0
    # Exponential growth to achieve exact final value
    daily_r = (final / initial) ** (1 / n_days) - 1
    values = [initial * (1 + daily_r) ** i for i in range(n_days)]
    return _make_daily_pf(values)


@pytest.fixture
def drawdown_portfolio():
    """
    Portfolio: rises to ₹1.2B, drops to ₹0.9B, recovers to ₹1.1B.
    Max drawdown = (0.9 - 1.2) / 1.2 = -25%.
    84 days rise + 84 days fall + 84 days recovery = 252 days.
    """
    n = 84
    rise = np.linspace(1_000_000_000.0, 1_200_000_000.0, n)
    fall = np.linspace(1_200_000_000.0, 900_000_000.0, n)
    recover = np.linspace(900_000_000.0, 1_100_000_000.0, n)
    values = np.concatenate([rise, fall, recover])
    return _make_daily_pf(values.tolist())


@pytest.fixture
def comparison_df_identical():
    """
    Portfolio vs index DataFrame where portfolio exactly matches the index.
    252 days, both grow from 1B to 1.1B linearly.
    """
    n = 252
    dates = pd.bdate_range(start="2023-01-02", periods=n)
    values = np.linspace(1_000_000_000.0, 1_100_000_000.0, n)
    returns = ((values - values[0]) / values[0]) * 100
    return pd.DataFrame({
        "date": dates,
        "pf_value": values,
        "pf_return": returns,
        "index_fund_value": values.copy(),
        "index_return": returns.copy(),
        "alpha": np.zeros(n),
    })


@pytest.fixture
def comparison_df_outperforming():
    """
    Portfolio outperforms index. Portfolio: 1B→1.2B, Index: 1B→1.05B.
    """
    n = 252
    dates = pd.bdate_range(start="2023-01-02", periods=n)
    pf_values = np.linspace(1_000_000_000.0, 1_200_000_000.0, n)
    idx_values = np.linspace(1_000_000_000.0, 1_050_000_000.0, n)
    pf_returns = ((pf_values - pf_values[0]) / pf_values[0]) * 100
    idx_returns = ((idx_values - idx_values[0]) / idx_values[0]) * 100
    alpha = pf_returns - idx_returns
    return pd.DataFrame({
        "date": dates,
        "pf_value": pf_values,
        "pf_return": pf_returns,
        "index_fund_value": idx_values,
        "index_return": idx_returns,
        "alpha": alpha,
    })


# =============================================================================
# PHASE 1: REPORTING ANALYTICS FIXTURES
# =============================================================================

@pytest.fixture
def daily_returns_df():
    """
    Date-indexed DataFrame with 'Portfolio' and 'Benchmark' daily returns.
    ~504 trading days (2 years) of deterministic small positive returns.
    """
    n = 504
    dates = pd.bdate_range(start="2022-01-03", periods=n)
    np.random.seed(42)
    pf_returns = np.random.normal(0.0004, 0.01, n)     # ~10% annual, ~16% vol
    bm_returns = np.random.normal(0.0003, 0.008, n)    # ~7.5% annual, ~12.7% vol
    df = pd.DataFrame({"Portfolio": pf_returns, "Benchmark": bm_returns}, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def monthly_returns_df(daily_returns_df):
    """Monthly compounded returns derived from daily_returns_df."""
    from reporting.backtest_report import compute_monthly_returns_from_daily
    return compute_monthly_returns_from_daily(daily_returns_df, input_frequency="daily")


@pytest.fixture
def trade_results_with_mcap():
    """
    Synthetic trade results with mcap/category info for reporting tests.
    2 quarters × varying stocks per category with TP/SL outcomes.
    """
    rows = []
    for q in [Q1, Q2]:
        for i, name in enumerate(STOCK_NAMES[:6]):
            cat = MCAP_CATEGORIES[name]
            tp_triggered = i % 3 == 0
            sl_triggered = i % 3 == 1
            rows.append({
                "quarter": q,
                "co_name": name,
                "cat": cat,
                "cat_weight": 0.33 if cat == "largecap" else (0.33 if cat == "midcap" else 0.34),
                "mcap_category": cat,
                "entry_price": 100.0 + i * 10,
                "exit_price": (100.0 + i * 10) * (1.05 if tp_triggered else (0.95 if sl_triggered else 1.02)),
                "holding_period": 20 + i * 5,
                "TP_triggered": tp_triggered,
                "SL_triggered": sl_triggered,
                "stock_return": 0.05 if tp_triggered else (-0.05 if sl_triggered else 0.02),
            })
    return pd.DataFrame(rows)


# =============================================================================
# PHASE 2: DYNAMIC LEVELS FIXTURES
# =============================================================================

@pytest.fixture
def price_data_for_atr():
    """
    Price data for a single stock with known ATR.
    30 trading days of data with controlled H/L/C values.
    """
    dates = pd.bdate_range(start="2023-01-02", periods=30)
    rows = []
    for i, d in enumerate(dates):
        close = 100 + i * 0.5  # steady uptrend
        rows.append({
            "date": d,
            "co_name": "TEST_STOCK",
            "open": close - 0.3,
            "high": close + 1.0,    # consistent range
            "low": close - 1.0,
            "close": close,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def index_data_for_volatility():
    """
    Index data for volatility/regime calculations.
    60 trading days of deterministic price data.
    """
    dates = pd.bdate_range(start="2023-01-02", periods=60)
    values = [1000.0 * (1 + 0.0005) ** i for i in range(60)]
    return pd.DataFrame({"date": dates, "value": values})


# =============================================================================
# PHASE 4: SIMULATION FIXTURES
# =============================================================================

@pytest.fixture
def sized_trades():
    """
    Trades with position sizing columns for simulation tests.
    6 stocks in one quarter with allocated capital and shares.
    """
    stocks = STOCK_NAMES[:6]
    exit_dates = pd.bdate_range(start="2023-04-01", periods=6)
    rows = []
    for i, name in enumerate(stocks):
        entry_price = 100.0 + i * 10
        exit_price = entry_price * (1 + 0.02 * (i - 2))  # varied returns
        allocated_capital = 1_000_000_000.0 / 6
        shares = allocated_capital / entry_price
        rows.append({
            "quarter": Q1,
            "co_name": name,
            "cat": MCAP_CATEGORIES[name],
            "cat_weight": 0.33 if MCAP_CATEGORIES[name] != "smallcap" else 0.34,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_date": exit_dates[i],
            "allocated_capital": allocated_capital,
            "shares": shares,
        })
    return pd.DataFrame(rows)
