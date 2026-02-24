"""Portfolio value and drawdown metrics."""

import pandas as pd
import numpy as np

from config.defaults import (
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    DAYS_PER_YEAR,
    VAR_CONFIDENCE_LEVEL,
)
from utils.metrics import compute_cagr, compute_max_drawdown, compute_calmar_ratio


def compute_portfolio_metrics(daily_pf: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> dict:
    """
    Compute comprehensive portfolio performance metrics from daily portfolio values.

    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - risk_free_rate: Annual risk-free rate (default 6.5% for India)

    Returns:
    - Dictionary with metrics
    """
    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Daily returns
    df['daily_return'] = df['portfolio_value'].pct_change()

    # Basic return metrics
    initial_value = df['portfolio_value'].iloc[0]
    final_value = df['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value

    # Time period
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    years = (end_date - start_date).days / DAYS_PER_YEAR

    # CAGR (shared implementation)
    cagr = compute_cagr(daily_pf) or (total_return if years <= 0 else 0)

    # Volatility (annualized)
    daily_vol = df['daily_return'].std()
    annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe Ratio
    excess_return = cagr - risk_free_rate
    sharpe = excess_return / annualized_vol if annualized_vol > 0 else 0

    # Sortino Ratio
    target_return = risk_free_rate / TRADING_DAYS_PER_YEAR
    downside_diff = df['daily_return'] - target_return
    downside_diff = np.where(downside_diff < 0, downside_diff, 0)
    downside_dev = np.sqrt(np.mean(downside_diff**2)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = excess_return / downside_dev if downside_dev > 0 else 0

    # Maximum Drawdown (shared implementation)
    mdd = compute_max_drawdown(daily_pf)
    max_drawdown = -(mdd if mdd is not None else 0)  # negative for display

    # Calmar Ratio (shared implementation)
    calmar_val = compute_calmar_ratio(daily_pf)
    calmar = calmar_val if calmar_val is not None else 0

    # Value at Risk (95%)
    var_95 = df['daily_return'].quantile(VAR_CONFIDENCE_LEVEL)

    # Best and worst days
    best_day = df['daily_return'].max()
    worst_day = df['daily_return'].min()

    # Positive days percentage
    valid_returns = df['daily_return'].dropna()
    positive_days_pct = (valid_returns > 0).sum() / len(valid_returns) if len(valid_returns) > 0 else 0

    return {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'trading_days': len(df),
        'years': round(years, 2),
        'initial_value': initial_value,
        'final_value': final_value,
        'total_return_pct': round(total_return * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'volatility_pct': round(annualized_vol * 100, 2),
        'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3),
        'max_drawdown_pct': round(max_drawdown * 100, 2),
        'calmar_ratio': round(calmar, 3),
        'var_95_pct': round(var_95 * 100, 2),
        'best_day_pct': round(best_day * 100, 2),
        'worst_day_pct': round(worst_day * 100, 2),
        'positive_days_pct': round(positive_days_pct * 100, 2)
    }


def compute_drawdown_series(daily_pf: pd.DataFrame) -> pd.DataFrame:
    """Compute drawdown series from daily portfolio values."""
    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['cummax'] = df['portfolio_value'].cummax()
    df['drawdown'] = df['portfolio_value'] - df['cummax']
    df['drawdown_pct'] = (df['drawdown'] / df['cummax']) * 100

    return df[['date', 'portfolio_value', 'cummax', 'drawdown', 'drawdown_pct']]


def compute_cash_metrics(daily_pf: pd.DataFrame) -> dict | None:
    """
    Compute cash analytics from daily portfolio values that include cash_in_hand.

    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
                and optionally 'cash_in_hand'.

    Returns dict with:
      - avg_cash_pct: average cash as % of portfolio
      - avg_cash: average cash amount (₹)
      - cash_by_quarter: DataFrame(quarter, mean_cash_pct, mean_cash)
    Returns None if cash_in_hand is not present.
    """
    if 'cash_in_hand' not in daily_pf.columns:
        return None

    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ignore_index=True)

    df['_cash_pct'] = (df['cash_in_hand'] / df['portfolio_value']) * 100

    avg_cash_pct = float(df['_cash_pct'].mean())
    avg_cash = float(df['cash_in_hand'].mean())

    cash_by_quarter = (
        df.groupby('quarter')
        .agg(mean_cash_pct=('_cash_pct', 'mean'), mean_cash=('cash_in_hand', 'mean'))
        .reset_index()
    )

    return {
        'avg_cash_pct': round(avg_cash_pct, 2),
        'avg_cash': round(avg_cash, 2),
        'cash_by_quarter': cash_by_quarter,
    }


def compute_benchmark_metrics(comparison_df: pd.DataFrame) -> dict:
    """
    Compute benchmark comparison metrics from portfolio vs index data.

    Parameters:
    - comparison_df: DataFrame with columns
        ['date', 'pf_value', 'pf_return', 'index_fund_value', 'index_return', 'alpha']

    Returns:
    - Dictionary with comparison metrics
    """
    df = comparison_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Daily returns
    df['pf_daily_return'] = df['pf_value'].pct_change()
    df['index_daily_return'] = df['index_fund_value'].pct_change()

    # Drop NaN
    valid = df[['pf_daily_return', 'index_daily_return']].dropna()

    # Beta
    covariance = valid['pf_daily_return'].cov(valid['index_daily_return'])
    index_variance = valid['index_daily_return'].var()
    beta = covariance / index_variance if index_variance > 0 else 1

    # Tracking Error (annualized)
    df['excess_return'] = df['pf_daily_return'] - df['index_daily_return']
    tracking_error = df['excess_return'].std() * np.sqrt(252)

    # Information Ratio
    excess_return_mean = df['excess_return'].mean() * 252
    information_ratio = excess_return_mean / tracking_error if tracking_error > 0 else 0

    # Up/Down Capture Ratios
    up_days = valid[valid['index_daily_return'] > 0]
    down_days = valid[valid['index_daily_return'] < 0]

    up_capture = (up_days['pf_daily_return'].mean() / up_days['index_daily_return'].mean() * 100
                  if len(up_days) > 0 and up_days['index_daily_return'].mean() != 0 else 100)
    down_capture = (down_days['pf_daily_return'].mean() / down_days['index_daily_return'].mean() * 100
                    if len(down_days) > 0 and down_days['index_daily_return'].mean() != 0 else 100)

    # Correlation
    correlation = valid['pf_daily_return'].corr(valid['index_daily_return'])

    # Final metrics
    final_pf_return = df['pf_return'].iloc[-1]
    final_index_return = df['index_return'].iloc[-1]
    final_alpha = df['alpha'].iloc[-1]

    # Outperformance days
    outperform_days = (df['alpha'].diff() > 0).sum()
    total_days = len(df) - 1
    outperform_pct = outperform_days / total_days * 100 if total_days > 0 else 0

    return {
        'portfolio_return_pct': round(final_pf_return, 2),
        'index_return_pct': round(final_index_return, 2),
        'alpha_pct': round(final_alpha, 2),
        'beta': round(beta, 3),
        'tracking_error_pct': round(tracking_error * 100, 2),
        'information_ratio': round(information_ratio, 3),
        'up_capture_pct': round(up_capture, 2),
        'down_capture_pct': round(down_capture, 2),
        'correlation': round(correlation, 3),
        'outperformance_days_pct': round(outperform_pct, 2)
    }
