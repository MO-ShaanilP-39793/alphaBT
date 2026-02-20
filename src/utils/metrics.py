"""
Shared portfolio metrics used by both tuning and reporting.

Canonical implementations of CAGR, Maximum Drawdown, and Calmar Ratio.
All other modules should import from here instead of implementing their own.
"""

import pandas as pd
from typing import Optional

from config.defaults import DAYS_PER_YEAR, DEFAULT_CALMAR_CAP


def compute_cagr(daily_pf_values: pd.DataFrame) -> Optional[float]:
    """
    Compute Compound Annual Growth Rate (CAGR) from daily portfolio values.

    CAGR = (Final Value / Initial Value) ^ (1 / Years) - 1

    Parameters:
        daily_pf_values: DataFrame with columns ['date', 'portfolio_value', 'quarter']

    Returns:
        Annualized CAGR as a float, or None if computation fails
    """
    if daily_pf_values is None or daily_pf_values.empty:
        return None

    df = daily_pf_values.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ignore_index=True)

    if len(df) < 2:
        return None

    initial_value = df['portfolio_value'].iloc[0]
    final_value = df['portfolio_value'].iloc[-1]

    if initial_value <= 0 or final_value <= 0:
        return None

    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    years = (end_date - start_date).days / DAYS_PER_YEAR

    if years <= 0:
        return None

    cagr = (final_value / initial_value) ** (1 / years) - 1
    return cagr


def compute_max_drawdown(daily_pf_values: pd.DataFrame) -> Optional[float]:
    """
    Compute Maximum Drawdown (MDD) from daily portfolio values.

    MDD = min((Portfolio Value - Cumulative Max) / Cumulative Max)

    Returns the magnitude (positive value) of the maximum drawdown.

    Parameters:
        daily_pf_values: DataFrame with columns ['date', 'portfolio_value', 'quarter']

    Returns:
        Maximum drawdown as a positive float, or None if computation fails
    """
    if daily_pf_values is None or daily_pf_values.empty:
        return None

    df = daily_pf_values.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ignore_index=True)

    if len(df) < 2:
        return None

    df['cummax'] = df['portfolio_value'].cummax()
    df['drawdown'] = (df['portfolio_value'] - df['cummax']) / df['cummax']
    max_drawdown = df['drawdown'].min()

    return abs(max_drawdown)


def compute_calmar_ratio(
    daily_pf_values: pd.DataFrame,
    cap: float = DEFAULT_CALMAR_CAP,
) -> Optional[float]:
    """
    Compute Calmar ratio from daily portfolio values.

    Calmar Ratio = CAGR / |Max Drawdown|

    Parameters:
        daily_pf_values: DataFrame with columns ['date', 'portfolio_value', 'quarter']
        cap: Maximum Calmar ratio value to return (avoids inflated values)

    Returns:
        Calmar ratio (capped), or None if computation fails
    """
    if daily_pf_values is None or daily_pf_values.empty:
        return None

    if len(daily_pf_values) < 2:
        return None

    cagr = compute_cagr(daily_pf_values)
    if cagr is None:
        return None

    max_drawdown = compute_max_drawdown(daily_pf_values)
    if max_drawdown is None:
        return None

    if max_drawdown == 0:
        calmar = cap
    else:
        calmar = cagr / max_drawdown

    calmar = min(calmar, cap)
    return calmar
