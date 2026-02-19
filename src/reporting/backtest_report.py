"""
Consolidated Backtest Report Module

Single module that produces one Excel workbook (backtest_report_{timestamp}.xlsx)
containing all analytics, data sheets, and embedded charts for a backtesting run.

Merges functionality from the former analytics.py and summary_stats.py into a
unified report generator.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter, FuncFormatter
import io

from backtest.regime_dates import CRISIS_REGIMES, MARKET_REGIMES
from config.defaults import (
    ANNUALIZATION_FACTORS,
    SCALE_FACTORS,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    DAYS_PER_YEAR,
    VAR_CONFIDENCE_LEVEL,
    INITIAL_CAPITAL,
)


# =============================================================================
# HELPER FUNCTIONS (from summary_stats.py)
# =============================================================================

def _check_start_end_year(returns_df, start_year, end_year):
    """Check if start and end year exist in the dataframe."""
    data = returns_df.copy(deep=True)
    date_col = data.index.name
    data = data.reset_index()

    if isinstance(data[date_col].dtype, pd.core.dtypes.dtypes.PeriodDtype):
        data[date_col] = data[date_col].dt.to_timestamp().dt.year
    else:
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce').dt.year

    year_list = data[date_col].unique().tolist()

    if (start_year not in year_list) or (end_year not in year_list):
        return False
    return True


def _trim_df_year(returns_df, start_year, end_year):
    """Trim dataframe to specified year range."""
    start_year = str(start_year)
    end_year = str(end_year)

    date_col = returns_df.index.name
    returns_df = returns_df.reset_index()

    if isinstance(returns_df[date_col].dtype, pd.core.dtypes.dtypes.PeriodDtype):
        returns_df[date_col] = pd.to_datetime(returns_df[date_col].dt.to_timestamp())
    else:
        returns_df[date_col] = pd.to_datetime(returns_df[date_col], errors='coerce')

    returns_df = returns_df.set_index(date_col)
    returns_df = returns_df.loc[start_year:end_year, :]

    return returns_df


def _max_drawdown(returns_df):
    """Compute maximum drawdown from returns series."""
    log_drawdowns = (np.log(1 + returns_df).cumsum()) - (np.log(1 + returns_df).cumsum().cummax())
    max_drawdown = (np.exp(log_drawdowns) - 1).min()
    return np.round(max_drawdown, 4)


def _downside_dev(data_series, annual_factor=252):
    """Compute annualized downside deviation."""
    return data_series.loc[data_series < 0].std() * np.sqrt(annual_factor)


def _cumulative_returns(returns_df):
    """Compute cumulative returns."""
    c_r = returns_df.add(1).cumprod().sub(1)
    c_r = c_r.values[-1]
    return c_r


def _annualized_returns(data_array, annual_factor, no_data_points):
    """Compute annualized returns from cumulative returns."""
    annualized_formula = annual_factor / no_data_points
    a_r = (1 + data_array) ** annualized_formula
    a_r = a_r - 1
    return a_r


def _annualized_sd(returns_df, annual_factor):
    """Compute annualized standard deviation."""
    annualized_formula = np.sqrt(annual_factor)
    a_risk = returns_df.std()
    a_risk = a_risk * annualized_formula
    a_risk = a_risk.values
    return a_risk


def _return_risk_ratio(return_array, risk_array):
    """Compute return to risk ratio."""
    return return_array / risk_array


def _annualization_factor(input_frequency):
    """Get annualization factor based on data timeline."""
    return ANNUALIZATION_FACTORS.get(input_frequency, ANNUALIZATION_FACTORS['daily'])


def _scale_factor_roll(returns_frequency, roll_type):
    """Get scaling factor for rolling calculations."""
    key = f"{returns_frequency}_{roll_type}"
    return SCALE_FACTORS.get(key, SCALE_FACTORS['daily_yearly'])


def _compute_probability_buckets(returns_df, lower_bound=None, upper_bound=None):
    """Compute probability of returns within specified bounds."""
    if lower_bound is None:
        probability = (returns_df < upper_bound).mean()
    elif upper_bound is None:
        probability = (returns_df > lower_bound).mean()
    else:
        probability = ((returns_df > lower_bound) & (returns_df < upper_bound)).mean()
    return probability


# =============================================================================
# PORTFOLIO METRICS (from analytics.py — works on daily_pf DataFrame)
# =============================================================================

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

    # CAGR
    cagr = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else total_return

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

    # Maximum Drawdown
    df['cummax'] = df['portfolio_value'].cummax()
    df['drawdown'] = (df['portfolio_value'] - df['cummax']) / df['cummax']
    max_drawdown = df['drawdown'].min()

    # Calmar Ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

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


# =============================================================================
# BENCHMARK COMPARISON METRICS (from analytics.py)
# =============================================================================

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


# =============================================================================
# RETURNS-BASED ANALYSIS (from summary_stats.py — works on daily returns DataFrame)
# =============================================================================

def compute_portfolio_performance(returns_df, input_frequency="daily", start_year=None, end_year=None):
    """
    Compute comprehensive portfolio performance metrics from daily returns.

    Returns:
    - DataFrame with metrics: G-Rs100, AReturns, ARisk, Sharpe, DDown, Sortino,
      Skewness, Kurtosis, %Up_Periods, %Down_Periods
    """
    input_frequency = input_frequency.strip().lower()
    if input_frequency not in ["daily", "monthly", "yearly"]:
        raise ValueError("data_timeline must be 'daily', 'monthly', or 'yearly'")

    annual_factor = _annualization_factor(input_frequency)

    if start_year is not None and end_year is not None:
        if _check_start_end_year(returns_df, start_year, end_year):
            returns_df = _trim_df_year(returns_df, start_year, end_year)
        else:
            return pd.DataFrame()

    if returns_df.empty:
        return pd.DataFrame()

    cum_ret = _cumulative_returns(returns_df)
    ann_ret = _annualized_returns(cum_ret, annual_factor, returns_df.shape[0])
    ann_risk = _annualized_sd(returns_df, annual_factor)
    ann_sharpe = _return_risk_ratio(ann_ret, ann_risk)
    downside_risk = returns_df.apply(lambda x: _downside_dev(x, annual_factor)).values
    ann_sortino = _return_risk_ratio(ann_ret, downside_risk)
    period_drawdown = _max_drawdown(returns_df)
    skewness = returns_df.skew().values
    kurtosis = returns_df.kurt().values
    no_of_up_periods = (returns_df > 0).sum().values / returns_df.shape[0]
    no_of_down_periods = 1 - no_of_up_periods

    cum_ret_scaled = (cum_ret * 100) + 100
    ann_ret_scaled = ann_ret * 100
    ann_risk_scaled = ann_risk * 100
    period_drawdown_scaled = period_drawdown * 100
    no_of_up_periods_scaled = no_of_up_periods * 100
    no_of_down_periods_scaled = no_of_down_periods * 100

    strategy_returns = [
        cum_ret_scaled, ann_ret_scaled, ann_risk_scaled, ann_sharpe,
        period_drawdown_scaled, ann_sortino, skewness, kurtosis,
        no_of_up_periods_scaled, no_of_down_periods_scaled
    ]

    strategy_returns_df = pd.DataFrame(
        strategy_returns,
        index=["G-Rs100", "AReturns", "ARisk", "Sharpe", "DDown", "Sortino",
               "Skewness", "Kurtosis", "%Up_Periods", "%Down_Periods"],
        columns=returns_df.columns
    )

    strategy_returns_df = np.round(strategy_returns_df, 2)
    return strategy_returns_df.transpose()


def compute_rolling_performance(returns_df, input_frequency="daily", roll_period=1, roll_type="yearly"):
    """Compute rolling returns statistics with probability distributions."""
    input_frequency = input_frequency.strip().lower()
    roll_type = roll_type.strip().lower()

    scaling_factor = _scale_factor_roll(input_frequency, roll_type)
    roll_period_scaled = roll_period * scaling_factor
    timeline_factor = _annualization_factor(input_frequency)
    annualization_factor = timeline_factor / roll_period_scaled

    if returns_df.shape[0] < roll_period_scaled:
        return pd.DataFrame()

    rolling_returns = returns_df.add(1).rolling(roll_period_scaled).apply(np.prod).dropna().sub(1)
    ann_rolling_returns = rolling_returns.apply(lambda x: (1 + x) ** annualization_factor).sub(1)

    summary_df = rolling_returns.describe()
    summary_df = summary_df.drop("std", axis=0, errors='ignore')
    summary_df.iloc[1:, :] = summary_df.iloc[1:, :].apply(lambda x: (1 + x) ** annualization_factor).sub(1)

    summary_df.loc["<0%P"] = _compute_probability_buckets(ann_rolling_returns, upper_bound=0)
    summary_df.loc["0-10%P"] = _compute_probability_buckets(ann_rolling_returns, lower_bound=0, upper_bound=0.1)
    summary_df.loc["10-20%P"] = _compute_probability_buckets(ann_rolling_returns, lower_bound=0.1, upper_bound=0.2)
    summary_df.loc[">20%P"] = _compute_probability_buckets(ann_rolling_returns, lower_bound=0.2)

    summary_df.iloc[1:, :] = summary_df.iloc[1:, :] * 100
    summary_df = np.round(summary_df, 2)
    return summary_df.transpose()


def compute_calendar_year_performance(returns_df, input_frequency="daily"):
    """Compute calendar year returns."""
    input_frequency = input_frequency.strip().lower()
    if input_frequency not in ["daily", "monthly"]:
        raise ValueError("data_df_timeline must be 'daily' or 'monthly'")

    date_col = returns_df.index.name or 'Date'
    data = returns_df.reset_index()

    if isinstance(data[date_col].dtype, pd.core.dtypes.dtypes.PeriodDtype):
        data[date_col] = data[date_col].dt.to_timestamp().dt.to_period("Y")
    else:
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce').dt.to_period("Y")

    cut_off = 230 if input_frequency == "daily" else 11
    relevant_years = data[date_col].unique()[data[date_col].value_counts().sort_index() > cut_off]
    data_filter = data.loc[data[date_col].isin(relevant_years)]

    if data_filter.empty:
        return pd.DataFrame()

    numeric_cols = data_filter.select_dtypes(include=[np.number]).columns
    summary_df = data_filter.groupby(date_col)[numeric_cols].apply(lambda x: (1 + x).prod() - 1)
    summary_df = summary_df * 100
    summary_df = np.round(summary_df, 2)
    return summary_df


def compute_trailing_returns(returns_df, input_frequency="daily"):
    """Compute point-in-time trailing returns."""
    timeline_factor = _annualization_factor(input_frequency)

    def lookback_returns(df, lookback):
        if df.shape[0] < lookback:
            return pd.Series(np.nan, index=df.columns)
        return df.tail(lookback).add(1).prod().sub(1)

    results = {}

    if input_frequency == "daily":
        results["1-m"] = lookback_returns(returns_df, int(timeline_factor / 12))
        results["3-m"] = lookback_returns(returns_df, int(timeline_factor / 4))
        results["6-m"] = lookback_returns(returns_df, int(timeline_factor / 2))
    elif input_frequency == "monthly":
        results["1-m"] = lookback_returns(returns_df, 1)
        results["3-m"] = lookback_returns(returns_df, 3)
        results["6-m"] = lookback_returns(returns_df, 6)

    results["1-year"] = lookback_returns(returns_df, int(timeline_factor * 1))

    ret_3y = lookback_returns(returns_df, int(timeline_factor * 3))
    results["3-years"] = (ret_3y.add(1) ** (1/3)).sub(1) if not ret_3y.isna().all() else ret_3y

    ret_5y = lookback_returns(returns_df, int(timeline_factor * 5))
    results["5-years"] = (ret_5y.add(1) ** (1/5)).sub(1) if not ret_5y.isna().all() else ret_5y

    ret_10y = lookback_returns(returns_df, int(timeline_factor * 10))
    results["10-years"] = (ret_10y.add(1) ** (1/10)).sub(1) if not ret_10y.isna().all() else ret_10y

    summary_df = pd.DataFrame(results).T
    summary_df = summary_df * 100
    summary_df = np.round(summary_df, 2)
    return summary_df.T


def compute_monthly_returns_from_daily(returns_df, input_frequency="daily"):
    """Convert daily returns to monthly returns."""
    if input_frequency == "monthly":
        return returns_df.copy()

    monthly_df = returns_df.copy(deep=True)
    monthly_df = monthly_df.reset_index()
    date_col = monthly_df.columns[0]

    monthly_df["YM"] = pd.to_datetime(monthly_df[date_col]).dt.to_period("M")
    monthly_df = monthly_df.set_index(date_col)

    mask = monthly_df["YM"].value_counts() > 15
    mask = mask[mask].index
    monthly_df = monthly_df.loc[monthly_df["YM"].isin(mask)]

    numeric_cols = monthly_df.select_dtypes(include=[np.number]).columns
    monthly_df = monthly_df.groupby("YM")[numeric_cols].apply(lambda x: (1 + x).prod() - 1)
    monthly_df = monthly_df.to_timestamp("M")
    monthly_df.index = monthly_df.index.date
    monthly_df.index.name = "Date"

    return monthly_df


def compute_up_down_months(monthly_returns_df):
    """Compute number of up and down months."""
    up_months = (monthly_returns_df > 0).sum()
    down_months = monthly_returns_df.shape[0] - up_months

    up_down_df = pd.DataFrame(
        [up_months.values, down_months.values],
        index=["Up_Months", "Down_Months"],
        columns=monthly_returns_df.columns
    )
    return up_down_df


def compute_crisis_regime_returns(returns_df, data_start_date=None, data_end_date=None):
    """Compute returns during crisis and recovery periods."""
    results = []

    if data_start_date is None:
        data_start_date = returns_df.index.min()
    if data_end_date is None:
        data_end_date = returns_df.index.max()

    data_start_date = pd.to_datetime(data_start_date)
    data_end_date = pd.to_datetime(data_end_date)

    for regime, period in CRISIS_REGIMES.items():
        cris_start = pd.to_datetime(period["crisis_start"], format="%d-%m-%Y")
        cris_end = pd.to_datetime(period["crisis_end"], format="%d-%m-%Y")
        recov_start = pd.to_datetime(period["recovery_start"], format="%d-%m-%Y")
        recov_end = pd.to_datetime(period["recovery_end"], format="%d-%m-%Y")

        if cris_end < data_start_date or cris_start > data_end_date:
            continue

        cris_data = returns_df.loc[cris_start:cris_end]
        recov_data = returns_df.loc[recov_start:recov_end]

        if cris_data.empty and recov_data.empty:
            continue

        cris_returns = cris_data.add(1).prod().sub(1).values * 100 if not cris_data.empty else np.zeros(returns_df.shape[1])
        recov_returns = recov_data.add(1).prod().sub(1).values * 100 if not recov_data.empty else np.zeros(returns_df.shape[1])

        regime_df = pd.DataFrame(
            {regime: cris_returns, f"Recovery {regime}": recov_returns},
            index=returns_df.columns
        ).T
        regime_df.insert(0, "Start_Date", [cris_start.strftime("%d-%m-%Y"), recov_start.strftime("%d-%m-%Y")])
        regime_df.insert(1, "End_Date", [cris_end.strftime("%d-%m-%Y"), recov_end.strftime("%d-%m-%Y")])

        results.append(regime_df)

    if not results:
        return pd.DataFrame()

    crisis_df = pd.concat(results)
    crisis_df.index.name = "Crisis_Regime"
    return np.round(crisis_df, 2)


def compute_market_regime_returns(returns_df, data_start_date=None, data_end_date=None):
    """Compute returns during different market regimes (bull/bear/recovery)."""
    results = []

    if data_start_date is None:
        data_start_date = returns_df.index.min()
    if data_end_date is None:
        data_end_date = returns_df.index.max()

    data_start_date = pd.to_datetime(data_start_date)
    data_end_date = pd.to_datetime(data_end_date)

    for regime, period in MARKET_REGIMES.items():
        reg_start = pd.to_datetime(period["start"], format="%d-%m-%Y")
        reg_end = pd.to_datetime(period["end"], format="%d-%m-%Y")

        if reg_end < data_start_date or reg_start > data_end_date:
            continue

        reg_data = returns_df.loc[reg_start:reg_end]

        if reg_data.empty:
            continue

        reg_returns = reg_data.add(1).prod().sub(1).values * 100

        regime_df = pd.DataFrame({regime: reg_returns}, index=returns_df.columns).T
        regime_df.insert(0, "Start_Date", reg_start.strftime("%d-%m-%Y"))
        regime_df.insert(1, "End_Date", reg_end.strftime("%d-%m-%Y"))

        results.append(regime_df)

    if not results:
        return pd.DataFrame()

    market_df = pd.concat(results)
    market_df.index.name = "Market_Regime"
    return np.round(market_df, 2)


def compute_quarterly_alpha(daily_returns_df, trade_results):
    """Compute quarterly alpha (outperformance) for each quarter in the backtest."""
    if trade_results is None or 'quarter' not in trade_results.columns:
        return pd.DataFrame()

    quarters = sorted(trade_results['quarter'].unique())
    if len(quarters) == 0:
        return pd.DataFrame()

    def get_quarter_dates(quarter_int):
        quarter_str = str(quarter_int)
        year = int(quarter_str[:4])
        month = int(quarter_str[4:])

        if month == 2:
            start = pd.Timestamp(year=year, month=2, day=15)
            end = pd.Timestamp(year=year, month=5, day=30)
        elif month == 5:
            start = pd.Timestamp(year=year, month=5, day=31)
            end = pd.Timestamp(year=year, month=8, day=14)
        elif month == 8:
            start = pd.Timestamp(year=year, month=8, day=15)
            end = pd.Timestamp(year=year, month=11, day=14)
        elif month == 11:
            start = pd.Timestamp(year=year, month=11, day=15)
            end = pd.Timestamp(year=year + 1, month=2, day=14)
        else:
            raise ValueError(f"Invalid quarter month: {month}. Expected 2, 5, 8, or 11.")
        return start, end

    results = []
    cols = daily_returns_df.columns.tolist()
    pf_col = cols[0] if len(cols) > 0 else 'Portfolio'
    bench_col = cols[1] if len(cols) > 1 else 'Benchmark'

    for quarter in quarters:
        try:
            start_date, end_date = get_quarter_dates(quarter)
            quarter_data = daily_returns_df[
                (daily_returns_df.index >= start_date) &
                (daily_returns_df.index <= end_date)
            ]
            if len(quarter_data) == 0:
                continue

            pf_return = (1 + quarter_data[pf_col]).prod() - 1
            bench_return = (1 + quarter_data[bench_col]).prod() - 1
            outperformance = pf_return - bench_return

            results.append({
                'Quarter': quarter,
                'Portfolio_Return': pf_return,
                'Benchmark_Return': bench_return,
                'Outperformance': outperformance
            })
        except Exception as e:
            print(f"      Warning: Could not compute returns for quarter {quarter}: {e}")
            continue

    if len(results) == 0:
        return pd.DataFrame()

    quarterly_df = pd.DataFrame(results)

    total_quarters = len(quarterly_df)
    outperformance_quarters = (quarterly_df['Outperformance'] > 0).sum()
    outperformance_pct = outperformance_quarters / total_quarters if total_quarters > 0 else 0

    blank_row = pd.DataFrame([{
        'Quarter': '', 'Portfolio_Return': np.nan,
        'Benchmark_Return': np.nan, 'Outperformance': np.nan
    }])

    stats_rows = pd.DataFrame([
        {'Quarter': 'Total Quarters', 'Portfolio_Return': str(total_quarters),
         'Benchmark_Return': '', 'Outperformance': ''},
        {'Quarter': 'Outperformance Quarters', 'Portfolio_Return': str(outperformance_quarters),
         'Benchmark_Return': '', 'Outperformance': ''},
        {'Quarter': 'Outperformance Rate', 'Portfolio_Return': outperformance_pct,
         'Benchmark_Return': '', 'Outperformance': ''}
    ])

    result_df = pd.concat([quarterly_df, blank_row, stats_rows], ignore_index=True)
    return result_df


# =============================================================================
# TRADE-LEVEL ANALYSIS (from analytics.py)
# =============================================================================

def get_stock_counts_by_mcap(trade_results: pd.DataFrame) -> pd.DataFrame:
    """Get count of stocks by market cap category per quarter."""
    has_mcap_category = 'mcap_category' in trade_results.columns
    has_cat = 'cat' in trade_results.columns

    if not has_mcap_category and not has_cat:
        raise ValueError("Missing required column: need either 'mcap_category' or 'cat'")

    mcap_col = None
    if has_mcap_category:
        mcap_col = 'mcap_category'
    elif has_cat:
        cat_values = set(trade_results['cat'].dropna().unique())
        mcap_values = {'largecap', 'midcap', 'smallcap'}
        if mcap_values.intersection(cat_values):
            mcap_col = 'cat'

    if mcap_col is None:
        raise ValueError("No valid market cap category column found (largecap/midcap/smallcap)")

    if 'holding_period' not in trade_results.columns:
        raise ValueError("Missing required column: 'holding_period'")

    valid_trades = trade_results[trade_results['holding_period'].notna()].copy()
    mcap_counts = valid_trades.groupby(['quarter', mcap_col]).size().unstack(fill_value=0)

    for cat in ['largecap', 'midcap', 'smallcap']:
        if cat not in mcap_counts.columns:
            mcap_counts[cat] = 0

    mcap_counts['total_stocks'] = mcap_counts[['largecap', 'midcap', 'smallcap']].sum(axis=1)
    mcap_counts = mcap_counts[['total_stocks', 'largecap', 'midcap', 'smallcap']]
    mcap_counts = mcap_counts.reset_index()

    avg_row = pd.DataFrame([{
        'quarter': 'Average',
        'total_stocks': mcap_counts['total_stocks'].mean(),
        'largecap': mcap_counts['largecap'].mean(),
        'midcap': mcap_counts['midcap'].mean(),
        'smallcap': mcap_counts['smallcap'].mean()
    }])

    result = pd.concat([mcap_counts, avg_row], ignore_index=True)
    return result


def get_comprehensive_quarter_analysis(trade_results: pd.DataFrame) -> pd.DataFrame:
    """
    Get comprehensive analysis combining stock counts, category returns, holding periods,
    and TP/SL statistics per quarter.
    """
    has_cat_weight = 'cat_weight' in trade_results.columns
    has_stock_weight = 'stock_weight' in trade_results.columns

    if not (has_cat_weight or has_stock_weight):
        raise ValueError("Missing weight column: need either 'cat_weight' or 'stock_weight'")

    required_cols = {'quarter', 'cat', 'holding_period', 'stock_return'}
    if not required_cols.issubset(trade_results.columns):
        missing = required_cols - set(trade_results.columns)
        raise ValueError(f"Missing required columns: {missing}")

    has_tpsl = {'SL_triggered', 'TP_triggered'}.issubset(trade_results.columns)
    valid_trades = trade_results[trade_results['holding_period'].notna()].copy()
    results = []

    for quarter, group in valid_trades.groupby('quarter'):
        row = {'quarter': quarter}
        row['total_stocks'] = len(group)

        cat_counts = group['cat'].value_counts()
        for cat, count in cat_counts.items():
            row[f'{cat}_count'] = count

        row['avg_holding_period'] = round(group['holding_period'].mean(), 2)

        if has_tpsl:
            tp_count = group['TP_triggered'].sum()
            sl_count = group['SL_triggered'].sum()
            time_exit_count = len(group) - tp_count - sl_count

            row['TP_count'] = int(tp_count)
            row['SL_count'] = int(sl_count)
            row['time_exit_count'] = int(time_exit_count)

            sl_trades = group[group['SL_triggered'] == True]
            row['avg_holding_period_SL'] = round(sl_trades['holding_period'].mean(), 2) if len(sl_trades) > 0 else None

            tp_trades = group[group['TP_triggered'] == True]
            row['avg_holding_period_TP'] = round(tp_trades['holding_period'].mean(), 2) if len(tp_trades) > 0 else None

        valid_returns = group[group['stock_return'].notna()]

        for cat in group['cat'].unique():
            cat_trades = group[group['cat'] == cat]
            cat_returns = valid_returns[valid_returns['cat'] == cat]
            row[f'{cat}_avg_hp'] = round(cat_trades['holding_period'].mean(), 2)
            if len(cat_returns) > 0:
                row[f'{cat}_return'] = round(cat_returns['stock_return'].mean(), 6)
            else:
                row[f'{cat}_return'] = None

        if len(valid_returns) > 0:
            if has_cat_weight:
                cat_ret_df = valid_returns.groupby('cat').agg({
                    'stock_return': 'mean', 'cat_weight': 'first'
                })
                row['portfolio_return'] = round((cat_ret_df['stock_return'] * cat_ret_df['cat_weight']).sum(), 6)
            elif has_stock_weight:
                row['portfolio_return'] = round((valid_returns['stock_return'] * valid_returns['stock_weight']).sum(), 6)
        else:
            row['portfolio_return'] = None

        results.append(row)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('quarter').reset_index(drop=True)
    return result_df


# =============================================================================
# CHART FUNCTIONS — all return BytesIO buffers for embedding in Excel
# =============================================================================

def create_pf_vs_index_chart(comparison_df, first_quarter, last_quarter, initial_capital=INITIAL_CAPITAL):
    """
    Create Portfolio vs Index chart (2-panel: value comparison + alpha).
    Extracted from simulation.py plot_pf_vs_index.

    Returns:
    - BytesIO buffer containing the PNG image, or None if no data
    """
    if comparison_df is None or comparison_df.empty:
        return None

    df = comparison_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True)

    # Top: Portfolio Value vs Index
    ax1.plot(df['date'], df['pf_value'], color='#1f77b4', linewidth=2, label='Portfolio')
    ax1.plot(df['date'], df['index_fund_value'], color='#ff7f0e', linewidth=2, label='Index Fund')
    ax1.axhline(y=initial_capital, color='black', linestyle='--', alpha=0.5, label='Initial Capital')

    ax1.fill_between(df['date'], df['pf_value'], df['index_fund_value'],
                     where=(df['pf_value'] >= df['index_fund_value']),
                     interpolate=True, color='green', alpha=0.1, label='Outperformance')
    ax1.fill_between(df['date'], df['pf_value'], df['index_fund_value'],
                     where=(df['pf_value'] < df['index_fund_value']),
                     interpolate=True, color='red', alpha=0.1, label='Underperformance')

    ax1.set_title(f'Portfolio vs Index: {first_quarter} to {last_quarter} (Initial Capital: \u20B9{initial_capital/10000000:.0f} Cr)',
                  fontsize=14, pad=15)
    ax1.set_ylabel('Value (INR)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')

    def crores_formatter(x, pos):
        return f'{x/10000000:.1f} Cr'
    ax1.yaxis.set_major_formatter(FuncFormatter(crores_formatter))

    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    years = (end_date - start_date).days / DAYS_PER_YEAR

    final_pf_value = df['pf_value'].iloc[-1]
    final_index_value = df['index_fund_value'].iloc[-1]

    if years > 0:
        pf_cagr = ((final_pf_value / initial_capital) ** (1 / years) - 1) * 100
        index_cagr = ((final_index_value / initial_capital) ** (1 / years) - 1) * 100
    else:
        pf_cagr = 0
        index_cagr = 0

    annotation_text = (f"Portfolio CAGR: {pf_cagr:+.2f}%\n"
                       f"Index CAGR: {index_cagr:+.2f}%")
    ax1.text(0.98, 0.95, annotation_text, transform=ax1.transAxes,
             fontsize=11, fontweight='bold', verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

    # Bottom: Alpha
    ax2.fill_between(df['date'], df['alpha'], 0,
                     where=(df['alpha'] >= 0), interpolate=True, color='green', alpha=0.3)
    ax2.fill_between(df['date'], df['alpha'], 0,
                     where=(df['alpha'] < 0), interpolate=True, color='red', alpha=0.3)
    ax2.plot(df['date'], df['alpha'], color='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Alpha (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    date_range_days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    if date_range_days > 365:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    elif date_range_days > 180:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    else:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return buffer


def create_growth_of_wealth_chart(monthly_returns_df, chart_title="Growth of Rs 10,000"):
    """Create Growth of Wealth chart. Returns BytesIO buffer."""
    gofwealth = monthly_returns_df.add(1).cumprod().multiply(10000).reset_index()
    date_col = gofwealth.columns[0]

    plt.figure(figsize=(12, 6))
    colors = plt.colormaps["tab10"].colors

    for idx, col in enumerate(gofwealth.columns[1:]):
        plt.plot(gofwealth[date_col], gofwealth[col], label=col, linewidth=1, color=colors[idx % len(colors)])

    plt.ylabel("Growth of Rs 10,000", fontsize=10)
    plt.title(chart_title, fontsize=12)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('Rs. {x:,.0f}'))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    plt.close()
    buffer.seek(0)
    return buffer


def create_daily_drawdown_chart(daily_pf):
    """
    Create daily drawdown chart (2-panel: value + drawdown).
    Refactored from analytics.py plot_drawdown to return BytesIO buffer.
    """
    dd = compute_drawdown_series(daily_pf)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1], sharex=True)

    # Portfolio value
    ax1.plot(dd['date'], dd['portfolio_value'], color='#1f77b4', linewidth=1.5, label='Portfolio')
    ax1.plot(dd['date'], dd['cummax'], color='gray', linestyle='--', alpha=0.7, label='Peak')
    ax1.fill_between(dd['date'], dd['portfolio_value'], dd['cummax'], alpha=0.3, color='red')
    ax1.set_ylabel('Portfolio Value (Cr)')

    def crores_formatter(x, pos):
        return f'{x/1e7:.0f}'
    ax1.yaxis.set_major_formatter(FuncFormatter(crores_formatter))
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Portfolio Value and Drawdown Analysis', fontsize=14)

    # Drawdown
    ax2.fill_between(dd['date'], dd['drawdown_pct'], 0, color='red', alpha=0.5)
    ax2.plot(dd['date'], dd['drawdown_pct'], color='darkred', linewidth=1)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    max_dd_idx = dd['drawdown_pct'].idxmin()
    max_dd_date = dd.loc[max_dd_idx, 'date']
    max_dd_val = dd.loc[max_dd_idx, 'drawdown_pct']
    ax2.annotate(f'Max DD: {max_dd_val:.1f}%', xy=(max_dd_date, max_dd_val),
                 xytext=(10, 15), textcoords='offset points',
                 fontsize=10, color='darkred', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=1))

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return buffer


def create_monthly_returns_heatmap(daily_pf):
    """
    Create monthly returns heatmap.
    Refactored from analytics.py plot_monthly_returns_heatmap to return BytesIO buffer.
    """
    # Build monthly returns pivot
    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    monthly = df['portfolio_value'].resample('ME').last()
    monthly_returns = monthly.pct_change() * 100

    result = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    pivot = result.pivot(index='year', columns='month', values='return')

    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    pivot.columns = [month_names[m] for m in pivot.columns]

    monthly_only = pivot

    fig, ax = plt.subplots(figsize=(14, max(4, len(monthly_only) * 0.5 + 1)))

    im = ax.imshow(monthly_only.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

    ax.set_xticks(range(len(monthly_only.columns)))
    ax.set_xticklabels(monthly_only.columns)
    ax.set_yticks(range(len(monthly_only.index)))
    ax.set_yticklabels(monthly_only.index)

    for i in range(len(monthly_only.index)):
        for j in range(len(monthly_only.columns)):
            val = monthly_only.iloc[i, j]
            if pd.notna(val):
                text_color = 'white' if abs(val) > 5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=text_color, fontsize=9)

    ax.set_title('Monthly Returns (%)', fontsize=14)
    plt.colorbar(im, ax=ax, label='Return (%)')
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return buffer


def create_calendar_year_heatmap(calendar_year_df, chart_title="Calendar Year Returns Heatmap"):
    """Create Calendar Year Returns Heatmap. Returns BytesIO buffer."""
    if calendar_year_df.empty:
        return None

    n_rows, n_cols = calendar_year_df.shape
    cell_width = 0.8
    cell_height = 0.3
    fig_width = max(6, n_cols * cell_width)
    fig_height = max(4, n_rows * cell_height)

    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        calendar_year_df, annot=True, fmt=".1f", cmap="RdYlGn",
        center=0, linewidths=0.5, cbar_kws={"label": "Return (%)"}
    )

    plt.title(chart_title, fontsize=12)
    ax.set_xlabel("Strategies")
    ax.set_ylabel("Calendar Year", fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    plt.close()
    buffer.seek(0)
    return buffer


def create_correlation_heatmap(returns_df, chart_title="Correlation Matrix"):
    """Create Correlation Matrix Heatmap. Returns BytesIO buffer."""
    corr_matrix = returns_df.corr()

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, cbar_kws={"label": "Correlation"}
    )

    plt.title(chart_title, fontsize=12)
    ax.set_xlabel("Strategies")
    ax.set_ylabel("Strategies", fontsize=10)
    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    plt.close()
    buffer.seek(0)
    return buffer


def create_distribution_chart(monthly_returns_df, chart_title="Distribution of Monthly Returns"):
    """Create Bell Curve / Distribution chart. Returns BytesIO buffer."""
    long_df = monthly_returns_df.melt(var_name="Strategies", value_name="Monthly Return")

    plt.figure(figsize=(12, 6))
    colors = plt.colormaps["tab10"].colors
    assets = monthly_returns_df.columns.tolist()
    palette = {asset: colors[i % len(colors)] for i, asset in enumerate(assets)}

    sns.kdeplot(
        data=long_df, x='Monthly Return', hue='Strategies',
        palette=palette, common_norm=False, linewidth=2
    )

    plt.title(chart_title, fontsize=12)
    plt.xlabel('Monthly Return')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    plt.close()
    buffer.seek(0)
    return buffer


def create_box_plot(monthly_returns_df, chart_title="Box-Whisker Plot of Monthly Returns"):
    """Create Box-Whisker plot. Returns BytesIO buffer."""
    plt.figure(figsize=(12, 6))
    colors = plt.colormaps["tab10"].colors
    assets = monthly_returns_df.columns.tolist()
    palette = {asset: colors[i % len(colors)] for i, asset in enumerate(assets)}

    sns.boxplot(data=monthly_returns_df, palette=palette)

    plt.title(chart_title, fontsize=12)
    plt.xlabel('Asset')
    plt.ylabel('Monthly Return')

    handles = [plt.Line2D([], [], marker='s', linestyle='None', color=palette[asset], label=asset)
               for asset in assets]
    plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    plt.close()
    buffer.seek(0)
    return buffer


# =============================================================================
# MAIN REPORT GENERATION FUNCTION
# =============================================================================

def generate_backtest_report(
    daily_pf,
    trade_results,
    comparison_df=None,
    output_path=None,
    sub_periods=None,
    input_frequency="daily",
    report_title="Backtest Report",
    data_issues=None,
    first_quarter=None,
    last_quarter=None,
):
    """
    Generate the single consolidated backtest report Excel workbook.

    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - trade_results: DataFrame from trade simulation
    - comparison_df: DataFrame with portfolio vs index comparison (can be None)
    - output_path: Full path for the output Excel file
    - sub_periods: List of [start_year, end_year] pairs for sub-period analysis
    - input_frequency: 'daily' or 'monthly'
    - report_title: Title for the report
    - data_issues: DataFrame with data quality issues (can be None)
    - first_quarter: First quarter (int) for chart titles
    - last_quarter: Last quarter (int) for chart titles

    Returns:
    - Path to the generated Excel file
    """
    print("\nGenerating consolidated backtest report...")

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    # Prepare daily returns from portfolio values
    daily_pf_sorted = daily_pf.copy()
    daily_pf_sorted['date'] = pd.to_datetime(daily_pf_sorted['date'])
    daily_pf_sorted = daily_pf_sorted.sort_values('date')

    # Daily returns DataFrame (for returns-based analysis)
    returns_series = daily_pf_sorted.set_index('date')['portfolio_value'].pct_change().dropna()
    returns_series.name = 'Portfolio'

    # Build combined returns (Portfolio + Benchmark)
    if comparison_df is not None and not comparison_df.empty:
        comp = comparison_df.copy()
        comp['date'] = pd.to_datetime(comp['date'])
        comp = comp.set_index('date').sort_index()
        index_returns = comp['index_fund_value'].pct_change().dropna()
        index_returns.name = 'Benchmark'
        combined_returns = pd.concat([returns_series, index_returns], axis=1).dropna()
    else:
        combined_returns = returns_series.to_frame()

    combined_returns.index.name = 'Date'

    # Monthly returns (for charts)
    monthly_df = compute_monthly_returns_from_daily(combined_returns, input_frequency)

    # Date range
    data_start = combined_returns.index.min()
    data_end = combined_returns.index.max()

    # =========================================================================
    # COMPUTE ALL ANALYTICS
    # =========================================================================

    print("  - Computing portfolio metrics...")
    portfolio_metrics = None
    try:
        portfolio_metrics = compute_portfolio_metrics(daily_pf)
    except Exception as e:
        print(f"    Warning: Could not compute portfolio metrics: {e}")

    print("  - Computing benchmark metrics...")
    benchmark_metrics = None
    if comparison_df is not None:
        try:
            benchmark_metrics = compute_benchmark_metrics(comparison_df)
        except Exception as e:
            print(f"    Warning: Could not compute benchmark metrics: {e}")

    print("  - Computing periodic performance...")
    periodic_inception = compute_portfolio_performance(combined_returns, input_frequency)
    periodic_inception.index.name = "Since_Inception"

    sub_period_dfs = []
    if sub_periods:
        for start_y, end_y in sub_periods:
            sp_df = compute_portfolio_performance(combined_returns, input_frequency, start_y, end_y)
            if not sp_df.empty:
                sp_df.index.name = f"{start_y}_{end_y}"
                sub_period_dfs.append(sp_df)

    print("  - Computing rolling performance...")
    rolling_1y = compute_rolling_performance(combined_returns, input_frequency, 1, "yearly")
    rolling_1y.index.name = "Rolling_1Y" if not rolling_1y.empty else None
    rolling_3y = compute_rolling_performance(combined_returns, input_frequency, 3, "yearly")
    rolling_3y.index.name = "Rolling_3Y_ann" if not rolling_3y.empty else None
    rolling_5y = compute_rolling_performance(combined_returns, input_frequency, 5, "yearly")
    rolling_5y.index.name = "Rolling_5Y_ann" if not rolling_5y.empty else None

    print("  - Computing calendar year performance...")
    calendar_df = compute_calendar_year_performance(combined_returns, input_frequency)
    calendar_df.index.name = "Calendar_Year" if not calendar_df.empty else None

    print("  - Computing trailing returns...")
    trailing_df = compute_trailing_returns(combined_returns, input_frequency)
    last_date = pd.to_datetime(data_end).strftime("%Y-%m-%d")
    trailing_df.index.name = f"Trailing_Returns_{last_date}"

    print("  - Computing up/down months...")
    up_down_df = compute_up_down_months(monthly_df)

    print("  - Computing quarter analysis...")
    quarter_analysis = None
    try:
        quarter_analysis = get_comprehensive_quarter_analysis(trade_results)
    except Exception as e:
        print(f"    Warning: Could not compute quarter analysis: {e}")

    print("  - Computing regime returns...")
    crisis_df = compute_crisis_regime_returns(combined_returns, data_start, data_end)
    market_df = compute_market_regime_returns(combined_returns, data_start, data_end)

    # Quarterly alpha (needs benchmark)
    quarterly_alpha_df = None
    if comparison_df is not None and trade_results is not None:
        print("  - Computing quarterly alpha...")
        try:
            quarterly_alpha_df = compute_quarterly_alpha(combined_returns, trade_results)
        except Exception as e:
            print(f"    Warning: Could not compute quarterly alpha: {e}")

    # Stock counts by mcap (conditional)
    mcap_counts_df = None
    if trade_results is not None:
        has_mcap_category = 'mcap_category' in trade_results.columns
        has_cat_with_mcap = False
        if 'cat' in trade_results.columns:
            cat_values = set(trade_results['cat'].dropna().unique())
            mcap_values = {'largecap', 'midcap', 'smallcap'}
            has_cat_with_mcap = bool(mcap_values.intersection(cat_values))
        if has_mcap_category or has_cat_with_mcap:
            print("  - Computing stock counts by market cap...")
            try:
                mcap_counts_df = get_stock_counts_by_mcap(trade_results)
            except ValueError as e:
                print(f"    Warning: Could not compute mcap counts: {e}")

    # =========================================================================
    # CREATE CHARTS
    # =========================================================================

    print("  - Creating charts...")

    pf_vs_index_chart = None
    if comparison_df is not None:
        pf_vs_index_chart = create_pf_vs_index_chart(
            comparison_df, first_quarter, last_quarter, INITIAL_CAPITAL
        )

    gow_chart = create_growth_of_wealth_chart(monthly_df, f"Growth of Rs 10,000 - {report_title}")
    drawdown_chart = create_daily_drawdown_chart(daily_pf)
    heatmap_chart = create_monthly_returns_heatmap(daily_pf)
    cy_heatmap = create_calendar_year_heatmap(calendar_df) if not calendar_df.empty else None
    corr_heatmap = create_correlation_heatmap(combined_returns)
    bell_curve = create_distribution_chart(monthly_df)
    box_plot = create_box_plot(monthly_df)

    # =========================================================================
    # WRITE EXCEL WORKBOOK
    # =========================================================================

    print("  - Writing Excel workbook...")

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Format definitions
        format_decimal = workbook.add_format({'num_format': '0.00'})
        percent_format = workbook.add_format({'num_format': '0.0%', 'align': 'right'})

        # ----- Sheet: portfolio_metrics -----
        if portfolio_metrics is not None:
            pd.DataFrame([portfolio_metrics]).to_excel(
                writer, sheet_name="portfolio_metrics", index=False
            )

        # ----- Sheet: benchmark_metrics -----
        if benchmark_metrics is not None:
            pd.DataFrame([benchmark_metrics]).to_excel(
                writer, sheet_name="benchmark_metrics", index=False
            )

        # ----- Sheet: periodic_returns -----
        start_row = 0
        periodic_inception.reset_index().to_excel(
            writer, startrow=start_row, sheet_name="periodic_returns", index=False
        )
        for sp_df in sub_period_dfs:
            start_row += len(sp_df) + 5
            sp_df.reset_index().to_excel(
                writer, startrow=start_row, sheet_name="periodic_returns", index=False
            )

        # ----- Sheet: rolling_returns -----
        start_row = 0
        if not rolling_1y.empty:
            rolling_1y.reset_index().to_excel(
                writer, startrow=start_row, sheet_name="rolling_returns", index=False
            )
            start_row += len(rolling_1y) + 5
        if not rolling_3y.empty:
            rolling_3y.reset_index().to_excel(
                writer, startrow=start_row, sheet_name="rolling_returns", index=False
            )
            start_row += len(rolling_3y) + 5
        if not rolling_5y.empty:
            rolling_5y.reset_index().to_excel(
                writer, startrow=start_row, sheet_name="rolling_returns", index=False
            )

        # ----- Sheet: calendar_returns -----
        if not calendar_df.empty:
            calendar_df.reset_index().to_excel(
                writer, sheet_name="calendar_returns", index=False
            )

        # ----- Sheet: up_down_months -----
        up_down_df.reset_index().to_excel(
            writer, sheet_name="up_down_months", index=False
        )

        # ----- Sheet: trailing_returns -----
        trailing_df.reset_index().to_excel(
            writer, sheet_name="trailing_returns", index=False
        )

        # ----- Sheet: quarter_analysis -----
        if quarter_analysis is not None:
            quarter_analysis.to_excel(
                writer, sheet_name="quarter_analysis", index=False
            )

        # ----- Sheet: quarterly_alpha -----
        if quarterly_alpha_df is not None and not quarterly_alpha_df.empty:
            quarterly_alpha_df.to_excel(
                writer, sheet_name="quarterly_alpha", index=False
            )

        # ----- Sheet: crisis_regimes -----
        if not crisis_df.empty:
            crisis_df.reset_index().to_excel(
                writer, sheet_name="crisis_regimes", index=False
            )

        # ----- Sheet: market_regimes -----
        if not market_df.empty:
            market_df.reset_index().to_excel(
                writer, sheet_name="market_regimes", index=False
            )

        # ----- Sheet: stock_counts_by_mcap -----
        if mcap_counts_df is not None:
            mcap_counts_df.to_excel(
                writer, sheet_name="stock_counts_by_mcap", index=False
            )

        # ----- Sheet: charts (all embedded PNGs) -----
        charts_sheet = workbook.add_worksheet("charts")
        writer.sheets["charts"] = charts_sheet

        row_offset = 2

        if pf_vs_index_chart:
            charts_sheet.write(row_offset - 1, 0, "Portfolio vs Index")
            charts_sheet.insert_image(f'A{row_offset + 1}', "plot.png", {"image_data": pf_vs_index_chart})
            row_offset += 40

        charts_sheet.write(row_offset - 1, 0, "Growth of Wealth")
        charts_sheet.insert_image(f'A{row_offset + 1}', "plot.png", {"image_data": gow_chart})
        row_offset += 33

        charts_sheet.write(row_offset - 1, 0, "Daily Drawdown")
        charts_sheet.insert_image(f'A{row_offset + 1}', "plot.png", {"image_data": drawdown_chart})
        row_offset += 40

        charts_sheet.write(row_offset - 1, 0, "Monthly Returns Heatmap")
        charts_sheet.insert_image(f'A{row_offset + 1}', "plot.png", {"image_data": heatmap_chart})
        row_offset += 33

        if cy_heatmap:
            charts_sheet.write(row_offset - 1, 0, "Calendar Year Returns Heatmap")
            charts_sheet.insert_image(f'A{row_offset + 1}', "plot.png", {"image_data": cy_heatmap})
            row_offset += 22

        charts_sheet.write(row_offset - 1, 0, "Correlation Matrix")
        charts_sheet.insert_image(f'A{row_offset + 1}', "plot.png", {"image_data": corr_heatmap})
        row_offset += 22

        charts_sheet.write(row_offset - 1, 0, "Distribution of Monthly Returns")
        charts_sheet.insert_image(f'A{row_offset + 1}', "plot.png", {"image_data": bell_curve})
        row_offset += 33

        charts_sheet.write(row_offset - 1, 0, "Box-Whisker Plot")
        charts_sheet.insert_image(f'A{row_offset + 1}', "plot.png", {"image_data": box_plot})

        # ----- Sheet: trade_results (raw data) -----
        trade_results.to_excel(
            writer, sheet_name="trade_results", index=False
        )

        # ----- Sheet: daily_portfolio_values (raw data) -----
        daily_pf.to_excel(
            writer, sheet_name="daily_portfolio_values", index=False
        )

        # ----- Sheet: portfolio_vs_index (raw data, conditional) -----
        if comparison_df is not None:
            comparison_df.to_excel(
                writer, sheet_name="portfolio_vs_index", index=False
            )

        # ----- Sheet: data_quality_issues (conditional) -----
        if data_issues is not None and not data_issues.empty:
            data_issues.to_excel(
                writer, sheet_name="data_quality_issues", index=False
            )

        # ----- Apply formatting -----
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            if sheet_name in ["calendar_returns"]:
                worksheet.set_column('A:Z', 15, format_decimal)
            elif sheet_name in ["trailing_returns"]:
                worksheet.set_column('A:A', 25)
                worksheet.set_column('B:Z', 12, format_decimal)
            elif sheet_name in ["crisis_regimes", "market_regimes"]:
                worksheet.set_column('A:A', 25)
            elif sheet_name == "stock_counts_by_mcap":
                worksheet.set_column('A:A', 12)
                worksheet.set_column('B:E', 15, format_decimal)
            elif sheet_name == "quarterly_alpha":
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:D', 18, percent_format)
            elif sheet_name == "portfolio_metrics":
                worksheet.set_column('A:Z', 18)
            elif sheet_name == "benchmark_metrics":
                worksheet.set_column('A:Z', 22)
            elif sheet_name not in ["charts", "trade_results",
                                     "daily_portfolio_values", "portfolio_vs_index",
                                     "data_quality_issues"]:
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:Z', 12, format_decimal)

    print(f"  - Report saved to: {output_path}")
    return output_path
