"""
Summary Statistics Module for Portfolio Analysis

This module contains functions extracted from the MO Quant Strategies analysis notebook.
It provides comprehensive portfolio performance analysis including:
- Periodic performance (since inception + sub-periods)
- Rolling returns with probability distributions
- Calendar year performance
- Trailing returns (point-in-time)
- Crisis and market regime analysis
- Charts (Growth of Wealth, Drawdown, Distribution, etc.)

Functions expect daily returns as input (DataFrame with Date index, return columns).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
import io

from backtest.regime_dates import CRISIS_REGIMES, MARKET_REGIMES
from config.defaults import ANNUALIZATION_FACTORS, SCALE_FACTORS


# =============================================================================
# HELPER FUNCTIONS
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
    """
    Get annualization factor based on data timeline.
        daily -> 252
        monthly -> 12
        yearly -> 1
    """
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
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def compute_portfolio_performance(returns_df, input_frequency="daily", start_year=None, end_year=None):
    """
    Compute comprehensive portfolio performance metrics.
    
    Parameters:
    - returns_df: DataFrame with Date index and return columns (daily returns as decimals)
    - input_frequency: 'daily', 'monthly', or 'yearly'
    - start_year: Optional start year to trim data
    - end_year: Optional end year to trim data
    
    Returns:
    - DataFrame with metrics: G-Rs100, AReturns, ARisk, Sharpe, DDown, Sortino,
      Skewness, Kurtosis, %Up_Periods, %Down_Periods
    """
    input_frequency = input_frequency.strip().lower()
    if input_frequency not in ["daily", "monthly", "yearly"]:
        raise ValueError("data_timeline must be 'daily', 'monthly', or 'yearly'")
    
    annual_factor = _annualization_factor(input_frequency)
    
    # Trim by year if specified
    if start_year is not None and end_year is not None:
        if _check_start_end_year(returns_df, start_year, end_year):
            returns_df = _trim_df_year(returns_df, start_year, end_year)
        else:
            # Return empty dataframe if years not in data
            return pd.DataFrame()
    
    if returns_df.empty:
        return pd.DataFrame()
    
    # Cumulative Returns
    cum_ret = _cumulative_returns(returns_df)
    
    # Annual Returns
    ann_ret = _annualized_returns(cum_ret, annual_factor, returns_df.shape[0])
    
    # Annual Risk
    ann_risk = _annualized_sd(returns_df, annual_factor)
    
    # Annual Sharpe
    ann_sharpe = _return_risk_ratio(ann_ret, ann_risk)
    
    # Annual Downside Risk
    downside_risk = returns_df.apply(lambda x: _downside_dev(x, annual_factor)).values
    
    # Annual Sortino
    ann_sortino = _return_risk_ratio(ann_ret, downside_risk)
    
    # Maximum Drawdown
    period_drawdown = _max_drawdown(returns_df)
    
    # Skewness and Kurtosis
    skewness = returns_df.skew().values
    kurtosis = returns_df.kurt().values
    
    # Up and Down Days/Months
    no_of_up_periods = (returns_df > 0).sum().values / returns_df.shape[0]
    no_of_down_periods = 1 - no_of_up_periods
    
    # Scale for display
    cum_ret_scaled = (cum_ret * 100) + 100  # Growth of Rs 100
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
    """
    Compute rolling returns statistics with probability distributions.
    
    Parameters:
    - returns_df: DataFrame with Date index and return columns
    - input_frequency: 'daily', 'monthly', or 'yearly'
    - roll_period: Number of periods for rolling window (1-10 for yearly)
    - roll_type: 'daily', 'monthly', or 'yearly'
    
    Returns:
    - DataFrame with rolling stats: count, mean, min, 25%, 50%, 75%, max,
      and probability buckets (<0%P, 0-10%P, 10-20%P, >20%P)
    """
    input_frequency = input_frequency.strip().lower()
    roll_type = roll_type.strip().lower()
    
    # Get scaling factor
    scaling_factor = _scale_factor_roll(input_frequency, roll_type)
    roll_period_scaled = roll_period * scaling_factor
    
    # Get annualization factor
    timeline_factor = _annualization_factor(input_frequency)
    annualization_factor = timeline_factor / roll_period_scaled
    
    # Check data size
    if returns_df.shape[0] < roll_period_scaled:
        return pd.DataFrame()
    
    # Compute Rolling Returns
    rolling_returns = returns_df.add(1).rolling(roll_period_scaled).apply(np.prod).dropna().sub(1)
    ann_rolling_returns = rolling_returns.apply(lambda x: (1 + x) ** annualization_factor).sub(1)
    
    # Compute Stats
    summary_df = rolling_returns.describe()
    summary_df = summary_df.drop("std", axis=0, errors='ignore')
    
    # Annualize Rolling Returns Stats
    summary_df.iloc[1:, :] = summary_df.iloc[1:, :].apply(lambda x: (1 + x) ** annualization_factor).sub(1)
    
    # Compute Probability buckets
    summary_df.loc["<0%P"] = _compute_probability_buckets(ann_rolling_returns, upper_bound=0)
    summary_df.loc["0-10%P"] = _compute_probability_buckets(ann_rolling_returns, lower_bound=0, upper_bound=0.1)
    summary_df.loc["10-20%P"] = _compute_probability_buckets(ann_rolling_returns, lower_bound=0.1, upper_bound=0.2)
    summary_df.loc[">20%P"] = _compute_probability_buckets(ann_rolling_returns, lower_bound=0.2)
    
    # Scale to percentage
    summary_df.iloc[1:, :] = summary_df.iloc[1:, :] * 100
    summary_df = np.round(summary_df, 2)
    
    return summary_df.transpose()


def compute_calendar_year_performance(returns_df, input_frequency="daily"):
    """
    Compute calendar year returns.
    
    Parameters:
    - returns_df: DataFrame with Date index and return columns
    - input_frequency: 'daily' or 'monthly'
    
    Returns:
    - DataFrame with calendar year as index, strategies as columns, returns as values (%)
    """
    input_frequency = input_frequency.strip().lower()
    if input_frequency not in ["daily", "monthly"]:
        raise ValueError("data_df_timeline must be 'daily' or 'monthly'")
    
    date_col = returns_df.index.name or 'Date'
    data = returns_df.reset_index()
    
    # Convert to Year period
    if isinstance(data[date_col].dtype, pd.core.dtypes.dtypes.PeriodDtype):
        data[date_col] = data[date_col].dt.to_timestamp().dt.to_period("Y")
    else:
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce').dt.to_period("Y")
    
    # Filter out incomplete years
    cut_off = 230 if input_frequency == "daily" else 11
    relevant_years = data[date_col].unique()[data[date_col].value_counts().sort_index() > cut_off]
    data_filter = data.loc[data[date_col].isin(relevant_years)]
    
    if data_filter.empty:
        return pd.DataFrame()
    
    # Compute Calendar Year Returns
    numeric_cols = data_filter.select_dtypes(include=[np.number]).columns
    summary_df = data_filter.groupby(date_col)[numeric_cols].apply(lambda x: (1 + x).prod() - 1)
    
    # Scale to percentage
    summary_df = summary_df * 100
    summary_df = np.round(summary_df, 2)
    
    return summary_df


def compute_trailing_returns(returns_df, input_frequency="daily"):
    """
    Compute point-in-time trailing returns.
    
    Parameters:
    - returns_df: DataFrame with Date index and return columns
    - input_frequency: 'daily', 'monthly', or 'yearly'
    
    Returns:
    - DataFrame with trailing returns: 1m, 3m, 6m, 1-year, 3-years, 5-years, 10-years
    """
    timeline_factor = _annualization_factor(input_frequency)
    
    def lookback_returns(df, lookback):
        if df.shape[0] < lookback:
            return pd.Series(np.nan, index=df.columns)
        return df.tail(lookback).add(1).prod().sub(1)
    
    results = {}
    
    # Monthly returns (for daily data)
    if input_frequency == "daily":
        results["1-m"] = lookback_returns(returns_df, int(timeline_factor / 12))
        results["3-m"] = lookback_returns(returns_df, int(timeline_factor / 4))
        results["6-m"] = lookback_returns(returns_df, int(timeline_factor / 2))
    elif input_frequency == "monthly":
        results["1-m"] = lookback_returns(returns_df, 1)
        results["3-m"] = lookback_returns(returns_df, 3)
        results["6-m"] = lookback_returns(returns_df, 6)
    
    # Yearly returns
    results["1-year"] = lookback_returns(returns_df, int(timeline_factor * 1))
    
    # Multi-year (annualized)
    ret_3y = lookback_returns(returns_df, int(timeline_factor * 3))
    results["3-years"] = (ret_3y.add(1) ** (1/3)).sub(1) if not ret_3y.isna().all() else ret_3y
    
    ret_5y = lookback_returns(returns_df, int(timeline_factor * 5))
    results["5-years"] = (ret_5y.add(1) ** (1/5)).sub(1) if not ret_5y.isna().all() else ret_5y
    
    ret_10y = lookback_returns(returns_df, int(timeline_factor * 10))
    results["10-years"] = (ret_10y.add(1) ** (1/10)).sub(1) if not ret_10y.isna().all() else ret_10y
    
    summary_df = pd.DataFrame(results).T
    summary_df = summary_df * 100
    summary_df = np.round(summary_df, 2)
    
    # Transpose so strategies are rows
    return summary_df.T


def compute_monthly_returns(returns_df, input_frequency="daily"):
    """
    Convert daily returns to monthly returns if needed.
    
    Parameters:
    - returns_df: DataFrame with Date index and return columns
    - input_frequency: 'daily' or 'monthly'
    
    Returns:
    - DataFrame with monthly returns
    """
    if input_frequency == "monthly":
        return returns_df.copy()
    
    monthly_df = returns_df.copy(deep=True)
    monthly_df = monthly_df.reset_index()
    date_col = monthly_df.columns[0]
    
    monthly_df["YM"] = pd.to_datetime(monthly_df[date_col]).dt.to_period("M")
    monthly_df = monthly_df.set_index(date_col)
    
    # Filter out months with less than 15 days
    mask = monthly_df["YM"].value_counts() > 15
    mask = mask[mask].index
    monthly_df = monthly_df.loc[monthly_df["YM"].isin(mask)]
    
    # Compute monthly returns
    numeric_cols = monthly_df.select_dtypes(include=[np.number]).columns
    monthly_df = monthly_df.groupby("YM")[numeric_cols].apply(lambda x: (1 + x).prod() - 1)
    monthly_df = monthly_df.to_timestamp("M")
    monthly_df.index = monthly_df.index.date
    monthly_df.index.name = "Date"
    
    return monthly_df


def compute_up_down_months(monthly_returns_df):
    """
    Compute number of up and down months.
    
    Parameters:
    - monthly_returns_df: DataFrame with monthly returns
    
    Returns:
    - DataFrame with Up_Months and Down_Months counts
    """
    up_months = (monthly_returns_df > 0).sum()
    down_months = monthly_returns_df.shape[0] - up_months
    
    up_down_df = pd.DataFrame(
        [up_months.values, down_months.values],
        index=["Up_Months", "Down_Months"],
        columns=monthly_returns_df.columns
    )
    
    return up_down_df


def compute_crisis_regime_returns(returns_df, data_start_date=None, data_end_date=None):
    """
    Compute returns during crisis and recovery periods.
    
    Parameters:
    - returns_df: DataFrame with Date index and return columns
    - data_start_date: Start date of the backtest period (for filtering regimes)
    - data_end_date: End date of the backtest period (for filtering regimes)
    
    Returns:
    - DataFrame with crisis and recovery returns, or empty DataFrame if no regimes in range
    """
    results = []
    
    # Determine data date range
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
        
        # Skip if regime is outside data range
        if cris_end < data_start_date or cris_start > data_end_date:
            continue
        
        # Compute returns (handle partial overlap)
        cris_data = returns_df.loc[cris_start:cris_end]
        recov_data = returns_df.loc[recov_start:recov_end]
        
        if cris_data.empty and recov_data.empty:
            continue
        
        cris_returns = cris_data.add(1).prod().sub(1).values * 100 if not cris_data.empty else np.zeros(returns_df.shape[1])
        recov_returns = recov_data.add(1).prod().sub(1).values * 100 if not recov_data.empty else np.zeros(returns_df.shape[1])
        
        # Build dataframe rows
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
    """
    Compute returns during different market regimes (bull/bear/recovery).
    
    Parameters:
    - returns_df: DataFrame with Date index and return columns
    - data_start_date: Start date of the backtest period (for filtering regimes)
    - data_end_date: End date of the backtest period (for filtering regimes)
    
    Returns:
    - DataFrame with market regime returns, or empty DataFrame if no regimes in range
    """
    results = []
    
    # Determine data date range
    if data_start_date is None:
        data_start_date = returns_df.index.min()
    if data_end_date is None:
        data_end_date = returns_df.index.max()
    
    data_start_date = pd.to_datetime(data_start_date)
    data_end_date = pd.to_datetime(data_end_date)
    
    for regime, period in MARKET_REGIMES.items():
        reg_start = pd.to_datetime(period["start"], format="%d-%m-%Y")
        reg_end = pd.to_datetime(period["end"], format="%d-%m-%Y")
        
        # Skip if regime is outside data range
        if reg_end < data_start_date or reg_start > data_end_date:
            continue
        
        # Compute returns (handle partial overlap)
        reg_data = returns_df.loc[reg_start:reg_end]
        
        if reg_data.empty:
            continue
        
        reg_returns = reg_data.add(1).prod().sub(1).values * 100
        
        # Build dataframe row
        regime_df = pd.DataFrame({regime: reg_returns}, index=returns_df.columns).T
        regime_df.insert(0, "Start_Date", reg_start.strftime("%d-%m-%Y"))
        regime_df.insert(1, "End_Date", reg_end.strftime("%d-%m-%Y"))
        
        results.append(regime_df)
    
    if not results:
        return pd.DataFrame()
    
    market_df = pd.concat(results)
    market_df.index.name = "Market_Regime"
    return np.round(market_df, 2)


# =============================================================================
# CHARTING FUNCTIONS
# =============================================================================

def create_growth_of_wealth_chart(monthly_returns_df, chart_title="Growth of Rs 10,000"):
    """
    Create Growth of Wealth chart.
    
    Parameters:
    - monthly_returns_df: DataFrame with monthly returns
    - chart_title: Title for the chart
    
    Returns:
    - BytesIO buffer containing the PNG image
    """
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


def create_drawdown_chart(monthly_returns_df, chart_title="Monthly Drawdowns"):
    """
    Create Drawdown chart.
    
    Parameters:
    - monthly_returns_df: DataFrame with monthly returns
    - chart_title: Title for the chart
    
    Returns:
    - BytesIO buffer containing the PNG image
    """
    drawdown_df = (np.log(1 + monthly_returns_df).cumsum()) - (np.log(1 + monthly_returns_df).cumsum().cummax())
    drawdown_df = (np.exp(drawdown_df) - 1)
    drawdown_df = drawdown_df.reset_index()
    date_col = drawdown_df.columns[0]
    
    plt.figure(figsize=(12, 6))
    colors = plt.colormaps["tab10"].colors
    
    for idx, col in enumerate(drawdown_df.columns[1:]):
        plt.plot(drawdown_df[date_col], drawdown_df[col] * 100, label=col, linewidth=0.75, color=colors[idx % len(colors)])
    
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.ylabel("Drawdown (%)", fontsize=10)
    plt.title(chart_title, fontsize=12)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}%'))
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


def create_calendar_year_heatmap(calendar_year_df, chart_title="Calendar Year Returns Heatmap"):
    """
    Create Calendar Year Returns Heatmap.
    
    Parameters:
    - calendar_year_df: DataFrame with calendar year returns
    - chart_title: Title for the chart
    
    Returns:
    - BytesIO buffer containing the PNG image
    """
    if calendar_year_df.empty:
        return None
    
    n_rows, n_cols = calendar_year_df.shape
    cell_width = 0.8
    cell_height = 0.3
    
    fig_width = max(6, n_cols * cell_width)
    fig_height = max(4, n_rows * cell_height)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    ax = sns.heatmap(
        calendar_year_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Return (%)"}
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
    """
    Create Correlation Matrix Heatmap.
    
    Parameters:
    - returns_df: DataFrame with returns
    - chart_title: Title for the chart
    
    Returns:
    - BytesIO buffer containing the PNG image
    """
    corr_matrix = returns_df.corr()
    
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"}
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
    """
    Create Bell Curve / Distribution chart.
    
    Parameters:
    - monthly_returns_df: DataFrame with monthly returns
    - chart_title: Title for the chart
    
    Returns:
    - BytesIO buffer containing the PNG image
    """
    long_df = monthly_returns_df.melt(var_name="Strategies", value_name="Monthly Return")
    
    plt.figure(figsize=(12, 6))
    
    colors = plt.colormaps["tab10"].colors
    assets = monthly_returns_df.columns.tolist()
    palette = {asset: colors[i % len(colors)] for i, asset in enumerate(assets)}
    
    sns.kdeplot(
        data=long_df,
        x='Monthly Return',
        hue='Strategies',
        palette=palette,
        common_norm=False,
        linewidth=2
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
    """
    Create Box-Whisker plot.
    
    Parameters:
    - monthly_returns_df: DataFrame with monthly returns
    - chart_title: Title for the chart
    
    Returns:
    - BytesIO buffer containing the PNG image
    """
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


def compute_quarterly_alpha(daily_returns_df, trade_results):
    """
    Compute quarterly alpha (outperformance) for each quarter in the backtest.
    
    Parameters:
    - daily_returns_df: DataFrame with Date index and return columns (Portfolio, Benchmark)
    - trade_results: DataFrame with 'quarter' column to identify which quarters were traded
    
    Returns:
    - DataFrame with columns: Quarter, Portfolio_Return, Benchmark_Return, Outperformance
      Plus a summary row with total quarters and outperformance count
    """
    if trade_results is None or 'quarter' not in trade_results.columns:
        return pd.DataFrame()
    
    # Get unique quarters from trade_results
    quarters = sorted(trade_results['quarter'].unique())
    
    if len(quarters) == 0:
        return pd.DataFrame()
    
    # Helper function to get date range for a quarter
    def get_quarter_dates(quarter_int):
        """
        Convert quarter integer (e.g., 202202) to date range.
        Returns (start_date, end_date) tuple.
        """
        quarter_str = str(quarter_int)
        year = int(quarter_str[:4])
        month = int(quarter_str[4:])
        
        if month == 2:  # Feb quarter: Feb 15 to May 30
            start = pd.Timestamp(year=year, month=2, day=15)
            end = pd.Timestamp(year=year, month=5, day=30)
        elif month == 5:  # May quarter: May 31 to Aug 14
            start = pd.Timestamp(year=year, month=5, day=31)
            end = pd.Timestamp(year=year, month=8, day=14)
        elif month == 8:  # Aug quarter: Aug 15 to Nov 14
            start = pd.Timestamp(year=year, month=8, day=15)
            end = pd.Timestamp(year=year, month=11, day=14)
        elif month == 11:  # Nov quarter: Nov 15 to Feb 14 (next year)
            start = pd.Timestamp(year=year, month=11, day=15)
            end = pd.Timestamp(year=year + 1, month=2, day=14)
        else:
            raise ValueError(f"Invalid quarter month: {month}. Expected 2, 5, 8, or 11.")
        
        return start, end
    
    results = []
    
    # Determine column names for Portfolio and Benchmark
    cols = daily_returns_df.columns.tolist()
    pf_col = cols[0] if len(cols) > 0 else 'Portfolio'
    bench_col = cols[1] if len(cols) > 1 else 'Benchmark'
    
    for quarter in quarters:
        try:
            start_date, end_date = get_quarter_dates(quarter)
            
            # Filter daily returns for this quarter
            quarter_data = daily_returns_df[
                (daily_returns_df.index >= start_date) & 
                (daily_returns_df.index <= end_date)
            ]
            
            if len(quarter_data) == 0:
                # No data for this quarter
                continue
            
            # Calculate cumulative returns for the quarter
            # Using (1 + r1) * (1 + r2) * ... - 1 formula
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
    
    # Create main DataFrame
    quarterly_df = pd.DataFrame(results)
    
    # Add summary statistics
    total_quarters = len(quarterly_df)
    outperformance_quarters = (quarterly_df['Outperformance'] > 0).sum()
    outperformance_pct = outperformance_quarters / total_quarters if total_quarters > 0 else 0
    
    # Add blank row then summary stats
    blank_row = pd.DataFrame([{
        'Quarter': '',
        'Portfolio_Return': np.nan,
        'Benchmark_Return': np.nan,
        'Outperformance': np.nan
    }])
    
    # Create summary as text strings to avoid percentage formatting issues
    stats_rows = pd.DataFrame([
        {
            'Quarter': 'Total Quarters',
            'Portfolio_Return': str(total_quarters),  # Convert to string
            'Benchmark_Return': '',
            'Outperformance': ''
        },
        {
            'Quarter': 'Outperformance Quarters',
            'Portfolio_Return': str(outperformance_quarters),  # Convert to string
            'Benchmark_Return': '',
            'Outperformance': ''
        },
        {
            'Quarter': 'Outperformance Rate',
            'Portfolio_Return': outperformance_pct,  # Keep as number for % formatting
            'Benchmark_Return': '',
            'Outperformance': ''
        }
    ])
    
    # Combine all rows
    result_df = pd.concat([quarterly_df, blank_row, stats_rows], ignore_index=True)
    
    return result_df


# =============================================================================
# MAIN REPORT GENERATION FUNCTION
# =============================================================================

def generate_mo_report(
    daily_returns_df,
    output_path,
    sub_periods=None,
    input_frequency="daily",
    report_title="Portfolio Analysis",
    trade_results=None
):
    """
    Generate comprehensive Excel report with all analysis and embedded charts.
    
    Parameters:
    - daily_returns_df: DataFrame with Date index and return columns (Portfolio, Benchmark, etc.)
    - output_path: Path to save the Excel file
    - sub_periods: List of [start_year, end_year] pairs for sub-period analysis
    - input_frequency: 'daily' or 'monthly'
    - report_title: Title for the report
    - trade_results: Optional DataFrame from trade_results.csv for additional analysis
    
    Returns:
    - Path to the generated Excel file
    """
    print("  - Generating detailed analysis report...")
    
    # Ensure Date index
    if daily_returns_df.index.name is None:
        daily_returns_df.index.name = 'Date'
    
    # Convert daily to monthly for certain analyses
    monthly_df = compute_monthly_returns(daily_returns_df, input_frequency)
    
    # Get date range
    data_start = daily_returns_df.index.min()
    data_end = daily_returns_df.index.max()
    
    # Compute all analyses
    print("    - Computing periodic performance...")
    periodic_inception = compute_portfolio_performance(daily_returns_df, input_frequency)
    periodic_inception.index.name = "Since_Inception"
    
    # Sub-period analyses
    sub_period_dfs = []
    if sub_periods:
        for start_y, end_y in sub_periods:
            sp_df = compute_portfolio_performance(daily_returns_df, input_frequency, start_y, end_y)
            if not sp_df.empty:
                sp_df.index.name = f"{start_y}_{end_y}"
                sub_period_dfs.append(sp_df)
    
    print("    - Computing rolling performance...")
    rolling_1y = compute_rolling_performance(daily_returns_df, input_frequency, 1, "yearly")
    rolling_1y.index.name = "Rolling_1Y" if not rolling_1y.empty else None
    
    rolling_3y = compute_rolling_performance(daily_returns_df, input_frequency, 3, "yearly")
    rolling_3y.index.name = "Rolling_3Y_ann" if not rolling_3y.empty else None
    
    rolling_5y = compute_rolling_performance(daily_returns_df, input_frequency, 5, "yearly")
    rolling_5y.index.name = "Rolling_5Y_ann" if not rolling_5y.empty else None
    
    print("    - Computing calendar year performance...")
    calendar_df = compute_calendar_year_performance(daily_returns_df, input_frequency)
    calendar_df.index.name = "Calendar_Year" if not calendar_df.empty else None
    
    print("    - Computing trailing returns...")
    trailing_df = compute_trailing_returns(daily_returns_df, input_frequency)
    last_date = pd.to_datetime(data_end).strftime("%Y-%m-%d")
    trailing_df.index.name = f"Trailing_Returns_{last_date}"
    
    print("    - Computing up/down months...")
    up_down_df = compute_up_down_months(monthly_df)
    
    print("    - Computing crisis regime returns...")
    crisis_df = compute_crisis_regime_returns(daily_returns_df, data_start, data_end)
    
    print("    - Computing market regime returns...")
    market_df = compute_market_regime_returns(daily_returns_df, data_start, data_end)
    
    # Compute mcap stock counts if trade_results provided and has mcap information
    mcap_counts_df = None
    if trade_results is not None:
        # Check if we have mcap_category column or cat column with mcap values
        has_mcap_category = 'mcap_category' in trade_results.columns
        has_cat_with_mcap = False
        if 'cat' in trade_results.columns:
            cat_values = set(trade_results['cat'].dropna().unique())
            mcap_values = {'largecap', 'midcap', 'smallcap'}
            has_cat_with_mcap = bool(mcap_values.intersection(cat_values))
        
        if has_mcap_category or has_cat_with_mcap:
            print("    - Computing stock counts by market cap...")
            from backtest.analytics import get_stock_counts_by_mcap
            try:
                mcap_counts_df = get_stock_counts_by_mcap(trade_results)
            except ValueError as e:
                print(f"      Warning: Could not compute mcap counts: {e}")
    
    # Compute quarterly alpha if trade_results provided
    quarterly_alpha_df = None
    if trade_results is not None:
        print("    - Computing quarterly alpha...")
        try:
            quarterly_alpha_df = compute_quarterly_alpha(daily_returns_df, trade_results)
        except Exception as e:
            print(f"      Warning: Could not compute quarterly alpha: {e}")
    
    # Create charts
    print("    - Creating charts...")
    gow_chart = create_growth_of_wealth_chart(monthly_df, f"Growth of Rs 10,000 - {report_title}")
    drawdown_chart = create_drawdown_chart(monthly_df, f"Monthly Drawdowns - {report_title}")
    cy_heatmap = create_calendar_year_heatmap(calendar_df) if not calendar_df.empty else None
    corr_heatmap = create_correlation_heatmap(daily_returns_df)
    bell_curve = create_distribution_chart(monthly_df)
    box_plot = create_box_plot(monthly_df)
    
    # Write to Excel
    print("    - Writing Excel file...")
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Format definitions
        format_decimal = workbook.add_format({'num_format': '0.00'})
        percent_format = workbook.add_format({'num_format': '0.0%', 'align': 'right'})
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
        
        # Sheet 1: Periodic Returns
        start_row = 0
        periodic_inception.reset_index().to_excel(writer, startrow=start_row, sheet_name="periodic_returns", index=False)
        
        for sp_df in sub_period_dfs:
            start_row += len(sp_df) + 5
            sp_df.reset_index().to_excel(writer, startrow=start_row, sheet_name="periodic_returns", index=False)
        
        # Sheet 2: Rolling Returns
        start_row = 0
        if not rolling_1y.empty:
            rolling_1y.reset_index().to_excel(writer, startrow=start_row, sheet_name="rolling_returns", index=False)
            start_row += len(rolling_1y) + 5
        
        if not rolling_3y.empty:
            rolling_3y.reset_index().to_excel(writer, startrow=start_row, sheet_name="rolling_returns", index=False)
            start_row += len(rolling_3y) + 5
        
        if not rolling_5y.empty:
            rolling_5y.reset_index().to_excel(writer, startrow=start_row, sheet_name="rolling_returns", index=False)
        
        # Sheet 3: Calendar Returns
        if not calendar_df.empty:
            calendar_df.reset_index().to_excel(writer, sheet_name="calendar_returns", index=False)
        
        # Sheet 4: Monthly Returns
        monthly_df.reset_index().to_excel(writer, sheet_name="monthly_returns", index=False)
        
        # Sheet 5: Up/Down Months
        up_down_df.reset_index().to_excel(writer, sheet_name="up_down_months", index=False)
        
        # Sheet 6: Trailing Returns
        trailing_df.reset_index().to_excel(writer, sheet_name="trailing_returns", index=False)
        
        # Sheet 7: Crisis Regimes
        if not crisis_df.empty:
            crisis_df.reset_index().to_excel(writer, sheet_name="crisis_regimes", index=False)
        
        # Sheet 8: Market Regimes
        if not market_df.empty:
            market_df.reset_index().to_excel(writer, sheet_name="market_regimes", index=False)
        
        # Sheet 9: Stock Counts by Mcap (if available)
        if mcap_counts_df is not None:
            mcap_counts_df.to_excel(writer, sheet_name="stock_counts_by_mcap", index=False)
        
        # Sheet 10: Quarterly Alpha (if available)
        if quarterly_alpha_df is not None and not quarterly_alpha_df.empty:
            quarterly_alpha_df.to_excel(writer, sheet_name="quarterly_alpha", index=False)
        
        # Sheet 11: Charts - Distribution and Box Plot
        charts_sheet_01 = workbook.add_worksheet("charts_01")
        writer.sheets["charts_01"] = charts_sheet_01
        
        charts_sheet_01.insert_image('A2', "plot.png", {"image_data": bell_curve})
        charts_sheet_01.insert_image('A35', "plot.png", {"image_data": box_plot})
        
        # Sheet 12: Charts - Growth, Drawdown, Calendar Year Heatmap, Correlation
        charts_sheet_02 = workbook.add_worksheet("charts_02")
        writer.sheets["charts_02"] = charts_sheet_02
        
        row_offset = 2
        charts_sheet_02.insert_image(f'A{row_offset}', "plot.png", {"image_data": gow_chart})
        row_offset += 33
        charts_sheet_02.insert_image(f'A{row_offset}', "plot.png", {"image_data": drawdown_chart})
        row_offset += 33
        if cy_heatmap:
            charts_sheet_02.insert_image(f'A{row_offset}', "plot.png", {"image_data": cy_heatmap})
            row_offset += 22
        charts_sheet_02.insert_image(f'A{row_offset}', "plot.png", {"image_data": corr_heatmap})
        
        # Apply formatting to sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            if sheet_name in ["monthly_returns"]:
                worksheet.set_column('A:A', 20, date_format)
                worksheet.set_column('B:Z', 15, percent_format)
            elif sheet_name in ["calendar_returns"]:
                worksheet.set_column('A:Z', 15, format_decimal)
            elif sheet_name in ["trailing_returns"]:
                worksheet.set_column('A:A', 25)
                worksheet.set_column('B:Z', 12, format_decimal)
            elif sheet_name in ["crisis_regimes", "market_regimes"]:
                worksheet.set_column('A:A', 25)
            elif sheet_name == "stock_counts_by_mcap":
                worksheet.set_column('A:A', 12)  # quarter column
                worksheet.set_column('B:E', 15, format_decimal)  # stock count columns
                worksheet.set_column('B:C', 15)
                worksheet.set_column('D:Z', 12, format_decimal)
            elif sheet_name == "quarterly_alpha":
                worksheet.set_column('A:A', 20)  # quarter column
                worksheet.set_column('B:D', 18, percent_format)  # return columns as percentages
            elif sheet_name != "charts":
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:Z', 12, format_decimal)
    
    print(f"  - Detailed report saved to: {output_path}")
    return output_path
