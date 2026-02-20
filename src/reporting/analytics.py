"""Returns-based performance analytics and trade-level analysis."""

from typing import Optional

import pandas as pd
import numpy as np

from config.regime_dates import CRISIS_REGIMES, MARKET_REGIMES
from utils.quarter import get_quarter_dates
from utils.logging_config import get_logger

logger = get_logger(__name__)

from ._helpers import (
    _annualization_factor,
    _check_start_end_year,
    _trim_df_year,
    _cumulative_returns,
    _annualized_returns,
    _annualized_sd,
    _return_risk_ratio,
    _max_drawdown,
    _downside_dev,
    _scale_factor_roll,
    _compute_probability_buckets,
)


def compute_portfolio_performance(returns_df: pd.DataFrame, input_frequency: str = "daily", start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.DataFrame:
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


def compute_rolling_performance(returns_df: pd.DataFrame, input_frequency: str = "daily", roll_period: int = 1, roll_type: str = "yearly") -> pd.DataFrame:
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


def compute_calendar_year_performance(returns_df: pd.DataFrame, input_frequency: str = "daily") -> pd.DataFrame:
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


def compute_trailing_returns(returns_df: pd.DataFrame, input_frequency: str = "daily") -> pd.DataFrame:
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


def compute_monthly_returns_from_daily(returns_df: pd.DataFrame, input_frequency: str = "daily") -> pd.DataFrame:
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


def compute_up_down_months(monthly_returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute number of up and down months."""
    up_months = (monthly_returns_df > 0).sum()
    down_months = monthly_returns_df.shape[0] - up_months

    up_down_df = pd.DataFrame(
        [up_months.values, down_months.values],
        index=["Up_Months", "Down_Months"],
        columns=monthly_returns_df.columns
    )
    return up_down_df


def compute_crisis_regime_returns(returns_df: pd.DataFrame, data_start_date: Optional[pd.Timestamp] = None, data_end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
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


def compute_market_regime_returns(returns_df: pd.DataFrame, data_start_date: Optional[pd.Timestamp] = None, data_end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
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


def compute_quarterly_alpha(daily_returns_df: pd.DataFrame, trade_results: pd.DataFrame) -> pd.DataFrame:
    """Compute quarterly alpha (outperformance) for each quarter in the backtest."""
    if trade_results is None or 'quarter' not in trade_results.columns:
        return pd.DataFrame()

    quarters = sorted(trade_results['quarter'].unique())
    if len(quarters) == 0:
        return pd.DataFrame()

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
        except (ValueError, KeyError) as e:
            logger.warning("Could not compute returns for quarter %s: %s", quarter, e)
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
