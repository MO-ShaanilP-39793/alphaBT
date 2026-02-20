"""Private utility functions for reporting."""

import pandas as pd
import numpy as np

from config.defaults import ANNUALIZATION_FACTORS, SCALE_FACTORS


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
