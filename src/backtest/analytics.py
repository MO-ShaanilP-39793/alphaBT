import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# DAILY PORTFOLIO VALUE ANALYSIS (from daily_portfolio_values.csv)
# =============================================================================

def compute_portfolio_metrics(daily_pf: pd.DataFrame, risk_free_rate: float = 0.065) -> dict:
    """
    Compute comprehensive portfolio performance metrics from daily portfolio values.
    
    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - risk_free_rate: Annual risk-free rate (default 6.5% for India)
    
    Returns:
    - Dictionary with metrics: total_return, cagr, volatility, sharpe, sortino,
      max_drawdown, calmar_ratio, var_95, best_day, worst_day, positive_days_pct
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
    years = (end_date - start_date).days / 365.25
    
    # CAGR
    cagr = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else total_return
    
    # Volatility (annualized)
    daily_vol = df['daily_return'].std()
    annualized_vol = daily_vol * np.sqrt(252)
    
    # Sharpe Ratio
    excess_return = cagr - risk_free_rate
    sharpe = excess_return / annualized_vol if annualized_vol > 0 else 0
    
    # Sortino Ratio (downside deviation in the denominator)
    # Using risk_free_rate/252 as the target return for downside deviation calculation
    target_return = risk_free_rate / 252
    downside_diff = df['daily_return'] - target_return
    # We only care about returns below the target
    downside_diff = np.where(downside_diff < 0, downside_diff, 0)
    # Downside deviation is the square root of the mean of squared downside differences
    downside_dev = np.sqrt(np.mean(downside_diff**2)) * np.sqrt(252)
    sortino = excess_return / downside_dev if downside_dev > 0 else 0
    
    # Maximum Drawdown
    df['cummax'] = df['portfolio_value'].cummax()
    df['drawdown'] = (df['portfolio_value'] - df['cummax']) / df['cummax']
    max_drawdown = df['drawdown'].min()
    
    # Calmar Ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Value at Risk (95%)
    var_95 = df['daily_return'].quantile(0.05)
    
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
    """
    Compute drawdown series from daily portfolio values.
    
    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    
    Returns:
    - DataFrame with columns: ['date', 'portfolio_value', 'cummax', 'drawdown', 'drawdown_pct']
    """
    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df['cummax'] = df['portfolio_value'].cummax()
    df['drawdown'] = df['portfolio_value'] - df['cummax']
    df['drawdown_pct'] = (df['drawdown'] / df['cummax']) * 100
    
    return df[['date', 'portfolio_value', 'cummax', 'drawdown', 'drawdown_pct']]


def compute_rolling_returns(daily_pf: pd.DataFrame, windows: list = [21, 63, 126, 252]) -> pd.DataFrame:
    """
    Compute rolling returns for various windows.
    
    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - windows: List of rolling window sizes in trading days (default: 1M, 3M, 6M, 1Y)
    
    Returns:
    - DataFrame with date and rolling returns for each window
    """
    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    result = df[['date', 'portfolio_value']].copy()
    
    window_names = {21: '1M', 63: '3M', 126: '6M', 252: '1Y'}
    
    for window in windows:
        col_name = f'return_{window_names.get(window, f"{window}d")}'
        result[col_name] = df['portfolio_value'].pct_change(window) * 100
    
    return result


def compute_calendar_rolling_returns(daily_pf: pd.DataFrame, periods: list = [1, 3, 6, 12]) -> pd.DataFrame:
    """
    Compute rolling returns based on calendar months (not trading days).
    
    For example, the 1M return on Feb 5th compares to Jan 5th (or nearest available date),
    rather than going back 21 trading days.
    
    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - periods: List of rolling periods in calendar months (default: 1M, 3M, 6M, 12M)
    
    Returns:
    - DataFrame with date, portfolio_value, and calendar-based rolling returns for each period
    """
    from pandas.tseries.offsets import DateOffset
    
    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    
    result = df[['portfolio_value']].copy()
    
    period_names = {1: '1M', 3: '3M', 6: '6M', 12: '1Y'}
    
    for period in periods:
        col_name = f'return_{period_names.get(period, f"{period}M")}_cal'
        returns = []
        
        for current_date in result.index:
            # Calculate target date by going back N calendar months
            target_date = current_date - DateOffset(months=period)
            
            # Find the closest available date in the data (on or before target_date)
            available_dates = df.index[df.index <= target_date]
            
            if len(available_dates) == 0:
                # No data available that far back
                returns.append(np.nan)
            else:
                # Get the closest date to target_date
                closest_date = available_dates[-1]  # Last date on or before target
                
                # Calculate return
                current_value = result.loc[current_date, 'portfolio_value']
                past_value = df.loc[closest_date, 'portfolio_value']
                
                if past_value != 0:
                    ret = ((current_value - past_value) / past_value) * 100
                    returns.append(ret)
                else:
                    returns.append(np.nan)
        
        result[col_name] = returns
    
    result = result.reset_index()
    return result


def compute_monthly_returns(daily_pf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly returns table.
    
    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    
    Returns:
    - DataFrame pivoted with Year as rows, Month as columns, values are returns (%)
    """
    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Resample to month-end values
    df.set_index('date', inplace=True)
    monthly = df['portfolio_value'].resample('ME').last()
    monthly_returns = monthly.pct_change() * 100
    
    # Create pivot table
    result = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    
    pivot = result.pivot(index='year', columns='month', values='return')
    
    # Map month numbers to names (only for months that exist in the data)
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    pivot.columns = [month_names[m] for m in pivot.columns]
    
    # Add annual return
    annual_returns = {}
    years = pivot.index.tolist()
    
    for year in years:
        # Get data for this year (date is now the index after set_index)
        year_mask = df.index.year == year
        year_data = df[year_mask]
        
        if not year_data.empty:
            end_val = year_data['portfolio_value'].iloc[-1]
            
            # Start value is either end of previous year or start of this year (if first year)
            prev_year_mask = df.index.year == (year - 1)
            prev_year_data = df[prev_year_mask]
            
            if not prev_year_data.empty:
                start_val = prev_year_data['portfolio_value'].iloc[-1]
            else:
                # First year in dataset: return from inception to year end
                start_val = year_data['portfolio_value'].iloc[0]
            
            if start_val != 0:
                annual_returns[year] = (end_val - start_val) / start_val * 100
            else:
                annual_returns[year] = 0.0
        else:
            annual_returns[year] = None

    pivot['Annual'] = pivot.index.map(annual_returns)
    
    return pivot.round(2)


# =============================================================================
# BENCHMARK COMPARISON ANALYSIS (from portfolio_vs_index.csv)
# =============================================================================

def compute_benchmark_metrics(comparison_df: pd.DataFrame) -> dict:
    """
    Compute benchmark comparison metrics from portfolio vs index data.
    
    Parameters:
    - comparison_df: DataFrame with columns 
        ['date', 'pf_value', 'pf_return', 'index_fund_value', 'index_return', 'alpha']
    
    Returns:
    - Dictionary with comparison metrics: 
        beta, tracking_error, information_ratio,
        up_capture, down_capture, correlation, final_alpha
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
    # what it measures:
    # How much the portfolio's returns deviate from the benchmark (annualized standard deviation of excess returns).
    # Low TE (~1-2%): Portfolio closely tracks the index (passive/index fund)
    # High TE (~5%+): Portfolio diverges significantly (active management)
    df['excess_return'] = df['pf_daily_return'] - df['index_daily_return']
    tracking_error = df['excess_return'].std() * np.sqrt(252)
    
    # Information Ratio
    # what it measures: 
    # Risk-adjusted outperformance—how much excess return you're getting per unit of tracking error.
    # IR > 0.5: Good active management
    # IR > 1.0: Excellent active management
    excess_return_mean = df['excess_return'].mean() * 252  # Annualized
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
    
    # Final metrics from comparison_df
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


def compute_rolling_alpha(comparison_df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    Compute rolling alpha (outperformance vs index).
    
    Parameters:
    - comparison_df: DataFrame with columns ['date', 'pf_value', 'index_fund_value']
    - window: Rolling window in trading days (default 63 = 3 months)
    
    Returns:
    - DataFrame with date and rolling alpha columns
    """
    df = comparison_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df['pf_rolling_return'] = df['pf_value'].pct_change(window)
    df['index_rolling_return'] = df['index_fund_value'].pct_change(window)
    df['rolling_alpha'] = (df['pf_rolling_return'] - df['index_rolling_return']) * 100
    
    return df[['date', 'rolling_alpha']].dropna()


def compute_rolling_alpha_calendar(comparison_df: pd.DataFrame, period: int = 3) -> pd.DataFrame:
    """
    Compute rolling alpha (outperformance vs index) based on calendar months.
    
    Unlike compute_rolling_alpha which uses trading days, this function uses
    calendar months. For example, the 3-month rolling alpha on April 15th
    compares returns from January 15th (or nearest available date).
    
    Parameters:
    - comparison_df: DataFrame with columns ['date', 'pf_value', 'index_fund_value']
    - period: Rolling period in calendar months (default 3)
    
    Returns:
    - DataFrame with columns: ['date', 'pf_rolling_return', 'index_rolling_return', 'rolling_alpha']
    """
    from pandas.tseries.offsets import DateOffset
    
    df = comparison_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    
    pf_rolling_returns = []
    index_rolling_returns = []
    alphas_rolling = []
    dates = []
    
    for current_date in df.index:
        # Calculate target date by going back N calendar months
        target_date = current_date - DateOffset(months=period)
        
        # Find the closest available date in the data (on or before target_date)
        available_dates = df.index[df.index <= target_date]
        
        if len(available_dates) == 0:
            # No data available that far back
            pf_rolling_returns.append(np.nan)
            index_rolling_returns.append(np.nan)
            alphas_rolling.append(np.nan)
        else:
            # Get the closest date to target_date
            closest_date = available_dates[-1]  # Last date on or before target
            
            # Calculate portfolio return
            current_pf = df.loc[current_date, 'pf_value']
            past_pf = df.loc[closest_date, 'pf_value']
            
            # Calculate index return
            current_idx = df.loc[current_date, 'index_fund_value']
            past_idx = df.loc[closest_date, 'index_fund_value']
            
            if past_pf != 0 and past_idx != 0:
                pf_ret = ((current_pf - past_pf) / past_pf) * 100
                idx_ret = ((current_idx - past_idx) / past_idx) * 100
                pf_rolling_returns.append(pf_ret)
                index_rolling_returns.append(idx_ret)
                alphas_rolling.append(pf_ret - idx_ret)
            else:
                pf_rolling_returns.append(np.nan)
                index_rolling_returns.append(np.nan)
                alphas_rolling.append(np.nan)
        
        dates.append(current_date)
    
    result = pd.DataFrame({
        'date': dates,
        'pf_rolling_return': pf_rolling_returns,
        'index_rolling_return': index_rolling_returns,
        'rolling_alpha': alphas_rolling
    })
    
    return result.dropna().reset_index(drop=True)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_drawdown(daily_pf: pd.DataFrame, save_path: str = None):
    """
    Plot drawdown chart from daily portfolio values.
    
    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - save_path: If provided, saves plot to this path
    """
    dd = compute_drawdown_series(daily_pf)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1], sharex=True)
    
    # Portfolio value
    ax1.plot(dd['date'], dd['portfolio_value'], color='#1f77b4', linewidth=1.5, label='Portfolio')
    ax1.plot(dd['date'], dd['cummax'], color='gray', linestyle='--', alpha=0.7, label='Peak')
    ax1.fill_between(dd['date'], dd['portfolio_value'], dd['cummax'], alpha=0.3, color='red')
    ax1.set_ylabel('Portfolio Value')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Portfolio Value and Drawdown Analysis', fontsize=14)
    
    # Drawdown
    ax2.fill_between(dd['date'], dd['drawdown_pct'], 0, color='red', alpha=0.5)
    ax2.plot(dd['date'], dd['drawdown_pct'], color='darkred', linewidth=1)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    # Max drawdown annotation
    max_dd_idx = dd['drawdown_pct'].idxmin()
    max_dd_date = dd.loc[max_dd_idx, 'date']
    max_dd_val = dd.loc[max_dd_idx, 'drawdown_pct']
    ax2.annotate(f'Max DD: {max_dd_val:.1f}%', xy=(max_dd_date, max_dd_val),
                 xytext=(10, -20), textcoords='offset points',
                 fontsize=10, color='darkred', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_monthly_returns_heatmap(daily_pf: pd.DataFrame, save_path: str = None):
    """
    Plot monthly returns heatmap.
    
    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - save_path: If provided, saves plot to this path
    """
    monthly = compute_monthly_returns(daily_pf)
    
    # Remove 'Annual' column for heatmap
    monthly_only = monthly.drop(columns=['Annual'], errors='ignore')
    
    fig, ax = plt.subplots(figsize=(14, max(4, len(monthly_only) * 0.5 + 1)))
    
    # Create heatmap
    im = ax.imshow(monthly_only.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    
    # Labels
    ax.set_xticks(range(len(monthly_only.columns)))
    ax.set_xticklabels(monthly_only.columns)
    ax.set_yticks(range(len(monthly_only.index)))
    ax.set_yticklabels(monthly_only.index)
    
    # Annotate cells
    for i in range(len(monthly_only.index)):
        for j in range(len(monthly_only.columns)):
            val = monthly_only.iloc[i, j]
            if pd.notna(val):
                text_color = 'white' if abs(val) > 5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=text_color, fontsize=9)
    
    ax.set_title('Monthly Returns (%)', fontsize=14)
    plt.colorbar(im, ax=ax, label='Return (%)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_return_distribution(daily_pf: pd.DataFrame, save_path: str = None):
    """
    Plot histogram of daily returns distribution.
    
    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - save_path: If provided, saves plot to this path
    """
    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['daily_return'] = df['portfolio_value'].pct_change() * 100
    returns = df['daily_return'].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(returns, bins=50, edgecolor='black', alpha=0.7)
    
    # Color bins based on positive/negative
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('green')
    
    # Add statistics
    mean_ret = returns.mean()
    std_ret = returns.std()
    ax.axvline(mean_ret, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_ret:.2f}%')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # VaR lines
    var_95 = returns.quantile(0.05)
    var_99 = returns.quantile(0.01)
    ax.axvline(var_95, color='orange', linestyle=':', linewidth=2, label=f'VaR 95%: {var_95:.2f}%')
    ax.axvline(var_99, color='red', linestyle=':', linewidth=2, label=f'VaR 99%: {var_99:.2f}%')
    
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Daily Returns', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stats box
    stats_text = f'Mean: {mean_ret:.2f}%\nStd: {std_ret:.2f}%\nSkew: {returns.skew():.2f}\nKurtosis: {returns.kurtosis():.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_rolling_volatility(daily_pf: pd.DataFrame, window: int = 21, 
                            index_data: pd.DataFrame = None, save_path: str = None):
    """
    Plot rolling volatility (annualized), optionally compared to index.
    
    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - window: Rolling window in trading days (default 21 = 1 month)
    - index_data: Optional DataFrame with ['date', 'value' or 'close'] for benchmark comparison
    - save_path: If provided, saves plot to this path
    """
    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['daily_return'] = df['portfolio_value'].pct_change()
    df['rolling_vol'] = df['daily_return'].rolling(window).std() * np.sqrt(252) * 100
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Portfolio volatility
    ax.plot(df['date'], df['rolling_vol'], color='purple', linewidth=1.5, label='Portfolio')
    ax.fill_between(df['date'], df['rolling_vol'], alpha=0.2, color='purple')
    
    # Index volatility (if provided)
    if index_data is not None:
        idx = index_data.copy()
        idx['date'] = pd.to_datetime(idx['date'])
        idx = idx.sort_values('date')
        
        # Handle both 'value' and 'close' column names
        price_col = 'value' if 'value' in idx.columns else 'close'
        idx['daily_return'] = idx[price_col].pct_change()
        idx['rolling_vol'] = idx['daily_return'].rolling(window).std() * np.sqrt(252) * 100
        
        ax.plot(idx['date'], idx['rolling_vol'], color='gray', linewidth=1.5, 
                linestyle='--', label='Index', alpha=0.8)
        
        # Index average volatility
        avg_idx_vol = idx['rolling_vol'].mean()
        ax.axhline(avg_idx_vol, color='gray', linestyle=':', alpha=0.5, 
                   label=f'Index Avg: {avg_idx_vol:.1f}%')
    
    # Portfolio average volatility
    avg_vol = df['rolling_vol'].mean()
    ax.axhline(avg_vol, color='purple', linestyle=':', alpha=0.7, 
               label=f'Portfolio Avg: {avg_vol:.1f}%')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (% annualized)')
    ax.set_title(f'{window}-Day Rolling Volatility', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_rolling_sharpe(daily_pf: pd.DataFrame, window: int = 126, risk_free_rate: float = 0.065, save_path: str = None):
    """
    Plot rolling Sharpe ratio.
    
    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - window: Rolling window in trading days (default 126 = 6 months)
    - risk_free_rate: Annual risk-free rate
    - save_path: If provided, saves plot to this path
    """
    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['daily_return'] = df['portfolio_value'].pct_change()
    
    daily_rf = risk_free_rate / 252
    df['excess_return'] = df['daily_return'] - daily_rf
    
    rolling_mean = df['excess_return'].rolling(window).mean() * 252
    rolling_std = df['daily_return'].rolling(window).std() * np.sqrt(252)
    df['rolling_sharpe'] = rolling_mean / rolling_std
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(df['date'], df['rolling_sharpe'], color='teal', linewidth=1.5)
    ax.fill_between(df['date'], df['rolling_sharpe'], 0, 
                    where=df['rolling_sharpe'] >= 0, alpha=0.3, color='green')
    ax.fill_between(df['date'], df['rolling_sharpe'], 0,
                    where=df['rolling_sharpe'] < 0, alpha=0.3, color='red')
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.axhline(1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title(f'{window}-Day Rolling Sharpe Ratio', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_full_analysis_report(daily_pf: pd.DataFrame, comparison_df: pd.DataFrame = None, 
                                   output_dir: str = None) -> dict:
    """
    Generate a comprehensive analysis report with all metrics and plots.
    
    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - comparison_df: Optional DataFrame with benchmark comparison data
    - output_dir: If provided, saves all plots to this directory
    
    Returns:
    - Dictionary with all computed metrics
    """
    import os
    
    report = {}
    
    # Portfolio metrics
    report['portfolio_metrics'] = compute_portfolio_metrics(daily_pf)
    
    # Monthly returns table
    report['monthly_returns'] = compute_monthly_returns(daily_pf)
    
    # Rolling returns
    report['rolling_returns'] = compute_rolling_returns(daily_pf)
    
    # Drawdown series
    report['drawdown_series'] = compute_drawdown_series(daily_pf)
    
    # Benchmark metrics (if available)
    if comparison_df is not None:
        report['benchmark_metrics'] = compute_benchmark_metrics(comparison_df)
        report['rolling_alpha'] = compute_rolling_alpha(comparison_df)
    
    # Generate plots if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        plot_drawdown(daily_pf, save_path=os.path.join(output_dir, 'drawdown_chart.png'))
        plot_monthly_returns_heatmap(daily_pf, save_path=os.path.join(output_dir, 'monthly_returns_heatmap.png'))
        plot_return_distribution(daily_pf, save_path=os.path.join(output_dir, 'return_distribution.png'))
        plot_rolling_volatility(daily_pf, save_path=os.path.join(output_dir, 'rolling_volatility.png'))
        plot_rolling_sharpe(daily_pf, save_path=os.path.join(output_dir, 'rolling_sharpe.png'))
        
        print(f"Analysis plots saved to: {output_dir}")
    
    return report


# =============================================================================
# CATEGORY-LEVEL PLOTTING FUNCTIONS (from trade_results.csv)
# =============================================================================

def plot_category_returns_by_quarter(trade_results: pd.DataFrame, save_path: str = None):
    """
    Plot category returns comparison by quarter as a grouped bar chart.
    
    Parameters:
    - trade_results: DataFrame from trade_results.csv with columns including:
        ['quarter', 'cat', 'cat_weight', 'stock_return', 'holding_period']
    - save_path: If provided, saves plot to this path
    """
    cat_returns = get_category_returns_by_quarter(trade_results)
    pivot = cat_returns.pivot(index='quarter', columns='category', values='category_return')
    
    # Define colors for categories
    category_colors = {
        'largecap': '#2E86AB',   # Blue
        'midcap': '#A23B72',     # Purple/Magenta
        'smallcap': '#F18F01',   # Orange
        'low_vol': '#2E86AB',    # Blue (for volatility-based)
        'med_vol': '#A23B72',    # Purple
        'high_vol': '#F18F01',   # Orange
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(pivot.index))
    width = 0.25
    n_categories = len(pivot.columns)
    
    for i, category in enumerate(pivot.columns):
        offset = (i - n_categories / 2 + 0.5) * width
        color = category_colors.get(category, f'C{i}')
        bars = ax.bar(x + offset, pivot[category] * 100, width, 
                      label=category.title(), color=color, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, pivot[category] * 100):
            if pd.notna(val):
                height = bar.get_height()
                ax.annotate(f'{val:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -10),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=8, rotation=90)
    
    ax.set_xlabel('Quarter', fontsize=11)
    ax.set_ylabel('Return (%)', fontsize=11)
    ax.set_title('Category Returns by Quarter', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_category_cumulative_returns(trade_results: pd.DataFrame, save_path: str = None):
    """
    Plot cumulative returns by category over quarters as a line chart.
    
    Parameters:
    - trade_results: DataFrame from trade_results.csv with columns including:
        ['quarter', 'cat', 'cat_weight', 'stock_return', 'holding_period']
    - save_path: If provided, saves plot to this path
    """
    cat_returns = get_category_returns_by_quarter(trade_results)
    pivot = cat_returns.pivot(index='quarter', columns='category', values='category_return')
    
    # Compute cumulative returns: (1 + r1) * (1 + r2) * ... - 1
    cumulative = (1 + pivot).cumprod() - 1
    
    # Define colors and markers for categories
    category_styles = {
        'largecap': {'color': '#2E86AB', 'marker': 'o'},
        'midcap': {'color': '#A23B72', 'marker': 's'},
        'smallcap': {'color': '#F18F01', 'marker': '^'},
        'low_vol': {'color': '#2E86AB', 'marker': 'o'},
        'med_vol': {'color': '#A23B72', 'marker': 's'},
        'high_vol': {'color': '#F18F01', 'marker': '^'},
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, category in enumerate(cumulative.columns):
        style = category_styles.get(category, {'color': f'C{i}', 'marker': 'o'})
        ax.plot(cumulative.index, cumulative[category] * 100, 
                label=category.title(), color=style['color'], 
                marker=style['marker'], markersize=6, linewidth=2)
    
    ax.set_xlabel('Quarter', fontsize=11)
    ax.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax.set_title('Cumulative Returns by Category', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add final return annotations
    for category in cumulative.columns:
        final_val = cumulative[category].iloc[-1] * 100
        if pd.notna(final_val):
            style = category_styles.get(category, {'color': 'black'})
            ax.annotate(f'{final_val:.1f}%', 
                       xy=(cumulative.index[-1], final_val),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=9, color=style['color'], fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_category_distribution(trade_results: pd.DataFrame, save_path: str = None):
    """
    Plot stock count distribution by category per quarter as a stacked bar chart.
    
    Parameters:
    - trade_results: DataFrame from trade_results.csv with columns including:
        ['quarter', 'cat', 'holding_period']
    - save_path: If provided, saves plot to this path
    """
    counts_pivot = get_stock_counts_pivot(trade_results)
    counts_pivot = counts_pivot.set_index('quarter')
    
    # Define colors for categories
    category_colors = {
        'largecap': '#2E86AB',
        'midcap': '#A23B72',
        'smallcap': '#F18F01',
        'low_vol': '#2E86AB',
        'med_vol': '#A23B72',
        'high_vol': '#F18F01',
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create stacked bar chart
    bottom = np.zeros(len(counts_pivot))
    
    for i, category in enumerate(counts_pivot.columns):
        color = category_colors.get(category, f'C{i}')
        bars = ax.bar(counts_pivot.index, counts_pivot[category], bottom=bottom,
                      label=category.title(), color=color, edgecolor='white', linewidth=0.5)
        
        # Add count labels in the middle of each segment
        for j, (bar, count) in enumerate(zip(bars, counts_pivot[category])):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, 
                       bottom[j] + count / 2,
                       str(int(count)), ha='center', va='center', 
                       fontsize=9, fontweight='bold', color='white')
        
        bottom += counts_pivot[category].values
    
    ax.set_xlabel('Quarter', fontsize=11)
    ax.set_ylabel('Number of Stocks', fontsize=11)
    ax.set_title('Stock Distribution by Category', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_category_performance_summary(trade_results: pd.DataFrame, save_path: str = None):
    """
    Plot a summary dashboard with category-level performance metrics.
    Creates a 2x2 subplot with:
    - Category returns by quarter (bar)
    - Cumulative returns (line)
    - Stock distribution (stacked bar)
    - Average holding period by category (bar)
    
    Parameters:
    - trade_results: DataFrame from trade_results.csv
    - save_path: If provided, saves plot to this path
    """
    cat_returns = get_category_returns_by_quarter(trade_results)
    pivot_returns = cat_returns.pivot(index='quarter', columns='category', values='category_return')
    cumulative = (1 + pivot_returns).cumprod() - 1
    
    counts_pivot = get_stock_counts_pivot(trade_results).set_index('quarter')
    holding_periods = get_holding_period_by_quarter(trade_results)
    
    # Define colors
    category_colors = {
        'largecap': '#2E86AB', 'midcap': '#A23B72', 'smallcap': '#F18F01',
        'low_vol': '#2E86AB', 'med_vol': '#A23B72', 'high_vol': '#F18F01',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- Plot 1: Category Returns by Quarter ---
    ax1 = axes[0, 0]
    x = np.arange(len(pivot_returns.index))
    width = 0.25
    n_categories = len(pivot_returns.columns)
    
    for i, category in enumerate(pivot_returns.columns):
        offset = (i - n_categories / 2 + 0.5) * width
        color = category_colors.get(category, f'C{i}')
        ax1.bar(x + offset, pivot_returns[category] * 100, width, 
                label=category.title(), color=color, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Category Returns by Quarter', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pivot_returns.index, rotation=45, ha='right')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 2: Cumulative Returns ---
    ax2 = axes[0, 1]
    for i, category in enumerate(cumulative.columns):
        color = category_colors.get(category, f'C{i}')
        ax2.plot(cumulative.index, cumulative[category] * 100, 
                 label=category.title(), color=color, marker='o', markersize=5, linewidth=2)
    
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.set_title('Cumulative Returns by Category', fontweight='bold')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.sca(ax2)
    plt.xticks(rotation=45, ha='right')
    
    # --- Plot 3: Stock Distribution ---
    ax3 = axes[1, 0]
    bottom = np.zeros(len(counts_pivot))
    
    for i, category in enumerate(counts_pivot.columns):
        color = category_colors.get(category, f'C{i}')
        ax3.bar(counts_pivot.index, counts_pivot[category], bottom=bottom,
                label=category.title(), color=color, edgecolor='white', linewidth=0.5)
        bottom += counts_pivot[category].values
    
    ax3.set_xlabel('Quarter')
    ax3.set_ylabel('Number of Stocks')
    ax3.set_title('Stock Distribution by Category', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.sca(ax3)
    plt.xticks(rotation=45, ha='right')
    
    # --- Plot 4: Average Holding Period by Category ---
    ax4 = axes[1, 1]
    
    # Get holding period columns (those starting with 'avg_hp_')
    hp_cols = [col for col in holding_periods.columns if col.startswith('avg_hp_')]
    
    if hp_cols:
        hp_data = holding_periods.set_index('quarter')[hp_cols]
        hp_data.columns = [col.replace('avg_hp_', '') for col in hp_data.columns]
        
        x = np.arange(len(hp_data.index))
        width = 0.25
        n_cats = len(hp_data.columns)
        
        for i, category in enumerate(hp_data.columns):
            offset = (i - n_cats / 2 + 0.5) * width
            color = category_colors.get(category, f'C{i}')
            ax4.bar(x + offset, hp_data[category], width,
                    label=category.title(), color=color, edgecolor='black', linewidth=0.5)
        
        ax4.set_xlabel('Quarter')
        ax4.set_ylabel('Avg Holding Period (days)')
        ax4.set_title('Average Holding Period by Category', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(hp_data.index, rotation=45, ha='right')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No category-level\nholding period data', 
                 ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Average Holding Period by Category', fontweight='bold')
    
    plt.suptitle('Category-Level Performance Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_category_analysis_report(trade_results: pd.DataFrame, output_dir: str = None) -> dict:
    """
    Generate a comprehensive category-level analysis report with all metrics and plots.
    
    Parameters:
    - trade_results: DataFrame from trade_results.csv
    - output_dir: If provided, saves all plots to this directory
    
    Returns:
    - Dictionary with all computed category-level metrics
    """
    import os
    
    report = {}
    
    # Stock counts by category
    report['stock_counts'] = get_stock_counts_pivot(trade_results)
    
    # Category returns
    report['category_returns'] = get_category_returns_pivot(trade_results)
    
    # Holding periods
    report['holding_periods'] = get_holding_period_by_quarter(trade_results)
    
    # Comprehensive quarter analysis
    report['quarter_analysis'] = get_comprehensive_quarter_analysis(trade_results)
    
    # Generate plots if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        plot_category_returns_by_quarter(trade_results, 
            save_path=os.path.join(output_dir, 'category_returns_by_quarter.png'))
        plot_category_cumulative_returns(trade_results, 
            save_path=os.path.join(output_dir, 'category_cumulative_returns.png'))
        plot_category_distribution(trade_results, 
            save_path=os.path.join(output_dir, 'category_distribution.png'))
        plot_category_performance_summary(trade_results, 
            save_path=os.path.join(output_dir, 'category_performance_summary.png'))
        
        print(f"Category analysis plots saved to: {output_dir}")
    
    return report


# =============================================================================
# TRADE-LEVEL ANALYSIS (from trade_results.csv)
# =============================================================================

def get_stock_counts_by_category(trade_results: pd.DataFrame) -> pd.DataFrame:
    """
    Get count of stocks by category (largecap/midcap/smallcap or volatility categories) per quarter.
    
    Parameters:
    - trade_results: DataFrame from trade_results.csv with columns including:
        ['quarter', 'cat', 'holding_period']
    
    Returns:
    - DataFrame with columns: ['quarter', 'category', 'stock_count']
      Pivoted view also available via .pivot() on the result
    """
    required_cols = {'quarter', 'cat', 'holding_period'}
    if not required_cols.issubset(trade_results.columns):
        missing = required_cols - set(trade_results.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Filter to valid trades (entered positions)
    valid_trades = trade_results[trade_results['holding_period'].notna()].copy()
    
    # Count stocks by quarter and category
    counts = valid_trades.groupby(['quarter', 'cat']).size().reset_index(name='stock_count')
    counts = counts.rename(columns={'cat': 'category'})
    counts = counts.sort_values(['quarter', 'category']).reset_index(drop=True)
    
    return counts


def get_stock_counts_pivot(trade_results: pd.DataFrame) -> pd.DataFrame:
    """
    Get count of stocks by category per quarter in a pivoted (wide) format.
    
    Parameters:
    - trade_results: DataFrame from trade_results.csv
    
    Returns:
    - DataFrame with quarter as rows and categories as columns, values are stock counts
    """
    counts = get_stock_counts_by_category(trade_results)
    pivot = counts.pivot(index='quarter', columns='category', values='stock_count').fillna(0).astype(int)
    pivot = pivot.reset_index()
    return pivot


def get_category_returns_by_quarter(trade_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate returns for each category (largecap/midcap/smallcap or volatility categories) per quarter.
    
    Parameters:
    - trade_results: DataFrame from trade_results.csv with columns including:
        ['quarter', 'cat', 'cat_weight', 'stock_return', 'holding_period']
    
    Returns:
    - DataFrame with columns: ['quarter', 'category', 'category_return', 'stock_count', 'cat_weight']
      where category_return is the average return of stocks in that category
    """
    required_cols = {'quarter', 'cat', 'cat_weight', 'stock_return', 'holding_period'}
    if not required_cols.issubset(trade_results.columns):
        missing = required_cols - set(trade_results.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Filter to valid trades with returns
    valid_trades = trade_results[
        (trade_results['holding_period'].notna()) & 
        (trade_results['stock_return'].notna())
    ].copy()
    
    # Group by quarter and category
    category_stats = valid_trades.groupby(['quarter', 'cat']).agg({
        'stock_return': 'mean',
        'cat_weight': 'first',
        'co_name': 'count'
    }).reset_index()
    
    category_stats = category_stats.rename(columns={
        'cat': 'category',
        'stock_return': 'category_return',
        'co_name': 'stock_count'
    })
    
    # Round for readability
    category_stats['category_return'] = category_stats['category_return'].round(6)
    category_stats = category_stats.sort_values(['quarter', 'category']).reset_index(drop=True)
    
    return category_stats


def get_category_returns_pivot(trade_results: pd.DataFrame) -> pd.DataFrame:
    """
    Get category returns per quarter in a pivoted (wide) format.
    
    Parameters:
    - trade_results: DataFrame from trade_results.csv
    
    Returns:
    - DataFrame with quarter as rows and categories as columns, values are category returns
    """
    cat_returns = get_category_returns_by_quarter(trade_results)
    pivot = cat_returns.pivot(index='quarter', columns='category', values='category_return')
    pivot = pivot.reset_index()
    return pivot


def get_holding_period_by_quarter(trade_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average holding period per quarter (overall and by category).
    
    Parameters:
    - trade_results: DataFrame from trade_results.csv with columns including:
        ['quarter', 'cat', 'holding_period']
    
    Returns:
    - DataFrame with columns: ['quarter', 'avg_holding_period', 'avg_holding_period_largecap', 
      'avg_holding_period_midcap', 'avg_holding_period_smallcap'] (or volatility categories)
    """
    required_cols = {'quarter', 'cat', 'holding_period'}
    if not required_cols.issubset(trade_results.columns):
        missing = required_cols - set(trade_results.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Filter to valid trades
    valid_trades = trade_results[trade_results['holding_period'].notna()].copy()
    
    results = []
    
    for quarter, group in valid_trades.groupby('quarter'):
        row = {'quarter': quarter}
        
        # Overall average holding period
        row['avg_holding_period'] = round(group['holding_period'].mean(), 2)
        
        # Average holding period by category
        for cat in group['cat'].unique():
            cat_trades = group[group['cat'] == cat]
            col_name = f"avg_hp_{cat}"
            row[col_name] = round(cat_trades['holding_period'].mean(), 2)
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('quarter').reset_index(drop=True)
    
    return result_df


def get_comprehensive_quarter_analysis(trade_results: pd.DataFrame) -> pd.DataFrame:
    """
    Get a comprehensive analysis combining stock counts, category returns, holding periods,
    and TP/SL statistics per quarter.
    
    Parameters:
    - trade_results: DataFrame from trade_results.csv
    
    Returns:
    - DataFrame with comprehensive quarter-level statistics including:
        ['quarter', 'total_stocks', '{cat}_count', 'avg_holding_period', '{cat}_avg_hp',
         '{cat}_return', 'portfolio_return', 'TP_count', 'SL_count', 'time_exit_count',
         'avg_holding_period_SL', 'avg_holding_period_TP']
    """
    required_cols = {'quarter', 'cat', 'cat_weight', 'holding_period', 'stock_return'}
    if not required_cols.issubset(trade_results.columns):
        missing = required_cols - set(trade_results.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check if TP/SL columns are available
    has_tpsl = {'SL_triggered', 'TP_triggered'}.issubset(trade_results.columns)
    
    # Filter to valid trades
    valid_trades = trade_results[trade_results['holding_period'].notna()].copy()
    
    results = []
    
    for quarter, group in valid_trades.groupby('quarter'):
        row = {'quarter': quarter}
        
        # Total stocks (the number of stocks that were traded in this quarter)
        row['total_stocks'] = len(group)
        
        # Stock counts by category (the number of stocks traded in each category)
        cat_counts = group['cat'].value_counts()
        for cat, count in cat_counts.items():
            row[f'{cat}_count'] = count
        
        # Overall average holding period
        row['avg_holding_period'] = round(group['holding_period'].mean(), 2)
        
        # TP/SL statistics (if columns are available)
        if has_tpsl:
            tp_count = group['TP_triggered'].sum()
            sl_count = group['SL_triggered'].sum()
            time_exit_count = len(group) - tp_count - sl_count
            
            row['TP_count'] = int(tp_count)
            row['SL_count'] = int(sl_count)
            row['time_exit_count'] = int(time_exit_count)
            
            # Average holding period for SL triggered
            sl_trades = group[group['SL_triggered'] == True]
            row['avg_holding_period_SL'] = round(sl_trades['holding_period'].mean(), 2) if len(sl_trades) > 0 else None
            
            # Average holding period for TP triggered
            tp_trades = group[group['TP_triggered'] == True]
            row['avg_holding_period_TP'] = round(tp_trades['holding_period'].mean(), 2) if len(tp_trades) > 0 else None
        
        # Category returns and holding periods
        valid_returns = group[group['stock_return'].notna()]
        
        for cat in group['cat'].unique():
            cat_trades = group[group['cat'] == cat]
            cat_returns = valid_returns[valid_returns['cat'] == cat]
            
            # Holding period for this category
            row[f'{cat}_avg_hp'] = round(cat_trades['holding_period'].mean(), 2)
            
            # Return for this category
            if len(cat_returns) > 0:
                row[f'{cat}_return'] = round(cat_returns['stock_return'].mean(), 6)
            else:
                row[f'{cat}_return'] = None
        
        # Portfolio return (weighted category returns)
        if len(valid_returns) > 0:
            cat_ret_df = valid_returns.groupby('cat').agg({
                'stock_return': 'mean',
                'cat_weight': 'first'
            })
            row['portfolio_return'] = round((cat_ret_df['stock_return'] * cat_ret_df['cat_weight']).sum(), 6)
        else:
            row['portfolio_return'] = None
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('quarter').reset_index(drop=True)
    
    return result_df


# =============================================================================
# PORTFOLIO CHURN & OVERLAP ANALYSIS (from trade_results.csv)
# # =============================================================================

# def compute_quarter_overlap(trade_results: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute overlap of stock positions between consecutive quarters.
    
#     For each pair of consecutive quarters, calculates:
#     - Number and list of stocks that are retained (appear in both)
#     - Number and list of new stocks (only in current quarter)
#     - Number and list of exited stocks (only in previous quarter)
#     - Overlap percentage (from both perspectives)
    
#     Parameters:
#     - trade_results: DataFrame from trade_results.csv with columns including:
#         ['quarter', 'co_name', 'holding_period']
    
#     Returns:
#     - DataFrame with columns: ['prev_quarter', 'curr_quarter', 'prev_count', 'curr_count',
#         'retained_count', 'new_count', 'exited_count', 'overlap_pct_of_prev', 
#         'overlap_pct_of_curr', 'turnover_rate', 'retained_stocks', 'new_stocks', 'exited_stocks']
#     """
#     required_cols = {'quarter', 'co_name', 'holding_period'}
#     if not required_cols.issubset(trade_results.columns):
#         missing = required_cols - set(trade_results.columns)
#         raise ValueError(f"Missing required columns: {missing}")
    
#     # Filter to valid trades (entered positions)
#     valid_trades = trade_results[trade_results['holding_period'].notna()].copy()
    
#     # Get unique quarters sorted
#     quarters = sorted(valid_trades['quarter'].unique())
    
#     if len(quarters) < 2:
#         return pd.DataFrame()
    
#     results = []
    
#     for i in range(1, len(quarters)):
#         prev_q = quarters[i - 1]
#         curr_q = quarters[i]
        
#         # Get stock sets for each quarter
#         prev_stocks = set(valid_trades[valid_trades['quarter'] == prev_q]['co_name'])
#         curr_stocks = set(valid_trades[valid_trades['quarter'] == curr_q]['co_name'])
        
#         # Calculate overlaps
#         retained = prev_stocks & curr_stocks  # Intersection
#         new_stocks = curr_stocks - prev_stocks  # Only in current
#         exited = prev_stocks - curr_stocks  # Only in previous
        
#         prev_count = len(prev_stocks)
#         curr_count = len(curr_stocks)
#         retained_count = len(retained)
#         new_count = len(new_stocks)
#         exited_count = len(exited)
        
#         # Overlap percentages
#         # the percentage of stocks from the previous quarter that were also traded in the current quarter
#         overlap_pct_prev = (retained_count / prev_count * 100) if prev_count > 0 else 0
#         # the percentage of stocks from the current quarter that were also traded in the previous quarter
#         overlap_pct_curr = (retained_count / curr_count * 100) if curr_count > 0 else 0
        
#         # Turnover rate: (new + exited) / (2 * average holdings)
#         avg_holdings = (prev_count + curr_count) / 2  
#         turnover_rate = ((new_count + exited_count) / (2 * avg_holdings) * 100) if avg_holdings > 0 else 0
        
#         results.append({
#             'prev_quarter': prev_q,
#             'curr_quarter': curr_q,
#             'prev_count': prev_count,
#             'curr_count': curr_count,
#             'retained_count': retained_count,
#             'new_count': new_count,
#             'exited_count': exited_count,
#             'overlap_pct_of_prev': round(overlap_pct_prev, 2),
#             'overlap_pct_of_curr': round(overlap_pct_curr, 2),
#             'turnover_rate': round(turnover_rate, 2),
#             'retained_stocks': sorted(list(retained)),
#             'new_stocks': sorted(list(new_stocks)),
#             'exited_stocks': sorted(list(exited))
#         })
    
#     return pd.DataFrame(results)


# def compute_stock_persistence(trade_results: pd.DataFrame) -> pd.DataFrame:
#     """
#     Track how many consecutive quarters each stock appears in the portfolio.
#     Identifies stocks that persist across multiple quarters vs. one-quarter holdings.
    
#     Parameters:
#     - trade_results: DataFrame from trade_results.csv with columns including:
#         ['quarter', 'co_name', 'holding_period']
    
#     Returns:
#     - DataFrame with columns: ['co_name', 'total_appearances', 'max_consecutive_quarters',
#         'quarters_held', 'first_quarter', 'last_quarter', 'is_persistent']
#       where is_persistent = True if stock appeared in 3+ consecutive quarters
#     """
#     required_cols = {'quarter', 'co_name', 'holding_period'}
#     if not required_cols.issubset(trade_results.columns):
#         missing = required_cols - set(trade_results.columns)
#         raise ValueError(f"Missing required columns: {missing}")
    
#     # Filter to valid trades
#     valid_trades = trade_results[trade_results['holding_period'].notna()].copy()
    
#     # Get sorted quarters for ordering
#     quarters = sorted(valid_trades['quarter'].unique())
#     quarter_to_idx = {q: i for i, q in enumerate(quarters)}
    
#     results = []
    
#     for co_name, group in valid_trades.groupby('co_name'):
#         stock_quarters = sorted(group['quarter'].unique())
#         total_appearances = len(stock_quarters)
        
#         # Find max consecutive quarters
#         if len(stock_quarters) == 1:
#             max_consecutive = 1
#         else:
#             # Convert to indices and find longest consecutive sequence
#             indices = [quarter_to_idx[q] for q in stock_quarters]
#             indices.sort()
            
#             max_consecutive = 1
#             current_streak = 1
            
#             for j in range(1, len(indices)):
#                 if indices[j] == indices[j-1] + 1:
#                     current_streak += 1
#                     max_consecutive = max(max_consecutive, current_streak)
#                 else:
#                     current_streak = 1
        
#         results.append({
#             'co_name': co_name,
#             'total_appearances': total_appearances,
#             'max_consecutive_quarters': max_consecutive,
#             'quarters_held': stock_quarters,
#             'first_quarter': stock_quarters[0],
#             'last_quarter': stock_quarters[-1],
#             'is_persistent': max_consecutive >= 3
#         })
    
#     result_df = pd.DataFrame(results)
#     result_df = result_df.sort_values('max_consecutive_quarters', ascending=False).reset_index(drop=True)
    
#     return result_df


# def compute_churn_summary(trade_results: pd.DataFrame) -> dict:
#     """
#     Compute comprehensive portfolio churn summary statistics.
    
#     Parameters:
#     - trade_results: DataFrame from trade_results.csv
    
#     Returns:
#     - Dictionary with summary metrics:
#         - avg_overlap_pct: Average quarter-to-quarter overlap
#         - avg_turnover_rate: Average turnover rate
#         - total_unique_stocks: Total unique stocks across all quarters
#         - avg_stocks_per_quarter: Average number of stocks held per quarter
#         - persistent_stocks_count: Stocks appearing 3+ consecutive quarters
#         - one_quarter_stocks_count: Stocks appearing only once
#         - avg_stock_tenure: Average number of quarters a stock is held
#         - max_stock_tenure: Maximum quarters any single stock was held
#     """
#     overlap_df = compute_quarter_overlap(trade_results)
#     persistence_df = compute_stock_persistence(trade_results)
    
#     # Filter valid trades
#     valid_trades = trade_results[trade_results['holding_period'].notna()].copy()
#     quarters = sorted(valid_trades['quarter'].unique())
    
#     # Compute stocks per quarter
#     stocks_per_quarter = valid_trades.groupby('quarter')['co_name'].nunique()
    
#     summary = {
#         'num_quarters': len(quarters),
#         'first_quarter': quarters[0] if quarters else None,
#         'last_quarter': quarters[-1] if quarters else None,
#         'total_unique_stocks': persistence_df['co_name'].nunique(),
#         'avg_stocks_per_quarter': round(stocks_per_quarter.mean(), 2),
#         'avg_overlap_pct': round(overlap_df['overlap_pct_of_prev'].mean(), 2) if len(overlap_df) > 0 else 0,
#         'avg_turnover_rate': round(overlap_df['turnover_rate'].mean(), 2) if len(overlap_df) > 0 else 0,
#         'persistent_stocks_count': persistence_df['is_persistent'].sum(),
#         'one_quarter_stocks_count': (persistence_df['total_appearances'] == 1).sum(),
#         'avg_stock_tenure': round(persistence_df['total_appearances'].mean(), 2),
#         'max_stock_tenure': persistence_df['total_appearances'].max(),
#         'median_consecutive_quarters': persistence_df['max_consecutive_quarters'].median()
#     }
    
#     # Distribution of stock tenures
#     tenure_dist = persistence_df['total_appearances'].value_counts().sort_index().to_dict()
#     summary['tenure_distribution'] = tenure_dist
    
#     return summary


# def compute_overlap_by_category(trade_results: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute quarter-to-quarter overlap broken down by category.
#     Useful to see if certain categories (e.g., largecap) have more stable holdings.
    
#     Parameters:
#     - trade_results: DataFrame from trade_results.csv with columns including:
#         ['quarter', 'co_name', 'cat', 'holding_period']
    
#     Returns:
#     - DataFrame with columns: ['prev_quarter', 'curr_quarter', 'category',
#         'prev_count', 'curr_count', 'retained_count', 'overlap_pct']
#     """
#     required_cols = {'quarter', 'co_name', 'cat', 'holding_period'}
#     if not required_cols.issubset(trade_results.columns):
#         missing = required_cols - set(trade_results.columns)
#         raise ValueError(f"Missing required columns: {missing}")
    
#     # Filter to valid trades
#     valid_trades = trade_results[trade_results['holding_period'].notna()].copy()
    
#     quarters = sorted(valid_trades['quarter'].unique())
#     categories = valid_trades['cat'].unique()
    
#     if len(quarters) < 2:
#         return pd.DataFrame()
    
#     results = []
    
#     for i in range(1, len(quarters)):
#         prev_q = quarters[i - 1]
#         curr_q = quarters[i]
        
#         for cat in categories:
#             # Get stock sets for each quarter and category
#             prev_stocks = set(valid_trades[
#                 (valid_trades['quarter'] == prev_q) & (valid_trades['cat'] == cat)
#             ]['co_name'])
#             curr_stocks = set(valid_trades[
#                 (valid_trades['quarter'] == curr_q) & (valid_trades['cat'] == cat)
#             ]['co_name'])
            
#             retained = prev_stocks & curr_stocks
            
#             prev_count = len(prev_stocks)
#             curr_count = len(curr_stocks)
#             retained_count = len(retained)
            
#             overlap_pct = (retained_count / prev_count * 100) if prev_count > 0 else 0
            
#             results.append({
#                 'prev_quarter': prev_q,
#                 'curr_quarter': curr_q,
#                 'category': cat,
#                 'prev_count': prev_count,
#                 'curr_count': curr_count,
#                 'retained_count': retained_count,
#                 'overlap_pct': round(overlap_pct, 2)
#             })
    
#     return pd.DataFrame(results)


# def identify_time_exit_overlap_candidates(trade_results: pd.DataFrame) -> pd.DataFrame:
#     """
#     Identify stocks that exited via time-exit in one quarter and were re-selected 
#     in the next quarter. These are candidates for potential "hold through" strategy.
    
#     Parameters:
#     - trade_results: DataFrame from trade_results.csv with columns including:
#         ['quarter', 'co_name', 'holding_period', 'SL_triggered', 'TP_triggered']
#         (TP_triggered and SL_triggered used to infer time exits)
    
#     Returns:
#     - DataFrame with columns: ['co_name', 'exit_quarter', 'reentry_quarter',
#         'exit_price', 'reentry_price', 'price_gap_pct', 'exit_return', 'reentry_return',
#         'combined_return_if_held']
#       showing stocks that time-exited and then re-entered the portfolio
#     """
#     required_cols = {'quarter', 'co_name', 'holding_period', 'exit_price', 'entry_price', 'stock_return'}
#     if not required_cols.issubset(trade_results.columns):
#         missing = required_cols - set(trade_results.columns)
#         raise ValueError(f"Missing required columns: {missing}")
    
#     # Check for TP/SL columns to identify time exits
#     has_tpsl = {'SL_triggered', 'TP_triggered'}.issubset(trade_results.columns)
#     has_regime = 'regime_exit' in trade_results.columns
    
#     # Filter to valid trades
#     valid_trades = trade_results[trade_results['holding_period'].notna()].copy()
    
#     # Identify time exits (not TP, not SL, not regime)
#     if has_tpsl:
#         time_exit_mask = ~(valid_trades['SL_triggered'].fillna(False) | 
#                           valid_trades['TP_triggered'].fillna(False))
#         if has_regime:
#             time_exit_mask = time_exit_mask & ~valid_trades['regime_exit'].fillna(False)
#         time_exits = valid_trades[time_exit_mask].copy()
#     else:
#         # If no TP/SL columns, assume all are time exits for analysis
#         time_exits = valid_trades.copy()
    
#     quarters = sorted(valid_trades['quarter'].unique())
#     quarter_to_next = {quarters[i]: quarters[i+1] for i in range(len(quarters)-1)}
    
#     results = []
    
#     for _, exit_row in time_exits.iterrows():
#         exit_quarter = exit_row['quarter']
#         co_name = exit_row['co_name']
        
#         # Check if there's a next quarter
#         if exit_quarter not in quarter_to_next:
#             continue
        
#         next_quarter = quarter_to_next[exit_quarter]
        
#         # Check if this stock appears in next quarter
#         next_q_trades = valid_trades[
#             (valid_trades['quarter'] == next_quarter) & 
#             (valid_trades['co_name'] == co_name)
#         ]
        
#         if len(next_q_trades) > 0:
#             reentry_row = next_q_trades.iloc[0]
            
#             exit_price = exit_row['exit_price']
#             reentry_price = reentry_row['entry_price']
            
#             # Price gap between exit and re-entry
#             price_gap_pct = ((reentry_price - exit_price) / exit_price * 100) if exit_price else None
            
#             # Individual returns
#             exit_return = exit_row['stock_return']
#             reentry_return = reentry_row['stock_return']
            
#             # Hypothetical combined return if held through
#             # (1 + r1) * (1 + r2) - 1, but using continuous price
#             if exit_row['entry_price'] and reentry_row['exit_price']:
#                 combined_return = (reentry_row['exit_price'] - exit_row['entry_price']) / exit_row['entry_price']
#             else:
#                 combined_return = None
            
#             results.append({
#                 'co_name': co_name,
#                 'exit_quarter': exit_quarter,
#                 'reentry_quarter': next_quarter,
#                 'category': exit_row['cat'],
#                 'exit_entry_price': exit_row['entry_price'],
#                 'exit_price': exit_price,
#                 'reentry_price': reentry_price,
#                 'reentry_exit_price': reentry_row['exit_price'],
#                 'price_gap_pct': round(price_gap_pct, 2) if price_gap_pct else None,
#                 'exit_return': round(exit_return, 4) if exit_return else None,
#                 'reentry_return': round(reentry_return, 4) if reentry_return else None,
#                 'combined_return_if_held': round(combined_return, 4) if combined_return else None,
#                 'sum_of_individual_returns': round(exit_return + reentry_return, 4) if (exit_return and reentry_return) else None
#             })
    
#     result_df = pd.DataFrame(results)
#     if len(result_df) > 0:
#         result_df = result_df.sort_values(['exit_quarter', 'co_name']).reset_index(drop=True)
    
#     return result_df


# def compute_hold_through_analysis(trade_results: pd.DataFrame) -> dict:
#     """
#     Analyze potential benefits of holding stocks through multiple quarters
#     instead of exiting and re-entering.
    
#     Parameters:
#     - trade_results: DataFrame from trade_results.csv
    
#     Returns:
#     - Dictionary with analysis results:
#         - time_exit_reentry_count: Number of time-exit + re-entry events
#         - avg_price_gap_pct: Average price gap between exit and re-entry
#         - total_transaction_cost_savings: Estimated savings if held through (assuming 0.5% round-trip cost)
#         - return_comparison: Comparison of actual vs hypothetical hold-through returns
#     """
#     overlap_candidates = identify_time_exit_overlap_candidates(trade_results)
    
#     if len(overlap_candidates) == 0:
#         return {
#             'time_exit_reentry_count': 0,
#             'message': 'No time-exit + re-entry events found'
#         }
    
#     # Filter rows with valid data
#     valid_candidates = overlap_candidates.dropna(subset=['combined_return_if_held', 'sum_of_individual_returns'])
    
#     # Assume 0.5% round-trip transaction cost (exit + re-entry)
#     transaction_cost = 0.005
    
#     analysis = {
#         'time_exit_reentry_count': len(overlap_candidates),
#         'unique_stocks_affected': overlap_candidates['co_name'].nunique(),
#         'avg_price_gap_pct': round(overlap_candidates['price_gap_pct'].mean(), 2) if len(overlap_candidates) > 0 else None,
#         'median_price_gap_pct': round(overlap_candidates['price_gap_pct'].median(), 2) if len(overlap_candidates) > 0 else None,
#     }
    
#     if len(valid_candidates) > 0:
#         # Compare returns: sum of individual vs hold-through
#         actual_returns = valid_candidates['sum_of_individual_returns']
#         holdthrough_returns = valid_candidates['combined_return_if_held']
        
#         # Return difference (positive = hold-through is better)
#         return_diff = holdthrough_returns - actual_returns
        
#         analysis['avg_actual_return'] = round(actual_returns.mean(), 4)
#         analysis['avg_holdthrough_return'] = round(holdthrough_returns.mean(), 4)
#         analysis['avg_return_difference'] = round(return_diff.mean(), 4)
#         analysis['cases_holdthrough_better'] = int((return_diff > 0).sum())
#         analysis['cases_actual_better'] = int((return_diff < 0).sum())
        
#         # Estimated transaction cost savings
#         analysis['estimated_transaction_cost_per_event'] = transaction_cost
#         analysis['total_potential_savings'] = round(len(overlap_candidates) * transaction_cost, 4)
        
#         # By category breakdown
#         category_analysis = valid_candidates.groupby('category').agg({
#             'co_name': 'count',
#             'price_gap_pct': 'mean',
#             'sum_of_individual_returns': 'mean',
#             'combined_return_if_held': 'mean'
#         }).round(4)
#         category_analysis.columns = ['count', 'avg_price_gap', 'avg_actual_return', 'avg_holdthrough_return']
#         analysis['by_category'] = category_analysis.to_dict('index')
    
#     return analysis


# def compute_stock_presence_matrix(trade_results: pd.DataFrame) -> pd.DataFrame:
#     """
#     Create a presence matrix showing which stocks were held in which quarters.
#     Useful for visualizing portfolio composition over time.
    
#     Parameters:
#     - trade_results: DataFrame from trade_results.csv with columns including:
#         ['quarter', 'co_name', 'holding_period']
    
#     Returns:
#     - DataFrame with stocks as rows, quarters as columns, values are 1 (present) or 0 (absent)
#     """
#     required_cols = {'quarter', 'co_name', 'holding_period'}
#     if not required_cols.issubset(trade_results.columns):
#         missing = required_cols - set(trade_results.columns)
#         raise ValueError(f"Missing required columns: {missing}")
    
#     # Filter to valid trades
#     valid_trades = trade_results[trade_results['holding_period'].notna()].copy()
    
#     # Create presence indicator
#     valid_trades['present'] = 1
    
#     # Pivot to create matrix
#     presence_matrix = valid_trades.pivot_table(
#         index='co_name', 
#         columns='quarter', 
#         values='present',
#         fill_value=0,
#         aggfunc='max'  # In case of duplicates, mark as present
#     )
    
#     # Sort by total appearances (most frequent at top)
#     presence_matrix['total'] = presence_matrix.sum(axis=1)
#     presence_matrix = presence_matrix.sort_values('total', ascending=False)
#     presence_matrix = presence_matrix.drop(columns='total')
    
#     return presence_matrix


# def plot_churn_over_time(trade_results: pd.DataFrame, save_path: str = None):
#     """
#     Plot portfolio churn metrics over time showing turnover and overlap trends.
    
#     Parameters:
#     - trade_results: DataFrame from trade_results.csv
#     - save_path: If provided, saves plot to this path
#     """
#     overlap_df = compute_quarter_overlap(trade_results)
    
#     if len(overlap_df) == 0:
#         print("Insufficient data for churn analysis (need at least 2 quarters)")
#         return
    
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
#     # Create x-axis labels
#     x_labels = [f"{row['prev_quarter']}\n→\n{row['curr_quarter']}" for _, row in overlap_df.iterrows()]
#     x = range(len(x_labels))
    
#     # --- Plot 1: Overlap Percentage ---
#     ax1 = axes[0, 0]
#     ax1.bar(x, overlap_df['overlap_pct_of_prev'], color='#2E86AB', alpha=0.8, label='% of Previous Quarter')
#     ax1.plot(x, overlap_df['overlap_pct_of_prev'], color='#2E86AB', marker='o', linewidth=2)
#     ax1.axhline(overlap_df['overlap_pct_of_prev'].mean(), color='red', linestyle='--', 
#                 label=f'Avg: {overlap_df["overlap_pct_of_prev"].mean():.1f}%')
#     ax1.set_xlabel('Quarter Transition')
#     ax1.set_ylabel('Overlap (%)')
#     ax1.set_title('Quarter-to-Quarter Overlap', fontweight='bold')
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(x_labels, fontsize=8)
#     ax1.legend()
#     ax1.grid(True, alpha=0.3, axis='y')
    
#     # --- Plot 2: Turnover Rate ---
#     ax2 = axes[0, 1]
#     ax2.bar(x, overlap_df['turnover_rate'], color='#F18F01', alpha=0.8)
#     ax2.plot(x, overlap_df['turnover_rate'], color='#F18F01', marker='s', linewidth=2)
#     ax2.axhline(overlap_df['turnover_rate'].mean(), color='red', linestyle='--',
#                 label=f'Avg: {overlap_df["turnover_rate"].mean():.1f}%')
#     ax2.set_xlabel('Quarter Transition')
#     ax2.set_ylabel('Turnover Rate (%)')
#     ax2.set_title('Portfolio Turnover Rate', fontweight='bold')
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(x_labels, fontsize=8)
#     ax2.legend()
#     ax2.grid(True, alpha=0.3, axis='y')
    
#     # --- Plot 3: Stock Counts Breakdown ---
#     ax3 = axes[1, 0]
#     width = 0.35
#     x_arr = np.arange(len(overlap_df))
#     ax3.bar(x_arr - width/2, overlap_df['retained_count'], width, label='Retained', color='#2E86AB')
#     ax3.bar(x_arr + width/2, overlap_df['new_count'], width, label='New', color='#A23B72')
#     ax3.bar(x_arr + width/2, -overlap_df['exited_count'], width, label='Exited', color='#F18F01', alpha=0.7)
#     ax3.axhline(0, color='black', linewidth=0.5)
#     ax3.set_xlabel('Quarter Transition')
#     ax3.set_ylabel('Number of Stocks')
#     ax3.set_title('Stock Flow: Retained vs New vs Exited', fontweight='bold')
#     ax3.set_xticks(x_arr)
#     ax3.set_xticklabels(x_labels, fontsize=8)
#     ax3.legend()
#     ax3.grid(True, alpha=0.3, axis='y')
    
#     # --- Plot 4: Total Portfolio Size ---
#     ax4 = axes[1, 1]
#     # Combine prev and curr counts for a line
#     all_quarters = []
#     all_counts = []
#     for _, row in overlap_df.iterrows():
#         if row['prev_quarter'] not in all_quarters:
#             all_quarters.append(row['prev_quarter'])
#             all_counts.append(row['prev_count'])
#     # Add the last quarter
#     all_quarters.append(overlap_df.iloc[-1]['curr_quarter'])
#     all_counts.append(overlap_df.iloc[-1]['curr_count'])
    
#     ax4.plot(all_quarters, all_counts, color='#2E86AB', marker='o', linewidth=2, markersize=8)
#     ax4.fill_between(all_quarters, all_counts, alpha=0.3, color='#2E86AB')
#     ax4.set_xlabel('Quarter')
#     ax4.set_ylabel('Number of Stocks')
#     ax4.set_title('Portfolio Size Over Time', fontweight='bold')
#     ax4.tick_params(axis='x', rotation=45)
#     ax4.grid(True, alpha=0.3)
    
#     plt.suptitle('Portfolio Churn Analysis', fontsize=14, fontweight='bold', y=1.02)
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()


# def plot_stock_persistence_distribution(trade_results: pd.DataFrame, save_path: str = None):
#     """
#     Plot distribution of how long stocks are held in the portfolio.
    
#     Parameters:
#     - trade_results: DataFrame from trade_results.csv
#     - save_path: If provided, saves plot to this path
#     """
#     persistence_df = compute_stock_persistence(trade_results)
    
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
#     # --- Plot 1: Histogram of Total Appearances ---
#     ax1 = axes[0]
#     max_appearances = persistence_df['total_appearances'].max()
#     bins = range(1, max_appearances + 2)
    
#     counts, edges, patches = ax1.hist(persistence_df['total_appearances'], bins=bins, 
#                                        edgecolor='black', alpha=0.7, color='#2E86AB', align='left')
    
#     # Color single-quarter holdings differently
#     if len(patches) > 0:
#         patches[0].set_facecolor('#F18F01')
    
#     ax1.set_xlabel('Number of Quarters Held')
#     ax1.set_ylabel('Number of Stocks')
#     ax1.set_title('Distribution of Stock Holding Duration', fontweight='bold')
#     ax1.set_xticks(range(1, max_appearances + 1))
#     ax1.grid(True, alpha=0.3, axis='y')
    
#     # Add percentage labels
#     total_stocks = len(persistence_df)
#     for i, count in enumerate(counts):
#         if count > 0:
#             pct = count / total_stocks * 100
#             ax1.annotate(f'{pct:.1f}%', xy=(i + 1, count), ha='center', va='bottom', fontsize=9)
    
#     # --- Plot 2: Consecutive Quarters Distribution ---
#     ax2 = axes[1]
#     max_consec = persistence_df['max_consecutive_quarters'].max()
#     bins_consec = range(1, max_consec + 2)
    
#     counts2, _, patches2 = ax2.hist(persistence_df['max_consecutive_quarters'], bins=bins_consec,
#                                      edgecolor='black', alpha=0.7, color='#A23B72', align='left')
    
#     ax2.set_xlabel('Max Consecutive Quarters')
#     ax2.set_ylabel('Number of Stocks')
#     ax2.set_title('Distribution of Max Consecutive Holding', fontweight='bold')
#     ax2.set_xticks(range(1, max_consec + 1))
#     ax2.grid(True, alpha=0.3, axis='y')
    
#     # Add percentage labels
#     for i, count in enumerate(counts2):
#         if count > 0:
#             pct = count / total_stocks * 100
#             ax2.annotate(f'{pct:.1f}%', xy=(i + 1, count), ha='center', va='bottom', fontsize=9)
    
#     plt.suptitle('Stock Persistence Analysis', fontsize=14, fontweight='bold')
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()


# def plot_presence_heatmap(trade_results: pd.DataFrame, top_n: int = 30, save_path: str = None):
#     """
#     Plot a heatmap showing stock presence across quarters.
    
#     Parameters:
#     - trade_results: DataFrame from trade_results.csv
#     - top_n: Number of top stocks to show (by frequency)
#     - save_path: If provided, saves plot to this path
#     """
#     presence_matrix = compute_stock_presence_matrix(trade_results)
    
#     # Limit to top N stocks
#     plot_data = presence_matrix.head(top_n)
    
#     fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.3)))
    
#     # Create heatmap
#     im = ax.imshow(plot_data.values, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
#     # Labels
#     ax.set_xticks(range(len(plot_data.columns)))
#     ax.set_xticklabels(plot_data.columns, rotation=45, ha='right')
#     ax.set_yticks(range(len(plot_data.index)))
#     ax.set_yticklabels(plot_data.index, fontsize=8)
    
#     ax.set_xlabel('Quarter')
#     ax.set_ylabel('Stock')
#     ax.set_title(f'Stock Presence Across Quarters (Top {top_n} by Frequency)', fontweight='bold')
    
#     # Add gridlines
#     ax.set_xticks(np.arange(-0.5, len(plot_data.columns), 1), minor=True)
#     ax.set_yticks(np.arange(-0.5, len(plot_data.index), 1), minor=True)
#     ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    
#     plt.colorbar(im, ax=ax, label='Present (1) / Absent (0)', shrink=0.5)
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()


# def generate_churn_analysis_report(trade_results: pd.DataFrame, output_dir: str = None) -> dict:
#     """
#     Generate a comprehensive churn analysis report with all metrics and plots.
    
#     Parameters:
#     - trade_results: DataFrame from trade_results.csv
#     - output_dir: If provided, saves all plots to this directory
    
#     Returns:
#     - Dictionary with all computed churn metrics and analysis
#     """
#     import os
    
#     report = {}
    
#     # Churn summary
#     report['summary'] = compute_churn_summary(trade_results)
    
#     # Quarter overlap details
#     report['quarter_overlap'] = compute_quarter_overlap(trade_results)
    
#     # Stock persistence
#     report['stock_persistence'] = compute_stock_persistence(trade_results)
    
#     # Category-level overlap
#     report['overlap_by_category'] = compute_overlap_by_category(trade_results)
    
#     # Hold-through analysis
#     report['hold_through_analysis'] = compute_hold_through_analysis(trade_results)
    
#     # Time exit overlap candidates
#     report['time_exit_reentry_candidates'] = identify_time_exit_overlap_candidates(trade_results)
    
#     # Presence matrix
#     report['presence_matrix'] = compute_stock_presence_matrix(trade_results)
    
#     # Generate plots if output_dir provided
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
        
#         plot_churn_over_time(trade_results, 
#             save_path=os.path.join(output_dir, 'churn_over_time.png'))
#         plot_stock_persistence_distribution(trade_results, 
#             save_path=os.path.join(output_dir, 'stock_persistence_distribution.png'))
#         plot_presence_heatmap(trade_results, 
#             save_path=os.path.join(output_dir, 'stock_presence_heatmap.png'))
        
#         print(f"Churn analysis plots saved to: {output_dir}")
    
#     return report