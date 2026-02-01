import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

# --------------------------------------------------------------------------------
# 1. POSITION SIZING LOGIC
# --------------------------------------------------------------------------------

def calculate_position_sizes(df, total_capital):
    """
    Allocates capital to each stock.
    
    Supports two modes based on available columns:
    - If 'stock_weight' column exists: Direct allocation (preselected portfolio mode)
    - If 'cat_weight' column exists: Category-based allocation (stock selection mode)

    Returns:
        the passed dataframe with additional cols: [allocated_capital, shares]
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    if 'stock_weight' in df.columns:
        # Preselected portfolio mode: stock_weight is the direct portfolio weight
        df['allocated_capital'] = total_capital * df['stock_weight']
    else:
        # Stock selection mode: category weights divided among stocks in category
        cat_counts = df['cat'].value_counts()
        
        def get_stock_allocation(row):
            # How many stocks are in this category?
            n_stocks = cat_counts[row['cat']]
            
            # Total capital assigned to this entire category
            category_total_cap = total_capital * row['cat_weight']
            
            # Capital assigned to this specific stock
            stock_cap = category_total_cap / n_stocks
            return stock_cap

        # Calculate allocated capital per stock
        df['allocated_capital'] = df.apply(get_stock_allocation, axis=1)
    
    # Calculate number of shares (Position Size)
    df['shares'] = df['allocated_capital'] / df['entry_price']
    
    return df

# --------------------------------------------------------------------------------
# 2. DAILY EQUITY CURVE GENERATION (daily series of portfolio value)
# --------------------------------------------------------------------------------

def generate_quarter_equity_curve(trades, price_data, quarter):
    """
    Creates a daily series of portfolio value.

    trades: [quarter, co_name, cat, cat_weight, exit_date, entry_price, exit_price, allocated_capital, shares]
    (already filtered for the target quarter)
    price_data: [date, co_name, open, high, low, close]
    quarter: integer like 202402

    Returns a dataframe with index: dates in the quarter
        and a column called Total_Portfolio_Value
    """
    # 1. Determine the Date Range of the quarter
    quarter_str = str(quarter)
    year = int(quarter_str[:4])
    mm = int(quarter_str[4:])
    
    if mm == 2: quarter_start_date = pd.Timestamp(year, 2, 15)
    elif mm == 5: quarter_start_date = pd.Timestamp(year, 5, 31)
    elif mm == 8: quarter_start_date = pd.Timestamp(year, 8, 15)
    elif mm == 11: quarter_start_date = pd.Timestamp(year, 11, 15)
    
    # 2. Identify the "First 3 Trading Days" for this quarter
    all_dates = sorted(price_data[price_data['date'] >= quarter_start_date]['date'].unique())
    
    if len(all_dates) >= 3:
        entry_phase_dates = all_dates[:3]
        chart_start_date = entry_phase_dates[0]
    else:
        # Fallback if data is sparse
        warnings.warn(
            f"Insufficient trading days for quarter {quarter}. "
            f"Found only {len(all_dates)} days (expected at least 3). "
            f"Using available dates: {all_dates}",
            UserWarning
        )
        entry_phase_dates = all_dates
        chart_start_date = quarter_start_date

    # Determine End Date for the chart
    max_exit = pd.to_datetime(trades['exit_date']).max()
    chart_end_date = max_exit # + pd.Timedelta(days=5)  # for visual clarity
    
    # Create the Master Date Range for the plot, or the limits of the date index
    date_range = pd.date_range(start=chart_start_date, end=chart_end_date, freq='B')
    
    # 3. Prepare Price Data for Fast Lookup
    # Pivot: Index=Date, Columns=Co_Name, Values=Close
    price_pivot = price_data.pivot(index='date', columns='co_name', values='close')
    price_pivot = price_pivot.reindex(date_range).ffill()  # Align to date_range and forward fill gaps
    
    # 4. Calculate Daily Value (to be populated)
    # columns will be stocks in the portfolio, and the value will be the 
    # value of the position in that stock on that day
    daily_values = pd.DataFrame(index=date_range)
    
    # Set of entry phase dates for fast checking
    entry_phase_set = set(entry_phase_dates)
    
    # Create mapping of stock to category for later aggregation
    stock_to_category = dict(zip(trades['co_name'], trades['cat']))
    
    # Handle stocks with NaN entry_price - hold their allocated capital as cash
    invalid_entry_mask = trades['entry_price'].isna()
    if invalid_entry_mask.any():
        invalid_stocks = trades.loc[invalid_entry_mask, 'co_name'].tolist()
        uninvested_cash = trades.loc[invalid_entry_mask, 'allocated_capital'].sum()
        warnings.warn(
            f"Entry price is NaN for {len(invalid_stocks)} stock(s): {invalid_stocks}. "
            f"Holding ₹{uninvested_cash/10000000:.2f} Cr as cash (could not enter these positions).",
            UserWarning
        )
        # Add uninvested cash as a constant column
        daily_values['_uninvested_cash'] = uninvested_cash
        # Filter to only valid stocks for the main loop
        trades = trades[~invalid_entry_mask].copy()
    
    for _, row in trades.iterrows():  # every iteration populates a column in daily_values
        co_name = row['co_name']
        shares = row['shares']
        entry_price = row['entry_price']
        exit_date = row['exit_date']
        exit_price = row['exit_price']
        
        stock_series = []
        
        for current_date in date_range:
            current_date = pd.Timestamp(current_date)
            
            # CONDITION 1: Entry Phase (First 3 Days)
            # Value is fixed at cost basis
            if current_date in entry_phase_set:
                daily_val = shares * entry_price
                
            # CONDITION 2: After Exit Date
            # Value is fixed at exit proceeds (Cash)
            elif current_date > exit_date:
                daily_val = shares * exit_price
                
            # CONDITION 3: Active Holding Phase
            # Value fluctuates with market price
            else:
                if current_date in price_pivot.index and co_name in price_pivot.columns:
                    current_price = price_pivot.loc[current_date, co_name]
                    
                    if pd.isna(current_price):
                        # Fallback if price is missing in the middle of trade
                        warnings.warn(
                            f"Missing price data for {co_name} on {current_date.date()}. "
                            f"Using entry price {entry_price} as fallback.",
                            UserWarning
                        )
                        daily_val = shares * entry_price 
                    else:
                        daily_val = shares * current_price
                else:
                    # Fallback if date not in price data (e.g. mismatch)
                    warnings.warn(
                        f"Date {current_date.date()} or stock {co_name} not found in price data. "
                        f"Using entry price {entry_price} as fallback.",
                        UserWarning
                    )
                    daily_val = shares * entry_price

            stock_series.append(daily_val)
        
        daily_values[co_name] = stock_series

    # Aggregate by category (skip placeholder categories like '_default')
    categories = trades['cat'].unique()
    real_categories = [c for c in categories if c != '_default']
    for category in real_categories:
        # Get all stocks in this category
        stocks_in_category = [stock for stock, cat in stock_to_category.items() if cat == category]
        # Only include stocks that are actually in daily_values (exclude invalid ones)
        stocks_in_category = [s for s in stocks_in_category if s in daily_values.columns]
        # Sum their values
        if stocks_in_category:
            daily_values[f'{category}_value'] = daily_values[stocks_in_category].sum(axis=1)
    
    # Sum columns to get Total Portfolio Value (only stock columns, not category columns)
    # Include _uninvested_cash if it exists
    stock_columns = [col for col in daily_values.columns if not col.endswith('_value')]
    daily_values['Total_Portfolio_Value'] = daily_values[stock_columns].sum(axis=1)
    
    return daily_values


def _generate_quarter_sequence(first_quarter, last_quarter):
    """
    Generates a list of quarters between first_quarter and last_quarter (inclusive).
    Quarters are in format YYYYMM where MM is one of [02, 05, 08, 11].
    """
    quarter_months = [2, 5, 8, 11]
    
    first_quarter_year = int(str(first_quarter)[:4])
    first_quarter_month = int(str(first_quarter)[4:])
    
    quarters = []
    current_quarter_year = first_quarter_year
    current_quarter_month_idx = quarter_months.index(first_quarter_month)
    
    while True:
        current_quarter_month = quarter_months[current_quarter_month_idx]
        current_quarter = current_quarter_year * 100 + current_quarter_month
        
        if current_quarter > last_quarter:
            break
            
        quarters.append(current_quarter)
        
        # Move to next quarter
        current_quarter_month_idx += 1
        if current_quarter_month_idx >= len(quarter_months):
            current_quarter_month_idx = 0
            current_quarter_year += 1
    
    return quarters


def compute_pf_value_over_quarter(trades, price_data, target_quarter, initial_capital=1000000000):
    '''
    trades: [quarter, co_name, cat, cat_weight, exit_date, entry_price, exit_price]
    price_data: [date, co_name, open, high, low, close]
    target_quarter: integer like 202402
    initial_capital: sum like 100 crs

    Returns a dataframe with index: dates in the quarter
        and a column called Total_Portfolio_Value
    '''
    # Filter for target quarter
    quarter_df = trades[trades['quarter'] == target_quarter].copy()
    if quarter_df.empty:
        print("No data found.")
        return

    # Sizing
    # quarter_df: [quarter, co_name, cat, cat_weight, exit_date, entry_price, exit_price]
    quarter_df_sized = calculate_position_sizes(quarter_df, initial_capital)
    # quarter_df_sized: [quarter, co_name, cat, cat_weight, exit_date, entry_price, exit_price, allocated_capital, shares]

    # Equity Curve
    equity_curve = generate_quarter_equity_curve(quarter_df_sized, price_data, target_quarter)

    return equity_curve


def compute_pf_value_over_quarters(trades, price_data, first_quarter, last_quarter, initial_capital=1000000000):
    '''
    Computes portfolio value across multiple quarters at daily frequency.
    The ending value of each quarter becomes the starting capital for the next quarter.

    trades: [quarter, co_name, cat, cat_weight, exit_date, entry_price, exit_price]
    price_data: [date, co_name, open, high, low, close]
    first_quarter: integer like 202402 (start quarter, inclusive)
    last_quarter: integer like 202411 (end quarter, inclusive)
    initial_capital: sum like 100 crs

    Returns a dataframe with index: dates across all quarters
        and a column called Total_Portfolio_Value
    '''
    # Generate the sequence of quarters
    quarters = _generate_quarter_sequence(first_quarter, last_quarter)
    
    if not quarters:
        print("No valid quarters in the specified range.")
        return None
    
    all_equity_curves = []
    current_capital = initial_capital
    
    for quarter in quarters:
        # Compute equity curve for this quarter
        equity_curve = compute_pf_value_over_quarter(
            trades, price_data, quarter, current_capital
        )
        
        if equity_curve is None or equity_curve.empty:
            warnings.warn(f"No data for quarter {quarter}, skipping.", UserWarning)
            continue
        
        # Store the equity curve with quarter info
        equity_curve = equity_curve[['Total_Portfolio_Value']].copy()
        equity_curve['quarter'] = quarter
        all_equity_curves.append(equity_curve)
        
        # Update capital for next quarter (ending value of this quarter)
        current_capital = equity_curve['Total_Portfolio_Value'].iloc[-1]
    
    if not all_equity_curves:
        print("No equity curves generated for any quarter.")
        return None
    
    # Concatenate all equity curves
    combined_equity_curve = pd.concat(all_equity_curves)
    
    # Check for and handle any overlapping dates
    duplicated_dates = combined_equity_curve.index[combined_equity_curve.index.duplicated(keep=False)]
    if len(duplicated_dates) > 0:
        unique_duplicated = duplicated_dates.unique()
        warnings.warn(
            f"Found {len(unique_duplicated)} overlapping date(s) between quarters: "
            f"{[d.strftime('%Y-%m-%d') for d in unique_duplicated[:5]]}{'...' if len(unique_duplicated) > 5 else ''}. "
            f"Keeping the later quarter's value.",
            UserWarning
        )
        combined_equity_curve = combined_equity_curve[~combined_equity_curve.index.duplicated(keep='last')]
    
    # Sort by date
    combined_equity_curve = combined_equity_curve.sort_index()
    
    return combined_equity_curve


def plot_pf_value_over_quarter(trades, price_data, target_quarter, initial_capital=1000000000):
    '''
    trades: [quarter, co_name, cat, cat_weight, exit_date, entry_price, exit_price]
    price_data: [date, co_name, open, high, low, close]
    target_quarter: integer like 202402
    initial_capital: sum like 100 crs
    '''
    equity_curve = compute_pf_value_over_quarter(trades, price_data, target_quarter, initial_capital)
    # equity_curve has column Total_Portfolio_Value with dates as index

    plt.figure(figsize=(12, 6))
    
    # Plot the equity curve
    plt.plot(equity_curve.index, equity_curve['Total_Portfolio_Value'], 
             color='#1f77b4', linewidth=2.5, label='Portfolio Value')
    
    # Plot Initial Capital Line
    plt.axhline(y=initial_capital, color='black', linestyle='--', alpha=0.7, label='Initial Capital')
    
    # Fill area between curve and capital line for visual profit/loss indication
    plt.fill_between(equity_curve.index, 
                     equity_curve['Total_Portfolio_Value'], 
                     initial_capital, 
                     where=(equity_curve['Total_Portfolio_Value'] >= initial_capital),
                     interpolate=True, color='green', alpha=0.1)
    
    plt.fill_between(equity_curve.index, 
                     equity_curve['Total_Portfolio_Value'], 
                     initial_capital, 
                     where=(equity_curve['Total_Portfolio_Value'] < initial_capital),
                     interpolate=True, color='red', alpha=0.1)

    # Styling
    plt.title(f'Portfolio Performance: {target_quarter} (Initial Capital: ₹{initial_capital/10000000:.0f} Cr)', fontsize=14, pad=15)
    plt.ylabel('Value (INR)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Formatters
    def crores_formatter(x, pos):
        return f'{x/10000000:.1f} Cr'
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(crores_formatter))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Add final return annotation
    final_val = equity_curve['Total_Portfolio_Value'].iloc[-1]
    ret_pct = ((final_val - initial_capital) / initial_capital) * 100
    color = 'green' if ret_pct >= 0 else 'red'
    
    plt.figtext(0.15, 0.8, f"Final Return: {ret_pct:.2f}%", 
                fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.5'))

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_pf_value_over_quarters(trades, price_data, first_quarter, last_quarter, initial_capital=1000000000):
    '''
    Plots portfolio value across multiple quarters at daily frequency.

    trades: [quarter, co_name, cat, cat_weight, exit_date, entry_price, exit_price]
    price_data: [date, co_name, open, high, low, close]
    first_quarter: integer like 202402 (start quarter, inclusive)
    last_quarter: integer like 202411 (end quarter, inclusive)
    initial_capital: sum like 100 crs
    '''
    equity_curve = compute_pf_value_over_quarters(trades, price_data, first_quarter, last_quarter, initial_capital)
    
    if equity_curve is None or equity_curve.empty:
        print("No data to plot.")
        return
    
    plt.figure(figsize=(14, 7))
    
    # Plot the equity curve
    plt.plot(equity_curve.index, equity_curve['Total_Portfolio_Value'], 
             color='#1f77b4', linewidth=2, label='Portfolio Value')
    
    # Plot Initial Capital Line
    plt.axhline(y=initial_capital, color='black', linestyle='--', alpha=0.7, label='Initial Capital')
    
    # Fill area between curve and capital line for visual profit/loss indication
    plt.fill_between(equity_curve.index, 
                     equity_curve['Total_Portfolio_Value'], 
                     initial_capital, 
                     where=(equity_curve['Total_Portfolio_Value'] >= initial_capital),
                     interpolate=True, color='green', alpha=0.1)
    
    plt.fill_between(equity_curve.index, 
                     equity_curve['Total_Portfolio_Value'], 
                     initial_capital, 
                     where=(equity_curve['Total_Portfolio_Value'] < initial_capital),
                     interpolate=True, color='red', alpha=0.1)
    
    # Add vertical lines at quarter boundaries
    quarters_in_data = equity_curve['quarter'].unique()
    quarter_colors = plt.cm.tab10.colors
    for i, q in enumerate(quarters_in_data):
        quarter_data = equity_curve[equity_curve['quarter'] == q]
        if not quarter_data.empty:
            first_date = quarter_data.index[0]
            plt.axvline(x=first_date, color=quarter_colors[i % len(quarter_colors)], 
                       linestyle=':', alpha=0.5, linewidth=1.5)
            # Add quarter label at the top
            plt.text(first_date, plt.gca().get_ylim()[1], f'Q{q}', 
                    rotation=90, va='top', ha='right', fontsize=8, alpha=0.7)

    # Styling
    plt.title(f'Portfolio Performance: {first_quarter} to {last_quarter} (Initial Capital: ₹{initial_capital/10000000:.0f} Cr)', 
              fontsize=14, pad=15)
    plt.ylabel('Value (INR)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Formatters
    def crores_formatter(x, pos):
        return f'{x/10000000:.1f} Cr'
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(crores_formatter))
    
    # Use appropriate date formatter based on time span
    date_range_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if date_range_days > 365:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    elif date_range_days > 180:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.xticks(rotation=45)
    
    # Add final return annotation
    final_val = equity_curve['Total_Portfolio_Value'].iloc[-1]
    ret_pct = ((final_val - initial_capital) / initial_capital) * 100
    color = 'green' if ret_pct >= 0 else 'red'
    
    plt.figtext(0.15, 0.8, f"Total Return: {ret_pct:.2f}%", 
                fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.5'))

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def compute_pf_vs_index(trades, price_data, index_price_data, first_quarter, last_quarter, initial_capital=1000000000):
    '''
    Computes portfolio performance vs benchmark index across multiple quarters.

    trades: [quarter, co_name, cat, cat_weight, exit_date, entry_price, exit_price]
    price_data: [date, co_name, open, high, low, close]
    index_price_data: [date, value] - benchmark index values
    first_quarter: integer like 202402 (start quarter, inclusive)
    last_quarter: integer like 202411 (end quarter, inclusive)
    initial_capital: sum like 100 crs

    Returns a dataframe with columns:
        - date: the date
        - pf_value: portfolio value on that date
        - pf_return: portfolio return vs initial capital (%)
        - index_fund_value: value if initial capital was invested in index
        - index_return: index fund return vs initial capital (%)
        - alpha: pf_return - index_return (%)
    '''
    # Get portfolio equity curve
    daily_pf_values = compute_pf_value_over_quarters(
        trades, price_data, first_quarter, last_quarter, initial_capital
    )
    
    if daily_pf_values is None or daily_pf_values.empty:
        print("No portfolio data found.")
        return None
    
    # Prepare index data
    index_data = index_price_data.copy()
    index_data['date'] = pd.to_datetime(index_data['date'])
    index_data = index_data.set_index('date').sort_index()
    
    # Get the date range from portfolio equity curve
    pf_dates = daily_pf_values.index
    start_date = pf_dates[0]
    end_date = pf_dates[-1]
    
    # Filter index data to match portfolio date range
    index_data = index_data[(index_data.index >= start_date) & (index_data.index <= end_date)]
    
    # Reindex index data to match portfolio dates (forward fill for missing dates)
    index_data = index_data.reindex(pf_dates).ffill()
    
    # Get initial index value (first date's value)
    initial_index_value = index_data['value'].iloc[0]
    
    if pd.isna(initial_index_value):
        warnings.warn(
            f"No index data found for start date {start_date.date()}. "
            f"Using first available index value.",
            UserWarning
        )
        initial_index_value = index_data['value'].dropna().iloc[0]
    
    # Compute index fund value: initial_capital * (current_index / initial_index)
    index_fund_value = initial_capital * (index_data['value'] / initial_index_value)
    
    # Build result dataframe
    result = pd.DataFrame({
        'date': pf_dates,
        'pf_value': daily_pf_values['Total_Portfolio_Value'].values,
        'index_fund_value': index_fund_value.values
    })
    
    # Compute returns (as percentage)
    result['pf_return'] = ((result['pf_value'] - initial_capital) / initial_capital) * 100
    result['index_return'] = ((result['index_fund_value'] - initial_capital) / initial_capital) * 100
    
    # Compute alpha (outperformance)
    result['alpha'] = result['pf_return'] - result['index_return']
    
    # Reorder columns
    result = result[['date', 'pf_value', 'pf_return', 'index_fund_value', 'index_return', 'alpha']]
    
    return result


def plot_pf_vs_index(trades, price_data, index_price_data, first_quarter, last_quarter, initial_capital=1000000000, save_path=None):
    '''
    Plots portfolio value vs benchmark index across multiple quarters.

    trades: [quarter, co_name, cat, cat_weight, exit_date, entry_price, exit_price]
    price_data: [date, co_name, open, high, low, close]
    index_price_data: [date, value] - benchmark index values
    first_quarter: integer like 202402 (start quarter, inclusive)
    last_quarter: integer like 202411 (end quarter, inclusive)
    initial_capital: sum like 100 crs
    save_path: if provided, saves the plot to this path instead of showing it

    Returns:
    - comparison_df: DataFrame with portfolio vs index comparison data
    '''
    comparison_df = compute_pf_vs_index(
        trades, price_data, index_price_data, 
        first_quarter, last_quarter, initial_capital
    )
    
    if comparison_df is None or comparison_df.empty:
        print("No data to plot.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True)
    
    # ----- Top Plot: Portfolio Value vs Index Fund Value -----
    ax1.plot(comparison_df['date'], comparison_df['pf_value'], 
             color='#1f77b4', linewidth=2, label='Portfolio')
    ax1.plot(comparison_df['date'], comparison_df['index_fund_value'], 
             color='#ff7f0e', linewidth=2, label='Index Fund')
    
    # Plot Initial Capital Line
    ax1.axhline(y=initial_capital, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Fill area between portfolio and index for visual comparison
    ax1.fill_between(comparison_df['date'], 
                     comparison_df['pf_value'], 
                     comparison_df['index_fund_value'],
                     where=(comparison_df['pf_value'] >= comparison_df['index_fund_value']),
                     interpolate=True, color='green', alpha=0.1, label='Outperformance')
    
    ax1.fill_between(comparison_df['date'], 
                     comparison_df['pf_value'], 
                     comparison_df['index_fund_value'],
                     where=(comparison_df['pf_value'] < comparison_df['index_fund_value']),
                     interpolate=True, color='red', alpha=0.1, label='Underperformance')
    
    # Styling for top plot
    ax1.set_title(f'Portfolio vs Index: {first_quarter} to {last_quarter} (Initial Capital: ₹{initial_capital/10000000:.0f} Cr)', 
                  fontsize=14, pad=15)
    ax1.set_ylabel('Value (INR)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # Y-axis formatter for crores
    def crores_formatter(x, pos):
        return f'{x/10000000:.1f} Cr'
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(crores_formatter))
    
    # Calculate CAGR for portfolio and index
    start_date = comparison_df['date'].iloc[0]
    end_date = comparison_df['date'].iloc[-1]
    years = (end_date - start_date).days / 365.25
    
    final_pf_value = comparison_df['pf_value'].iloc[-1]
    final_index_value = comparison_df['index_fund_value'].iloc[-1]
    
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
    
    # ----- Bottom Plot: Alpha (Outperformance) -----
    ax2.fill_between(comparison_df['date'], comparison_df['alpha'], 0,
                     where=(comparison_df['alpha'] >= 0),
                     interpolate=True, color='green', alpha=0.3)
    ax2.fill_between(comparison_df['date'], comparison_df['alpha'], 0,
                     where=(comparison_df['alpha'] < 0),
                     interpolate=True, color='red', alpha=0.3)
    ax2.plot(comparison_df['date'], comparison_df['alpha'], 
             color='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax2.set_ylabel('Alpha (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # X-axis date formatting
    date_range_days = (comparison_df['date'].iloc[-1] - comparison_df['date'].iloc[0]).days
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
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    return comparison_df