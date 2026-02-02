import pandas as pd
import warnings


def select_top_k_stocks(
    stock_probabilities,
    k,
    selection_method='probability',
    min_prob_threshold=None,
    weighting_scheme='equal'):
    """
    Selects top k stocks per quarter based on probability or risk-adjusted scores.

    Parameters:
    - stock_probabilities (pd.DataFrame): Input dataframe with columns ['quarter', 'co_name', 'prob'].
        If selection_method='risk_adjusted', also requires 'volatility' column.
        Optionally can include 'category' column which will be passed through to output.
    - k (int): Number of top stocks to select per quarter.
    - selection_method (str): Method to rank stocks:
        - 'probability': Sort by probability score (default)
        - 'risk_adjusted': Sort by prob/volatility (risk-adjusted score)
    - min_prob_threshold (float or None): Minimum probability threshold to filter stocks before selection.
        If None, no filtering is applied (default).
    - weighting_scheme (str): How to weight selected stocks:
        - 'equal': Each stock gets weight 1/n where n is number of stocks selected for that quarter.

    Returns:
        pd.DataFrame with columns ['quarter', 'co_name', 'stock_weight']
        If 'category' column exists in input, it will be included as 'cat' in output.
        If selection_method='risk_adjusted', also includes 'risk_adj_score' column.
    """
    
    # Validate selection_method
    valid_methods = ['probability', 'risk_adjusted']
    if selection_method not in valid_methods:
        raise ValueError(f"selection_method must be one of {valid_methods}, got '{selection_method}'")
    
    # Validate weighting_scheme
    valid_schemes = ['equal']
    if weighting_scheme not in valid_schemes:
        raise ValueError(f"weighting_scheme must be one of {valid_schemes}, got '{weighting_scheme}'")
    
    # Check for volatility column if using risk_adjusted method
    if selection_method == 'risk_adjusted' and 'volatility' not in stock_probabilities.columns:
        raise ValueError("selection_method='risk_adjusted' requires 'volatility' column in input dataframe")
    
    # Check if category column exists in input
    has_category = 'category' in stock_probabilities.columns
    
    final_selection = []

    # Process each quarter independently
    for quarter, group in stock_probabilities.groupby('quarter'):
        
        # Apply minimum probability threshold if specified
        if min_prob_threshold is not None:
            group = group[group['prob'] >= min_prob_threshold]
            if len(group) == 0:
                warnings.warn(f"Quarter {quarter}: No stocks above min_prob_threshold {min_prob_threshold}")
                continue
        
        group = group.copy()
        
        # Calculate risk-adjusted score if needed
        if selection_method == 'risk_adjusted':
            group['risk_adj_score'] = group['prob'] / (group['volatility'] + 1e-8)
            group = group.sort_values(by='risk_adj_score', ascending=False)
        else:
            group = group.sort_values(by='prob', ascending=False)
        
        # Select top k stocks
        available_count = len(group)
        if available_count < k:
            warnings.warn(
                f"Quarter {quarter}: Requested {k} stocks, but only {available_count} available."
            )
        
        selected = group.head(k).copy()
        n_selected = len(selected)
        
        # Assign weights based on weighting_scheme
        if weighting_scheme == 'equal':
            selected['stock_weight'] = 1.0 / n_selected
        
        final_selection.append(selected)

    # Concatenate all selections and format output
    if final_selection:
        result_df = pd.concat(final_selection)
        
        # Rename 'category' to 'cat' for consistency with other selection functions
        if has_category:
            result_df = result_df.rename(columns={'category': 'cat'})
        
        # Build output columns
        output_cols = ['quarter', 'co_name']
        if has_category:
            output_cols.append('cat')
        output_cols.append('stock_weight')
        if selection_method == 'risk_adjusted':
            output_cols.append('risk_adj_score')
        
        result_df = result_df[output_cols]
        result_df = result_df.reset_index(drop=True)
        return result_df
    else:
        # Return empty dataframe with correct columns if input was empty
        output_cols = ['quarter', 'co_name']
        if has_category:
            output_cols.append('cat')
        output_cols.append('stock_weight')
        if selection_method == 'risk_adjusted':
            output_cols.append('risk_adj_score')
        return pd.DataFrame(columns=output_cols)


def select_and_weight_stocks_volatility(
    stock_probabilities, 
    selection_counts, 
    category_weights,
    selection_method='probability',
    min_prob_threshold=None):
    """
    Selects stocks based on volatility categories and probability scores, 
    and assigns category weights.

    Parameters:
    - stock_probabilities (pd.DataFrame): Input dataframe with columns ['quarter', 'co_name', 'prob', 'volatility']
    - selection_counts (list): [n_high, n_med, n_low] Number of stocks to select for High, Med, Low vol.
    - category_weights (list): [w_high, w_med, w_low] Weights for High, Med, Low vol categories.
    - selection_method (str): Method to rank stocks within categories:
        - 'probability': Sort by probability score
        - 'risk_adjusted': Sort by prob/volatility (risk-adjusted score)
    - min_prob_threshold (float or None): Minimum probability threshold to filter stocks before selection.
        If None, no filtering is applied (default).

    Returns: 
        pd.DataFrame with columns ['quarter', 'co_name', 'cat', 'cat_weight']
        If selection_method='risk_adjusted', also includes 'risk_adj_score' column.
    """
    
    # Validation to ensure lists are length 3
    if len(selection_counts) != 3 or len(category_weights) != 3:
        raise ValueError("selection_counts and category_weights must both be lists of length 3 (High, Med, Low).")
    
    # Validate selection_method
    valid_methods = ['probability', 'risk_adjusted']
    if selection_method not in valid_methods:
        raise ValueError(f"selection_method must be one of {valid_methods}, got '{selection_method}'")

    specs = {
        'high_volatility':   {'count': selection_counts[0], 'weight': category_weights[0]},
        'medium_volatility': {'count': selection_counts[1], 'weight': category_weights[1]},
        'low_volatility':    {'count': selection_counts[2], 'weight': category_weights[2]}
    }

    final_selection = []

    # Process each quarter independently
    for quarter, group in stock_probabilities.groupby('quarter'):
        
        # Apply minimum probability threshold if specified
        if min_prob_threshold is not None:
            group = group[group['prob'] >= min_prob_threshold]
            if len(group) == 0:
                warnings.warn(f"Quarter {quarter}: No stocks above min_prob_threshold {min_prob_threshold}")
                continue
        
        # 1. Determine Volatility Thresholds (33rd and 66th percentiles)
        # q=0.33 gives the cutoff for the bottom 33% (Low)
        # q=0.66 gives the cutoff for the middle 33% (Medium up to High)
        low_cutoff = group['volatility'].quantile(1/3)
        high_cutoff = group['volatility'].quantile(2/3)

        # 2. Function to classify volatility
        def get_vol_cat(vol):
            if vol > high_cutoff:
                return 'high_volatility'
            elif vol <= low_cutoff:
                return 'low_volatility'
            else:
                return 'medium_volatility'

        # Create a copy to avoid SettingWithCopy warnings on the slice
        # We assign the category to the current quarter's data
        current_data = group.copy()
        current_data['cat'] = current_data['volatility'].apply(get_vol_cat)
        
        # Calculate risk-adjusted score if needed
        if selection_method == 'risk_adjusted':
            # Risk-adjusted score = probability / volatility (higher is better)
            # Add small epsilon to avoid division by zero
            current_data['risk_adj_score'] = current_data['prob'] / (current_data['volatility'] + 1e-8)

        # 3. Select stocks for each category
        for cat in ['high_volatility', 'medium_volatility', 'low_volatility']:
            # Filter for the specific category
            cat_data = current_data[current_data['cat'] == cat]
            
            # Sort based on selection method
            if selection_method == 'risk_adjusted':
                cat_data = cat_data.sort_values(by='risk_adj_score', ascending=False)
            else:
                # Default: sort by probability (descending) to pick the best stocks
                cat_data = cat_data.sort_values(by='prob', ascending=False)
            
            target_count = specs[cat]['count']
            available_count = len(cat_data)

            # Warning check
            if available_count < target_count:
                warnings.warn(
                    f"Quarter {quarter}, Category {cat}: Requested {target_count} stocks, "
                    f"but only {available_count} available."
                )

            # Select top N
            selected = cat_data.head(target_count).copy()
            
            # Assign Weight
            selected['cat_weight'] = specs[cat]['weight']
            
            final_selection.append(selected)

    # 4. Concatenate all selections and format output
    if final_selection:
        result_df = pd.concat(final_selection)
        # Select and reorder specific columns based on method
        if selection_method == 'risk_adjusted':
            result_df = result_df[['quarter', 'co_name', 'cat', 'cat_weight', 'risk_adj_score']]
        else:
            result_df = result_df[['quarter', 'co_name', 'cat', 'cat_weight']]
        # Reset index for cleanliness
        result_df = result_df.reset_index(drop=True)
        return result_df
    else:
        # Return empty dataframe with correct columns if input was empty
        if selection_method == 'risk_adjusted':
            return pd.DataFrame(columns=['quarter', 'co_name', 'cat', 'cat_weight', 'risk_adj_score'])
        return pd.DataFrame(columns=['quarter', 'co_name', 'cat', 'cat_weight'])


def select_and_weight_stocks_mcap(
    stock_probabilities, 
    lms_count, 
    lms_w,
    selection_method='probability',
    min_prob_threshold=None):
    """
    Selects stocks based on market cap categories and probability scores,
    and assigns category weights.

    Parameters:
    - stock_probabilities (pd.DataFrame): Input dataframe with columns ['quarter', 'co_name', 'prob', 'category']
        where category is one of 'largecap', 'midcap', 'smallcap'.
        If selection_method='risk_adjusted', also requires 'volatility' column.
    - lms_count (list): [n_large, n_mid, n_small] 
    - lms_w (list): [w_large, w_mid, w_small]
    - selection_method (str): Method to rank stocks within categories:
        - 'probability': Sort by probability score 
        - 'risk_adjusted': Sort by prob/volatility (risk-adjusted score)s
    - min_prob_threshold (float or None): Minimum probability threshold to filter stocks before selection.
        If None, no filtering is applied (default).

    Returns:
        pd.DataFrame with columns ['quarter', 'co_name', 'cat', 'cat_weight']
        If selection_method='risk_adjusted', also includes 'risk_adj_score' column.
    """
    
    # Validation to ensure lists are length 3
    if len(lms_count) != 3 or len(lms_w) != 3:
        raise ValueError("lms_count and lms_w must both be lists of length 3 (Large, Mid, Small).")
    
    # Validate selection_method
    valid_methods = ['probability', 'risk_adjusted']
    if selection_method not in valid_methods:
        raise ValueError(f"selection_method must be one of {valid_methods}, got '{selection_method}'")
    
    # Check for volatility column if using risk_adjusted method
    if selection_method == 'risk_adjusted' and 'volatility' not in stock_probabilities.columns:
        raise ValueError("selection_method='risk_adjusted' requires 'volatility' column in input dataframe")

    specs = {
        'largecap': {'count': lms_count[0], 'weight': lms_w[0]},
        'midcap':   {'count': lms_count[1], 'weight': lms_w[1]},
        'smallcap': {'count': lms_count[2], 'weight': lms_w[2]}
    }

    final_selection = []

    # Process each quarter independently
    for quarter, group in stock_probabilities.groupby('quarter'):
        
        # Apply minimum probability threshold if specified
        if min_prob_threshold is not None:
            group = group[group['prob'] >= min_prob_threshold]
            if len(group) == 0:
                warnings.warn(f"Quarter {quarter}: No stocks above min_prob_threshold {min_prob_threshold}")
                continue
        
        # Calculate risk-adjusted score if needed
        if selection_method == 'risk_adjusted':
            group = group.copy()
            group['risk_adj_score'] = group['prob'] / (group['volatility'] + 1e-8)
        
        # Select stocks for each category
        for cat in ['largecap', 'midcap', 'smallcap']:
            # Filter for the specific category
            cat_data = group[group['category'] == cat].copy()
            
            # Sort based on selection method
            if selection_method == 'risk_adjusted':
                cat_data = cat_data.sort_values(by='risk_adj_score', ascending=False)
            else:
                # Default: sort by probability (descending) to pick the best stocks
                cat_data = cat_data.sort_values(by='prob', ascending=False)
            
            target_count = specs[cat]['count']
            available_count = len(cat_data)

            # Warning check
            if available_count < target_count:
                warnings.warn(
                    f"Quarter {quarter}, Category {cat}: Requested {target_count} stocks, "
                    f"but only {available_count} available."
                )

            # Select top N
            selected = cat_data.head(target_count).copy()
            
            # Assign Weight
            selected['cat_weight'] = specs[cat]['weight']
            
            final_selection.append(selected)

    # Concatenate all selections and format output
    if final_selection:
        result_df = pd.concat(final_selection)
        # Rename 'category' to 'cat' for output
        result_df = result_df.rename(columns={'category': 'cat'})
        # Select and reorder specific columns based on method
        if selection_method == 'risk_adjusted':
            result_df = result_df[['quarter', 'co_name', 'cat', 'cat_weight', 'risk_adj_score']]
        else:
            result_df = result_df[['quarter', 'co_name', 'cat', 'cat_weight']]
        # Reset index for cleanliness
        result_df = result_df.reset_index(drop=True)
        return result_df
    else:
        # Return empty dataframe with correct columns if input was empty
        if selection_method == 'risk_adjusted':
            return pd.DataFrame(columns=['quarter', 'co_name', 'cat', 'cat_weight', 'risk_adj_score'])
        return pd.DataFrame(columns=['quarter', 'co_name', 'cat', 'cat_weight'])


# =============================================================================
# PRICE DATA VALIDATION FUNCTIONS
# =============================================================================

def get_entry_start_date(quarter):
    """
    Get the entry search start date for a quarter.
    
    This mirrors the logic in backtest/tpsl.py get_date_params().
    Entry price is calculated as the average of the first 3 trading days
    starting from this date.
    
    Parameters:
        quarter: Quarter in YYYYMM format (int or str), e.g., 202402
        
    Returns:
        pd.Timestamp of the entry search start date
    """
    quarter_str = str(quarter)
    year = int(quarter_str[:4])
    mm = int(quarter_str[4:])
    
    if mm == 2:  # Feb quarter: entry starts Feb 15
        return pd.Timestamp(year, 2, 15)
    elif mm == 5:  # May quarter: entry starts May 31
        return pd.Timestamp(year, 5, 31)
    elif mm == 8:  # Aug quarter: entry starts Aug 15
        return pd.Timestamp(year, 8, 15)
    elif mm == 11:  # Nov quarter: entry starts Nov 15
        return pd.Timestamp(year, 11, 15)
    else:
        raise ValueError(f"Invalid quarter month: {mm}. Expected 2, 5, 8, or 11.")


def validate_price_data_coverage(input_data, price_data, first_quarter=None, last_quarter=None):
    """
    Identify stock-quarter combinations with insufficient price data for trading.
    
    A stock-quarter is considered invalid if:
    1. There are fewer than 3 trading days available from the entry start date
    2. Any of the first 3 trading days have NaN close prices
    
    Parameters:
        input_data: DataFrame with stocks to validate (must have 'quarter', 'co_name')
        price_data: DataFrame with OHLCV data (must have 'date', 'co_name', 'close')
        first_quarter: Start quarter (inclusive). If None, uses min quarter in input_data.
        last_quarter: End quarter (inclusive). If None, uses max quarter in input_data.
    
    Returns:
        DataFrame with columns: [quarter, co_name, entry_start_date, issue, detail]
        Empty DataFrame if no issues found.
    """
    # Ensure date column is datetime
    price_data = price_data.copy()
    price_data['date'] = pd.to_datetime(price_data['date'])
    
    # Determine quarter range
    if first_quarter is None:
        first_quarter = input_data['quarter'].min()
    if last_quarter is None:
        last_quarter = input_data['quarter'].max()
    
    # Filter input data to quarter range
    filtered_input = input_data[
        (input_data['quarter'] >= first_quarter) & 
        (input_data['quarter'] <= last_quarter)
    ]
    
    issues = []
    
    # Get unique stock-quarter combinations
    stock_quarters = filtered_input[['quarter', 'co_name']].drop_duplicates()
    
    for _, row in stock_quarters.iterrows():
        quarter = row['quarter']
        stock = row['co_name']
        
        entry_start = get_entry_start_date(quarter)
        
        # Get price data for this stock from entry_start onwards
        stock_prices = price_data[
            (price_data['co_name'] == stock) & 
            (price_data['date'] >= entry_start)
        ].sort_values('date')
        
        # Check 1: Enough trading days?
        if len(stock_prices) < 3:
            issues.append({
                'quarter': quarter,
                'co_name': stock,
                'entry_start_date': entry_start.strftime('%Y-%m-%d'),
                'issue': 'insufficient_trading_days',
                'detail': f'Found {len(stock_prices)} days, need at least 3'
            })
            continue
        
        # Check 2: Are the first 3 days' close prices valid (non-NaN)?
        entry_closes = stock_prices.iloc[:3]['close']
        nan_count = entry_closes.isna().sum()
        if nan_count > 0:
            issues.append({
                'quarter': quarter,
                'co_name': stock,
                'entry_start_date': entry_start.strftime('%Y-%m-%d'),
                'issue': 'nan_close_price',
                'detail': f'{nan_count}/3 entry days have NaN close price'
            })
    
    return pd.DataFrame(issues)


def filter_tradeable_stocks(input_data, price_data, first_quarter=None, last_quarter=None):
    """
    Remove stock-quarter combinations that cannot be traded due to price data issues.
    
    This function validates price data coverage and filters out problematic
    stock-quarter combinations from the input data.
    
    Parameters:
        input_data: DataFrame with candidate stocks (must have 'quarter', 'co_name')
        price_data: DataFrame with OHLCV data (must have 'date', 'co_name', 'close')
        first_quarter: Start quarter (inclusive). If None, uses min quarter in input_data.
        last_quarter: End quarter (inclusive). If None, uses max quarter in input_data.
    
    Returns:
        Tuple of (filtered_input_data, issues_df):
        - filtered_input_data: input_data with problematic rows removed
        - issues_df: DataFrame describing what was filtered and why
    """
    # Get issues
    issues_df = validate_price_data_coverage(
        input_data, price_data, first_quarter, last_quarter
    )
    
    if issues_df.empty:
        return input_data.copy(), issues_df
    
    # Create a set of (quarter, co_name) tuples to filter out
    invalid_pairs = set(zip(issues_df['quarter'], issues_df['co_name']))
    
    # Filter out invalid stock-quarter combinations
    mask = input_data.apply(
        lambda row: (row['quarter'], row['co_name']) not in invalid_pairs, 
        axis=1
    )
    filtered_data = input_data[mask].copy()
    
    return filtered_data, issues_df
