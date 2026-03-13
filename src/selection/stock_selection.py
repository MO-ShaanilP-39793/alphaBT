import numpy as np
import pandas as pd
import warnings

from config.defaults import (
    DEFAULT_SORT_BY,
    DEFAULT_MIN_PROB_THRESHOLD,
    DEFAULT_TOP_K_WEIGHTING,
    DEFAULT_WEIGHTING_SCHEME,
    RISK_ADJUSTED_EPSILON,
    get_dimension_categories,
)
from utils.quarter import get_quarter_dates


def _assign_volatility_categories(series):
    """
    Assign volatility tercile categories based on 33rd/66th percentile cutoffs.
    
    Parameters:
        series: pandas Series of volatility values
        
    Returns:
        pandas Series of category labels ('high_volatility', 'medium_volatility', 'low_volatility')
    """
    low_cutoff = series.quantile(1/3)
    high_cutoff = series.quantile(2/3)
    
    def classify(vol):
        if vol > high_cutoff:
            return 'high_volatility'
        elif vol <= low_cutoff:
            return 'low_volatility'
        else:
            return 'medium_volatility'
    
    return series.apply(classify)


def select_top_k_stocks(
    stock_probabilities,
    k,
    sort_by=DEFAULT_SORT_BY,
    min_prob_threshold=DEFAULT_MIN_PROB_THRESHOLD,
    weighting_scheme=DEFAULT_TOP_K_WEIGHTING,
    category_caps=None):
    """
    Selects top k stocks per quarter based on probability or risk-adjusted scores.

    Parameters:
    - stock_probabilities (pd.DataFrame): Input dataframe with columns ['quarter', 'co_name', 'prob'].
        If sort_by='risk_adjusted_probability', also requires 'volatility' column.
        Optionally can include 'category' column which will be passed through to output.
    - k (int): Number of top stocks to select per quarter.
    - sort_by (str): Method to rank/sort stocks:
        - 'probability': Sort by probability score (default)
        - 'risk_adjusted_probability': Sort by prob/volatility (risk-adjusted score)
    - min_prob_threshold (float or None): Minimum probability threshold to filter stocks before selection.
        If None, no filtering is applied (default).
    - weighting_scheme (str): How to weight selected stocks:
        - 'equal': Each stock gets weight 1/n where n is number of stocks selected for that quarter.
    - category_caps (dict or None): Optional per-category maximum stock counts.
        Keys are mcap category names (e.g., 'smallcap', 'midcap', 'largecap'),
        values are the maximum number of stocks allowed from that category.
        Categories not listed are uncapped. Requires 'category' column in input.
        Example: {'smallcap': 10} — at most 10 smallcap stocks per quarter.

    Returns:
        pd.DataFrame with columns ['quarter', 'co_name', 'stock_weight']
        If 'category' column exists in input, it will be included as 'cat' in output.
        If sort_by='risk_adjusted_probability', also includes 'risk_adj_score' column.
    """
    
    # Validate sort_by
    valid_methods = ['probability', 'risk_adjusted_probability']
    if sort_by not in valid_methods:
        raise ValueError(f"sort_by must be one of {valid_methods}, got '{sort_by}'")
    
    # Validate weighting_scheme
    valid_schemes = ['equal']
    if weighting_scheme not in valid_schemes:
        raise ValueError(f"weighting_scheme must be one of {valid_schemes}, got '{weighting_scheme}'")
    
    # Check for volatility column if using risk_adjusted_probability method
    if sort_by == 'risk_adjusted_probability' and 'volatility' not in stock_probabilities.columns:
        raise ValueError("sort_by='risk_adjusted_probability' requires 'volatility' column in input dataframe")
    
    # Validate category_caps
    if category_caps is not None:
        if 'category' not in stock_probabilities.columns:
            raise ValueError(
                "category_caps requires 'category' column in input dataframe. "
                "Input data must include mcap categories (largecap/midcap/smallcap)."
            )
        from config.defaults import MCAP_CATEGORIES
        invalid_cats = set(category_caps.keys()) - set(MCAP_CATEGORIES)
        if invalid_cats:
            raise ValueError(
                f"category_caps keys must be valid mcap categories {MCAP_CATEGORIES}, "
                f"got invalid: {invalid_cats}"
            )
    
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
        
        # Calculate risk-adjusted score if needed and sort stocks probability / risk_adjusted_probability
        if sort_by == 'risk_adjusted_probability':
            group['risk_adj_score'] = group['prob'] / (group['volatility'] + RISK_ADJUSTED_EPSILON)
            group = group.sort_values(by='risk_adj_score', ascending=False)
        else:
            group = group.sort_values(by='prob', ascending=False)
        
        # Select top k stocks
        available_count = len(group)
        if available_count < k:
            warnings.warn(
                f"Quarter {quarter}: Requested {k} stocks, but only {available_count} available."
            )
        
        if category_caps is not None:
            # Iterative selection with per-category caps
            cat_counts = {}  # track how many stocks selected per category
            selected_indices = []
            for idx, row in group.iterrows():
                cat = row['category']
                cap = category_caps.get(cat)  # None means uncapped
                if cap is not None and cat_counts.get(cat, 0) >= cap:
                    continue  # skip — this category has hit its cap
                selected_indices.append(idx)
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                if len(selected_indices) >= k:
                    break
            selected = group.loc[selected_indices].copy()
            if len(selected) < k and available_count >= k:
                warnings.warn(
                    f"Quarter {quarter}: Requested {k} stocks, but only {len(selected)} "
                    f"available after applying category_caps {category_caps}."
                )
        else:
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
        output_cols = ['quarter', 'co_name', 'stock_weight']
        if has_category:
            output_cols.append('cat')
        if sort_by == 'risk_adjusted_probability':
            output_cols.append('risk_adj_score')
        if 'sector' in result_df.columns:
            output_cols.append('sector')
        
        result_df = result_df[output_cols]
        result_df = result_df.reset_index(drop=True)
        return result_df
    else:
        # Return empty dataframe with correct columns if input was empty
        output_cols = ['quarter', 'co_name', 'stock_weight']
        if has_category:
            output_cols.append('cat')
        if sort_by == 'risk_adjusted_probability':
            output_cols.append('risk_adj_score')
        if 'sector' in stock_probabilities.columns:
            output_cols.append('sector')
        return pd.DataFrame(columns=output_cols)


# =============================================================================
# CATEGORY BASED SELECTION AND WEIGHTING
# =============================================================================

def select_and_weight_stocks(
    stock_probabilities,
    selection_dimension,
    weighting_dimension,
    selection_counts,
    category_weights,
    sort_by=DEFAULT_SORT_BY,
    min_prob_threshold=DEFAULT_MIN_PROB_THRESHOLD,
    weighting_scheme=DEFAULT_WEIGHTING_SCHEME):
    """
    Unified stock selection with independent selection and weighting dimensions.
    
    Stocks are bucketed and selected by `selection_dimension`, then capital is
    allocated by `weighting_dimension`. These can be the same (traditional behavior)
    or different (cross-dimensional, e.g., select by volatility, weight by mcap).
    
    Parameters:
    - stock_probabilities (pd.DataFrame): Input with columns ['quarter', 'co_name', 'prob'].
        Also requires 'volatility' if any dimension is 'volatility',
        and 'category' if any dimension is 'mcap'.
        If sort_by='risk_adjusted_probability', requires 'volatility'.
    - selection_dimension (str): 'volatility' or 'mcap' — which axis to bucket/select stocks by
    - weighting_dimension (str): 'volatility' or 'mcap' — which axis to assign weights by
    - selection_counts (list): [n1, n2, n3] stocks to select per selection-dimension category
    - category_weights (list): [w1, w2, w3] weights per weighting-dimension category.
        Only used when weighting_scheme is 'use_category_weights'.
    - sort_by (str): 'probability' or 'risk_adjusted_probability' — how to rank/sort stocks
    - min_prob_threshold (float or None): Minimum probability threshold
    - weighting_scheme (str): 'use_category_weights' or 'equal'
    
    Returns:
        pd.DataFrame with columns:
        - 'quarter', 'co_name'
        - 'cat': weighting dimension categories (used for position sizing)
        - 'selection_cat': selection dimension categories (informational)
        - 'weight_cat': weighting dimension categories (informational)
        - 'stock_weight': always present, direct stock-level weight
        - 'cat_weight': present when weighting_scheme='use_category_weights' (informational)
        - 'risk_adj_score' (when sort_by='risk_adjusted_probability')
    """
    # Validate dimensions
    for dim_name, dim_val in [('selection_dimension', selection_dimension),
                               ('weighting_dimension', weighting_dimension)]:
        if dim_val not in ('volatility', 'mcap'):
            raise ValueError(f"{dim_name} must be 'volatility' or 'mcap', got '{dim_val}'")
    
    # Validate required input columns
    needs_volatility = (selection_dimension == 'volatility' or
                        weighting_dimension == 'volatility' or
                        sort_by == 'risk_adjusted_probability')
    needs_mcap = selection_dimension == 'mcap' or weighting_dimension == 'mcap'
    
    if needs_volatility and 'volatility' not in stock_probabilities.columns:
        raise ValueError(
            f"'volatility' column required for selection_dimension='{selection_dimension}', "
            f"weighting_dimension='{weighting_dimension}', sort_by='{sort_by}'"
        )
    if needs_mcap and 'category' not in stock_probabilities.columns:
        raise ValueError(
            f"'category' column required for selection_dimension='{selection_dimension}', "
            f"weighting_dimension='{weighting_dimension}'"
        )
    
    # Validate counts, weights, method, weighting_scheme
    if len(selection_counts) != 3:
        raise ValueError("selection_counts must be a list of length 3.")
    if weighting_scheme == 'use_category_weights' and len(category_weights) != 3:
        raise ValueError("category_weights must be a list of length 3 when weighting_scheme is 'use_category_weights'.")
    
    valid_methods = ['probability', 'risk_adjusted_probability']
    if sort_by not in valid_methods:
        raise ValueError(f"sort_by must be one of {valid_methods}, got '{sort_by}'")
    valid_weighting = ['use_category_weights', 'equal']
    if weighting_scheme not in valid_weighting:
        raise ValueError(f"weighting_scheme must be one of {valid_weighting}, got '{weighting_scheme}'")
    
    selection_cats = get_dimension_categories(selection_dimension)
    weighting_cats = get_dimension_categories(weighting_dimension)
    
    # Map selection categories to counts
    selection_specs = {cat: cnt for cat, cnt in zip(selection_cats, selection_counts)}
    
    # Map weighting categories to weights (only for use_category_weights)
    weight_map = {}
    if weighting_scheme == 'use_category_weights':
        weight_map = {cat: w for cat, w in zip(weighting_cats, category_weights)}
    
    final_selection = []
    
    for quarter, group in stock_probabilities.groupby('quarter'):
        # Apply probability threshold
        if min_prob_threshold is not None:
            group = group[group['prob'] >= min_prob_threshold]
            if len(group) == 0:
                warnings.warn(f"Quarter {quarter}: No stocks above min_prob_threshold {min_prob_threshold}")
                continue
        
        current_data = group.copy()
        
        # Assign selection dimension categories
        if selection_dimension == 'volatility':
            current_data['selection_cat'] = _assign_volatility_categories(current_data['volatility'])
        else:  # mcap
            current_data['selection_cat'] = current_data['category']
        
        # Assign weighting dimension categories
        if weighting_dimension == 'volatility':
            current_data['weight_cat'] = _assign_volatility_categories(current_data['volatility'])
        else:  # mcap
            current_data['weight_cat'] = current_data['category']
        
        # Calculate risk-adjusted score if needed
        if sort_by == 'risk_adjusted_probability':
            current_data['risk_adj_score'] = current_data['prob'] / (current_data['volatility'] + RISK_ADJUSTED_EPSILON)
        
        # Select stocks per selection-dimension category
        for sel_cat in selection_cats:
            cat_data = current_data[current_data['selection_cat'] == sel_cat]
            
            if sort_by == 'risk_adjusted_probability':
                cat_data = cat_data.sort_values(by='risk_adj_score', ascending=False)
            else:
                cat_data = cat_data.sort_values(by='prob', ascending=False)
            
            target_count = selection_specs[sel_cat]
            available_count = len(cat_data)
            
            if available_count < target_count:
                warnings.warn(
                    f"Quarter {quarter}, Category {sel_cat}: Requested {target_count} stocks, "
                    f"but only {available_count} available."
                )
            
            selected = cat_data.head(target_count).copy()
            
            # Assign cat_weight based on each stock's weighting-dimension category
            if weighting_scheme == 'use_category_weights':
                selected['cat_weight'] = selected['weight_cat'].map(weight_map)
            
            final_selection.append(selected)
    
    # Concatenate and format output
    if final_selection:
        result_df = pd.concat(final_selection)
        
        # the 'cat' column is used in downstream in equity curve breakdown
        # breakdown entails the value of categories on a daily basis
        if weighting_scheme == 'use_category_weights':
            result_df['cat'] = result_df['weight_cat']
        elif weighting_scheme == 'equal':
            result_df['cat'] = result_df['selection_cat']
        
        if weighting_scheme == 'equal':
            # Equal weighting: each stock gets 1/n of the portfolio
            result_df['stock_weight'] = result_df.groupby('quarter')['co_name'].transform(
                lambda x: 1.0 / len(x)
            )
        elif weighting_scheme == 'use_category_weights':
            # Category-based weighting: divide cat_weight by number of stocks in that category per quarter
            # Group by quarter and cat, then divide cat_weight by count (use transform for Series alignment)
            result_df['stock_weight'] = result_df.groupby(['quarter', 'cat'])['cat_weight'].transform(
                lambda x: x / len(x)
            )
        
        base_cols = ['quarter', 'co_name', 'cat', 'selection_cat', 'weight_cat', 'stock_weight']
        # Include cat_weight for informational purposes when using category weights
        if weighting_scheme == 'use_category_weights':
            base_cols.append('cat_weight')
        if sort_by == 'risk_adjusted_probability':
            base_cols.append('risk_adj_score')
        if 'sector' in result_df.columns:
            base_cols.append('sector')
        
        result_df = result_df[base_cols]
        result_df = result_df.reset_index(drop=True)
        return result_df
    else:
        # Return empty dataframe with correct columns
        base_cols = ['quarter', 'co_name', 'cat', 'selection_cat', 'weight_cat', 'stock_weight']
        if weighting_scheme == 'use_category_weights':
            base_cols.append('cat_weight')
        if sort_by == 'risk_adjusted_probability':
            base_cols.append('risk_adj_score')
        if 'sector' in stock_probabilities.columns:
            base_cols.append('sector')
        return pd.DataFrame(columns=base_cols)


# =============================================================================
# PRICE DATA VALIDATION FUNCTIONS
# to answer the question:
# do we have probabilities for some stocks in some quarter, 
# but problematic price data to trade that stock in that quarter?
# =============================================================================


def validate_price_data_coverage_full(input_data, price_data, first_quarter=None, last_quarter=None):
    """
    Flags stocks part of inference data that have incomplete price data.
    Enforces strict 100% coverage for the quarter to prevent downstream ValueError crashes.
    """
    price_data = price_data.copy()
    price_data['date'] = pd.to_datetime(price_data['date'])
    
    if first_quarter is None:
        first_quarter = input_data['quarter'].min()
    if last_quarter is None:
        last_quarter = input_data['quarter'].max()
        
    input_data = input_data[
        (input_data['quarter'] >= first_quarter) & 
        (input_data['quarter'] <= last_quarter)
    ]
    
    issues = []
    
    # Process input_data quarter by quarter
    for quarter, group in input_data.groupby('quarter'):
        q_start, q_end = get_quarter_dates(quarter) 
        
        # Slice price data for the quarter
        q_prices = price_data[
            (price_data['date'] >= q_start) & 
            (price_data['date'] <= q_end)
        ]
        
        # Get actual trading days in this quarter for the WHOLE market
        q_trading_days = np.sort(q_prices['date'].unique())
        total_q_days = len(q_trading_days)
        
        if total_q_days == 0:
            raise ValueError(f"No trading days in {quarter}")
            
        # Vectorized counting
        full_q_counts = q_prices.groupby('co_name')['close'].count()

        # Strict 100% validation check
        for stock in group['co_name'].unique():
            q_count = full_q_counts.get(stock, 0)
            
            # STRICT CHECK: Stock must have traded every single day the market was open
            if q_count != total_q_days:
                missing_days = total_q_days - q_count
                issues.append({
                    'quarter': quarter,
                    'co_name': stock,
                    'issue': 'missing_trading_days',
                    'detail': f'Missing {missing_days} day(s) of price data. Found {q_count}/{total_q_days} required market days.'
                })

    return pd.DataFrame(issues)


def filter_tradeable_stocks(input_data, price_data, first_quarter=None, last_quarter=None):
    """
    Returns:
        Tuple of (filtered_input_data, issues_df)
    """
    if first_quarter is None:
        first_quarter = input_data['quarter'].min()
    if last_quarter is None:
        last_quarter = input_data['quarter'].max()
        
    input_data = input_data[
        (input_data['quarter'] >= first_quarter) & 
        (input_data['quarter'] <= last_quarter)
    ].copy()

    # Get issues
    issues_df = validate_price_data_coverage_full(input_data, price_data, first_quarter, last_quarter)
    
    if issues_df.empty:
        return input_data, issues_df
    
    # Create a MultiIndex of the invalid pairs
    invalid_index = pd.MultiIndex.from_frame(issues_df[['quarter', 'co_name']])
    
    # Create a MultiIndex of the input data
    input_index = pd.MultiIndex.from_frame(input_data[['quarter', 'co_name']])
    
    # Filter by checking which input rows are NOT in the invalid index
    filtered_data = input_data[~input_index.isin(invalid_index)].copy()
    
    return filtered_data, issues_df