"""
Backtesting Strategy - Main Wrapper Script

This script orchestrates the full backtesting workflow:
1. Load configuration and data
2. Run stock selection
3. Simulate trades with TP/SL
4. Generate daily portfolio values
5. Compare with benchmark index
6. Save all outputs with traceability
"""

import pandas as pd
import yaml
import os
import sys
import warnings
from datetime import datetime
from io import StringIO

# Import from local packages
from selection import (
    select_and_weight_stocks_volatility, 
    select_and_weight_stocks_mcap,
    select_top_k_stocks,
    filter_tradeable_stocks,
    validate_price_data_coverage,
)
from backtest import (
    simulate_trades,
    compute_pf_value_over_quarters,
    compute_pf_vs_index,
)
from reporting import generate_backtest_report
from config.defaults import (
    INITIAL_CAPITAL,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_BASE_DIR,
    DEFAULT_CATEGORY_SCHEME,
    DEFAULT_RUN_STOCK_SELECTION,
    DEFAULT_SELECTION_TYPE,
    DEFAULT_CATEGORY_COUNTS,
    DEFAULT_CATEGORY_WEIGHTS,
    DEFAULT_SELECTION_METHOD,
    DEFAULT_MIN_PROB_THRESHOLD,
    DEFAULT_WEIGHTING_SCHEME,
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_WEIGHTING,
    DEFAULT_TP_MODE,
    DEFAULT_SL_MODE,
    DEFAULT_TP_ENABLED,
    DEFAULT_SL_ENABLED,
    DEFAULT_ENTRY_PRICE_WINDOW,
    DEFAULT_TPSL_FALLBACK_PCT,
    DEFAULT_GENERATE_REPORT,
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots


class TeeOutput:
    """
    A class that duplicates stdout to both console and a StringIO buffer.
    This allows capturing all print statements while still displaying them.
    """
    def __init__(self):
        self.terminal = sys.stdout
        self.buffer = StringIO()
        
    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)
        
    def flush(self):
        self.terminal.flush()
        
    def get_log_content(self):
        return self.buffer.getvalue()
    
    def save_to_file(self, filepath):
        """Save captured output to a file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.buffer.getvalue())


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_data(file_path):
    """
    Load data from CSV or Parquet file.
    
    Parameters:
    - file_path: Path to the data file (csv or parquet)
    
    Returns:
    - pd.DataFrame
    """
    if not file_path:
        raise ValueError(f"File path is empty or not specified.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .csv or .parquet")


def filter_data_by_quarters(df, first_quarter, last_quarter):
    """
    Filter dataframe to include only rows within the specified quarter range.
    
    Parameters:
    - df: DataFrame with 'quarter' column
    - first_quarter: Start quarter (inclusive), e.g., 202202
    - last_quarter: End quarter (inclusive), e.g., 202402
    
    Returns:
    - Filtered DataFrame
    """
    return df[(df['quarter'] >= first_quarter) & (df['quarter'] <= last_quarter)].copy()


def create_output_directory(base_dir=DEFAULT_OUTPUT_BASE_DIR):
    """
    Create a timestamped output directory for results.
    
    Parameters:
    - base_dir: Base directory for backtesting results
    
    Returns:
    - Path to the created directory
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(output_dir)
    
    return output_dir


def save_config_copy(config, output_dir):
    """
    Save a copy of the configuration to the output directory for traceability.
    
    Parameters:
    - config: Configuration dictionary
    - output_dir: Output directory path
    """
    # Save as YAML
    config_output_path = os.path.join(output_dir, 'config_used.yaml')
    with open(config_output_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    print(f"Configuration saved to: {config_output_path}")


def run_stock_selection(input_data, category_scheme, category_counts, category_weights,
                        selection_method=DEFAULT_SELECTION_METHOD, min_prob_threshold=DEFAULT_MIN_PROB_THRESHOLD,
                        selection_type=DEFAULT_SELECTION_TYPE, top_k_config=None,
                        category_based_weighting_scheme=DEFAULT_WEIGHTING_SCHEME):
    """
    Run stock selection based on the specified selection type and category scheme.
    
    Parameters:
    - input_data: DataFrame with stock data
    - category_scheme: 'volatility' or 'mcap' (used for category_based selection)
    - category_counts: [n1, n2, n3] stocks to select per category (category_based only)
    - category_weights: [w1, w2, w3] weights per category (category_based only,
      ignored when category_based_weighting_scheme is 'equal')
    - selection_method: 'probability' (default) or 'risk_adjusted'
    - min_prob_threshold: Minimum probability to consider (None = no filter)
    - selection_type: 'category_based' (default) or 'top_k'
    - top_k_config: dict with 'k' and 'weighting_scheme' (top_k only)
    - category_based_weighting_scheme: 'use_category_weights' (default) or 'equal'
      Controls how capital is allocated for category_based selection.
    
    Returns:
    - DataFrame with selected stocks
      - category_based (use_category_weights): [quarter, co_name, cat, cat_weight]
      - category_based (equal): [quarter, co_name, cat, stock_weight]
      - top_k: [quarter, co_name, stock_weight] (+ cat if category in input)
    """
    if selection_type == 'top_k':
        if top_k_config is None:
            top_k_config = {'k': DEFAULT_TOP_K, 'weighting_scheme': DEFAULT_TOP_K_WEIGHTING}
        
        return select_top_k_stocks(
            input_data,
            k=top_k_config.get('k', DEFAULT_TOP_K),
            selection_method=selection_method,
            min_prob_threshold=min_prob_threshold,
            weighting_scheme=top_k_config.get('weighting_scheme', DEFAULT_TOP_K_WEIGHTING)
        )
    elif selection_type == 'category_based':
        if category_scheme == 'volatility':
            return select_and_weight_stocks_volatility(
                input_data, 
                selection_counts=category_counts, 
                category_weights=category_weights,
                selection_method=selection_method,
                min_prob_threshold=min_prob_threshold,
                weighting_scheme=category_based_weighting_scheme
            )
        elif category_scheme == 'mcap':
            return select_and_weight_stocks_mcap(
                input_data, 
                lms_count=category_counts, 
                lms_w=category_weights,
                selection_method=selection_method,
                min_prob_threshold=min_prob_threshold,
                weighting_scheme=category_based_weighting_scheme
            )
        else:
            raise ValueError(f"Unknown category_scheme: {category_scheme}. Use 'volatility' or 'mcap'.")
    else:
        raise ValueError(f"Unknown selection_type: {selection_type}. Use 'category_based' or 'top_k'.")


def validate_preselected_input(df, category_scheme, tp_mode, sl_mode, tp_enabled, sl_enabled, has_default_tpsl):
    """
    Validate and prepare preselected portfolio input.
    
    Parameters:
    - df: Input DataFrame (should have quarter, co_name, stock_weight, and optionally category)
    - category_scheme: 'volatility' or 'mcap' (used for category validation if present)
    - tp_mode: Mode for take profit - 'fixed', 'flat', 'atr', or 'pivot'
    - sl_mode: Mode for stop loss - 'fixed', 'flat', 'atr', or 'pivot'
    - tp_enabled: Whether take profit is enabled
    - sl_enabled: Whether stop loss is enabled
    - has_default_tpsl: Whether default_tpsl config is present
    
    Returns:
    - Tuple of (prepared_df, has_category)
    """
    required_cols = {'quarter', 'co_name', 'stock_weight'}
    
    # Check required columns
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for preselected input: {missing}")
    
    # Check for category column
    has_category = 'category' in df.columns
    
    # Validate: if no category and a 'fixed' mode is used with TP/SL enabled, need default_tpsl
    # Note: 'flat', 'atr', and 'pivot' modes don't require category
    needs_fixed_category = (tp_enabled and tp_mode == 'fixed') or (sl_enabled and sl_mode == 'fixed')
    if not has_category and needs_fixed_category:
        if not has_default_tpsl:
            raise ValueError(
                "No 'category' column in preselected input and tp_mode/sl_mode is 'fixed'. "
                "Either:\n"
                "  1. Add 'category' column to input data, or\n"
                "  2. Add 'default_tpsl' section to config, or\n"
                "  3. Set tp_mode/sl_mode to 'flat', 'atr', or 'pivot', or\n"
                "  4. Set both tp_enabled and sl_enabled to false"
            )
    
    # Validate category values if present
    if has_category:
        valid_categories = {
            'volatility': {'high_volatility', 'medium_volatility', 'low_volatility'},
            'mcap': {'largecap', 'midcap', 'smallcap'}
        }
        unique_cats = set(df['category'].unique())
        invalid_cats = unique_cats - valid_categories.get(category_scheme, set())
        if invalid_cats:
            warnings.warn(
                f"Found category values not matching '{category_scheme}' scheme: {invalid_cats}. "
                f"Expected values: {valid_categories.get(category_scheme)}",
                UserWarning
            )
    
    # Check weights sum per quarter (warning only)
    for quarter in df['quarter'].unique():
        quarter_weights = df[df['quarter'] == quarter]['stock_weight'].sum()
        if abs(quarter_weights - 1.0) > 0.01:
            warnings.warn(
                f"Quarter {quarter}: stock weights sum to {quarter_weights:.4f}, not 1.0. "
                f"This may be intentional (e.g., partial allocation), but verify if unexpected.",
                UserWarning
            )
    
    # Prepare output: rename 'category' to 'cat' for pipeline compatibility
    result = df.copy()
    if has_category:
        result = result.rename(columns={'category': 'cat'})
    
    return result, has_category


def backtest_core(
        config: dict, 
        input_data: pd.DataFrame, 
        price_data: pd.DataFrame, 
        index_data: pd.DataFrame = None,
        verbose: bool = False,
        skip_price_data_validation: bool = True) -> dict:
    """
    Core backtesting logic.
    
    Parameters:
    - config: Configuration dictionary (same structure as YAML config)
    - input_data: DataFrame with stock data 
    - price_data: DataFrame with OHLCV price data
    - index_data: DataFrame with index data (optional)
    - verbose: If True, print progress messages (default: False)
    - skip_price_data_validation: If True, skip price data validation (use when data is pre-validated)
    
    Returns:
    - Dictionary with:
        - 'daily_pf_values': DataFrame with daily portfolio values
        - 'trade_results': DataFrame with trade-level results
        - 'first_quarter': First quarter used
        - 'last_quarter': Last quarter used
        and optionally,
        - 'data_issues': Price data validation issues (None if no issues)
      Returns None if backtest fails
    """
    try:
        # Extract config values
        first_quarter = config.get('first_quarter')
        last_quarter = config.get('last_quarter')
        category_scheme = config.get('category_scheme', DEFAULT_CATEGORY_SCHEME)
        
        # Stock selection mode
        run_stock_selection_flag = config.get('run_stock_selection', DEFAULT_RUN_STOCK_SELECTION)
        
        # Selection type: 'category_based' or 'top_k'
        selection_type = config.get('selection_type', DEFAULT_SELECTION_TYPE)
        
        # Selection-specific config
        category_counts = config.get('category_counts', DEFAULT_CATEGORY_COUNTS)
        category_weights = config.get('category_weights', DEFAULT_CATEGORY_WEIGHTS)
        selection_method = config.get('selection_method', DEFAULT_SELECTION_METHOD)
        min_prob_threshold = config.get('min_prob_threshold', DEFAULT_MIN_PROB_THRESHOLD)
        top_k_config = config.get('top_k_config', None)
        category_based_weighting_scheme = config.get('category_based_selection_weighting_scheme', DEFAULT_WEIGHTING_SCHEME)
        
        # Default TP/SL config
        default_tpsl = config.get('default_tpsl', None)
        has_default_tpsl = default_tpsl is not None
        
        # TP/SL mode configuration
        tp_mode = config.get('tp_mode', DEFAULT_TP_MODE)
        sl_mode = config.get('sl_mode', DEFAULT_SL_MODE)
        tp_enabled = config.get('tp_enabled', DEFAULT_TP_ENABLED)
        sl_enabled = config.get('sl_enabled', DEFAULT_SL_ENABLED)
        
        # Entry price window configuration
        entry_price_window = config.get('entry_price_window', DEFAULT_ENTRY_PRICE_WINDOW)
        if not isinstance(entry_price_window, int) or entry_price_window < 1:
            raise ValueError(f"entry_price_window must be an integer >= 1, got: {entry_price_window}")
        
        # ---------------------------------------------------------------------
        # Filter Input Data by Quarter Range
        # ---------------------------------------------------------------------
        if verbose:
            print("\n[1/4] Filtering data by quarter range...")
        
        if first_quarter is None and last_quarter is None:
            input_data_filtered = input_data.copy()
            first_quarter = int(input_data_filtered['quarter'].min())
            last_quarter = int(input_data_filtered['quarter'].max())
            if verbose:
                print(f"  - Using full input data: {len(input_data_filtered)} rows")
                print(f"  - Quarters in data: {sorted(input_data_filtered['quarter'].unique().tolist())}")
        else:
            input_data_filtered = filter_data_by_quarters(input_data, first_quarter, last_quarter)
            if verbose:
                print(f"  - Filtered input data: {len(input_data_filtered)} rows")
                print(f"  - Quarters in data: {sorted(input_data_filtered['quarter'].unique().tolist())}")
        
        if input_data_filtered.empty:
            return None
        
        # ---------------------------------------------------------------------
        # Run Stock Selection OR Use Preselected Portfolio
        # ---------------------------------------------------------------------
        data_issues = None  # Will store validation issues for return
        
        if run_stock_selection_flag:
            if verbose:
                print("\n[2/4] Running stock selection...")
            
            # Validate and filter price data before selection (unless pre-validated)
            if not skip_price_data_validation:
                input_data_filtered, data_issues = filter_tradeable_stocks(
                    input_data_filtered,
                    price_data,
                    first_quarter,
                    last_quarter,
                    min_prices_required=entry_price_window
                )
                
                if data_issues is not None and not data_issues.empty:
                    if verbose:
                        print(f"  - Filtered {len(data_issues)} stock-quarter combinations with price data issues")
            
            selected_stocks = run_stock_selection(
                input_data_filtered,
                category_scheme,
                category_counts,
                category_weights,
                selection_method=selection_method,
                min_prob_threshold=min_prob_threshold,
                selection_type=selection_type,
                top_k_config=top_k_config,
                category_based_weighting_scheme=category_based_weighting_scheme
            )
            if verbose:
                print(f"  - Selected stocks: {len(selected_stocks)} positions across all quarters")
        else:
            if verbose:
                print("\n[2/4] Using preselected portfolio...")
            
            # Validate price data coverage (but don't filter, since it's preselected)
            if not skip_price_data_validation:
                data_issues = validate_price_data_coverage(
                    input_data_filtered,
                    price_data,
                    first_quarter,
                    last_quarter,
                    min_prices_required=entry_price_window
                )
                
                if data_issues is not None and not data_issues.empty:
                    if verbose:
                        print(f"  - Warning: Found {len(data_issues)} stock-quarter combinations with price data issues")
                        print(f"            (Issues logged but stocks not filtered since using preselected portfolio)")
            
            selected_stocks, has_category = validate_preselected_input(
                input_data_filtered,
                category_scheme,
                tp_mode,
                sl_mode,
                tp_enabled,
                sl_enabled,
                has_default_tpsl
            )
            if verbose:
                print(f"  - Preselected stocks: {len(selected_stocks)} positions across all quarters")
                if has_category:
                    print(f"  - Categories found: {sorted(selected_stocks['cat'].unique().tolist())}")
                else:
                    print("  - No category column in input (using default TP/SL thresholds)")
        
        if selected_stocks is None or selected_stocks.empty:
            return None
        
        # ---------------------------------------------------------------------
        # Simulate Trades with TP/SL
        # ---------------------------------------------------------------------
        if verbose:
            print("\n[3/4] Simulating trades with TP/SL thresholds...")
        
        # Get configs from strategy config
        trade_results = simulate_trades(
            selected_stocks,
            price_data,
            category_scheme,
            index_data=index_data,
            tp_mode=tp_mode,
            sl_mode=sl_mode,
            tp_enabled=tp_enabled,
            sl_enabled=sl_enabled,
            tp_config=config.get('TP_CONFIG'),
            sl_config=config.get('SL_CONFIG'),
            default_tpsl=config.get('default_tpsl'),
            atr_config=config.get('atr_config'),
            pivot_config=config.get('pivot_config'),
            flat_config=config.get('flat_config'),
            index_exit_config=config.get('index_exit'),
            entry_price_window=entry_price_window
        )
        
        if trade_results is None or trade_results.empty:
            return None
        
        if verbose:
            print(f"  - Trade simulation complete: {len(trade_results)} trades")
        
        # ---------------------------------------------------------------------
        # Generate Daily Portfolio Values
        # ---------------------------------------------------------------------
        if verbose:
            print("\n[4/4] Generating daily portfolio values...")
        
        daily_pf_values = compute_pf_value_over_quarters(
            trade_results, 
            price_data, 
            first_quarter, 
            last_quarter, 
            INITIAL_CAPITAL,
            entry_price_window=entry_price_window
        )
        
        if verbose and daily_pf_values is not None and not daily_pf_values.empty:
            print(f"  - Daily portfolio values generated: {len(daily_pf_values)} days")
        
        return {
            'daily_pf_values': daily_pf_values,
            'trade_results': trade_results,
            'first_quarter': first_quarter,
            'last_quarter': last_quarter,
            'data_issues': data_issues,  # Price data validation issues (None if no issues)
        }
        
    except Exception as e:
        warnings.warn(f"Backtest core failed: {str(e)}")
        return None


def run_backtest(config_path=DEFAULT_CONFIG_PATH):
    """
    Main function to run the full backtesting workflow.
    
    Parameters:
    - config_path: Path to the configuration YAML file
    """
    # Start capturing output for log file
    tee = TeeOutput()
    sys.stdout = tee
    
    # Record start time
    start_time = datetime.now()
    
    print("=" * 60)
    print("BACKTESTING STRATEGY - STARTING")
    print("=" * 60)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # -------------------------------------------------------------------------
    # 1. Load Configuration
    # -------------------------------------------------------------------------
    print("\n[1/2] Loading configuration...")
    config = load_config(config_path)
    
    # Extract config values
    input_data_path = config['input_data_path']
    price_data_path = config['price_data_path']
    index_data_path = config['index_data_path']
    first_quarter = config['first_quarter']
    last_quarter = config['last_quarter']
    category_scheme = config['category_scheme']
    
    # Stock selection mode (new: can skip selection for preselected portfolios)
    run_stock_selection_flag = config.get('run_stock_selection', DEFAULT_RUN_STOCK_SELECTION)
    
    # Selection type: 'category_based' or 'top_k'
    selection_type = config.get('selection_type', DEFAULT_SELECTION_TYPE)
    
    # Selection-specific config (only used when run_stock_selection is True)
    category_counts = config.get('category_counts', DEFAULT_CATEGORY_COUNTS)
    category_weights = config.get('category_weights', DEFAULT_CATEGORY_WEIGHTS)
    selection_method = config.get('selection_method', DEFAULT_SELECTION_METHOD)
    min_prob_threshold = config.get('min_prob_threshold', DEFAULT_MIN_PROB_THRESHOLD)
    top_k_config = config.get('top_k_config', None)
    category_based_weighting_scheme = config.get('category_based_selection_weighting_scheme', DEFAULT_WEIGHTING_SCHEME)
    
    # Default TP/SL config (used when no category in preselected mode)
    default_tpsl = config.get('default_tpsl', None)
    has_default_tpsl = default_tpsl is not None
    
    # Analysis report options
    generate_report = config.get('generate_report', DEFAULT_GENERATE_REPORT)
    report_sub_periods = config.get('report_sub_periods', None)
    
    print(f"  - Input data: {input_data_path}")
    print(f"  - Price data: {price_data_path}")
    print(f"  - Index data: {index_data_path}")
    print(f"  - Quarter range: {first_quarter} to {last_quarter}")
    print(f"  - Category scheme: {category_scheme}")
    print(f"  - Run stock selection: {run_stock_selection_flag}")
    
    if run_stock_selection_flag:
        print(f"  - Selection type: {selection_type}")
        if selection_type == 'category_based':
            print(f"  - Category counts: {category_counts}")
            print(f"  - Category weights: {category_weights}")
            print(f"  - Weighting scheme: {category_based_weighting_scheme}")
        elif selection_type == 'top_k':
            k = top_k_config.get('k', DEFAULT_TOP_K) if top_k_config else DEFAULT_TOP_K
            weighting = top_k_config.get('weighting_scheme', DEFAULT_TOP_K_WEIGHTING) if top_k_config else DEFAULT_TOP_K_WEIGHTING
            print(f"  - Top k: {k}")
            print(f"  - Weighting scheme: {weighting}")
        print(f"  - Selection method: {selection_method}")
        if min_prob_threshold is not None:
            print(f"  - Min probability threshold: {min_prob_threshold}")
    else:
        print("  - Using preselected portfolio (selection config options ignored)")
        if has_default_tpsl:
            print(f"  - Default TP/SL: TP={default_tpsl.get('tp_pct', DEFAULT_TPSL_FALLBACK_PCT):.1%}, SL={default_tpsl.get('sl_pct', DEFAULT_TPSL_FALLBACK_PCT):.1%}")
    
    # TP/SL mode configuration
    tp_mode = config.get('tp_mode', DEFAULT_TP_MODE)
    sl_mode = config.get('sl_mode', DEFAULT_SL_MODE)
    tp_enabled = config.get('tp_enabled', DEFAULT_TP_ENABLED)
    sl_enabled = config.get('sl_enabled', DEFAULT_SL_ENABLED)
    print(f"  - TP mode: {tp_mode}" + (" (take profit exits disabled)" if not tp_enabled else ""))
    print(f"  - SL mode: {sl_mode}" + (" (stop loss exits disabled)" if not sl_enabled else ""))
    print(f"  - TP enabled: {tp_enabled}")
    print(f"  - SL enabled: {sl_enabled}")
    
    # Entry price window
    entry_price_window = config.get('entry_price_window', DEFAULT_ENTRY_PRICE_WINDOW)
    print(f"  - Entry price window: {entry_price_window} trading day(s)")
    
    # -------------------------------------------------------------------------
    # 2. Load Data
    # -------------------------------------------------------------------------
    print("\n[2/2] Loading data...")
    
    input_data = load_data(input_data_path)
    print(f"  - Input data loaded: {len(input_data)} rows")
    
    price_data = load_data(price_data_path)
    print(f"  - Price data loaded: {len(price_data)} rows")
    
    index_data = load_data(index_data_path)
    print(f"  - Index data loaded: {len(index_data)} rows")
    
    # -------------------------------------------------------------------------
    # 3. Run Backtest Core
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("RUNNING BACKTEST CORE")
    print("-" * 60)
    
    results = backtest_core(
        config=config,
        input_data=input_data,
        price_data=price_data,
        index_data=index_data,
        verbose=True
    )
    
    if results is None:
        print("\nERROR: Backtest core returned no results.")
        sys.stdout = tee.terminal
        return None, None, None
    
    # Extract results
    trade_results = results['trade_results']
    equity_curve = results['daily_pf_values']
    first_quarter = results['first_quarter']
    last_quarter = results['last_quarter']
    data_issues = results.get('data_issues')
    
    print("-" * 60)
    
    # -------------------------------------------------------------------------
    # 4. Create Output Directory and Save Results
    # -------------------------------------------------------------------------
    print("\nSaving results...")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"  - Output directory: {output_dir}")
    
    # Save config for traceability
    save_config_copy(config, output_dir)
    
    # -------------------------------------------------------------------------
    # 5. Compute comparison data (if index data available)
    # -------------------------------------------------------------------------
    comparison_df = None
    if index_data_path:
        comparison_df = compute_pf_vs_index(
            trade_results,
            price_data,
            index_data,
            first_quarter,
            last_quarter,
            INITIAL_CAPITAL,
            daily_pf_values=equity_curve  # Reuse pre-computed equity curve for consistency
        )
    else:
        print("  - Skipping index comparison (no index data path specified)")
    
    # -------------------------------------------------------------------------
    # 6. Generate Consolidated Report
    # -------------------------------------------------------------------------
    if generate_report and equity_curve is not None and not equity_curve.empty:
        # Prepare daily_pf in the expected format
        daily_pf = equity_curve.reset_index()
        daily_pf.columns = ['date', 'portfolio_value', 'quarter']
        
        report_path = os.path.join(output_dir, f'backtest_report.xlsx')
        generate_backtest_report(
            daily_pf=daily_pf,
            trade_results=trade_results,
            comparison_df=comparison_df,
            output_path=report_path,
            sub_periods=report_sub_periods,
            input_frequency="daily",
            report_title="Backtest Report",
            data_issues=data_issues,
            first_quarter=first_quarter,
            last_quarter=last_quarter,
        )
    elif generate_report:
        print("\nSkipping report (no equity curve data available)")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BACKTESTING COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFiles generated:")
    print(f"  - config_used.yaml (configuration traceability)")
    if generate_report and equity_curve is not None and not equity_curve.empty:
        print(f"  - backtest_report.xlsx (metrics, charts, trade data — all in one)")
    print(f"  - backtest_log.txt (full execution log)")
    
    # Calculate and print execution time
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    total_seconds = elapsed_time.total_seconds()
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if hours > 0:
        print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    elif minutes > 0:
        print(f"Total execution time: {int(minutes)}m {seconds:.2f}s")
    else:
        print(f"Total execution time: {seconds:.2f}s")
    
    # Save log file and restore stdout
    log_path = os.path.join(output_dir, 'backtest_log.txt')
    tee.save_to_file(log_path)
    sys.stdout = tee.terminal
    
    return output_dir, trade_results, equity_curve


if __name__ == '__main__':
    # Run the backtest
    output_dir, portfolio_results, equity_curve = run_backtest()
