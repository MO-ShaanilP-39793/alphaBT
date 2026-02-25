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

import logging
import pandas as pd
import yaml
import os
import warnings
from datetime import datetime

# Import from local packages
from selection import (
    select_and_weight_stocks,
    select_top_k_stocks,
    filter_tradeable_stocks,
    validate_price_data_coverage_full,
)
from backtest import (
    simulate_trades,
    compute_portfolio_value_over_quarters,
    compute_portfolio_vs_index,
)
from reporting import generate_backtest_report
from config.defaults import (
    INITIAL_CAPITAL,
    RISK_FREE_RATE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_BASE_DIR,
    DEFAULT_SELECTION_TYPE,
    DEFAULT_SORT_BY,
    DEFAULT_MIN_PROB_THRESHOLD,
    DEFAULT_WEIGHTING_SCHEME,
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_WEIGHTING,
)
from typing import Optional

from config.schema import BacktestConfig
from utils.logging_config import setup_logging, get_logger
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

logger = get_logger(__name__)


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> BacktestConfig:
    """Load and validate configuration from YAML file.

    Returns a validated :class:`BacktestConfig` instance.  Any typos or type
    errors in the YAML will surface as ``pydantic.ValidationError`` here.
    """
    with open(config_path, 'r') as file:
        raw = yaml.safe_load(file)
    return BacktestConfig(**raw)


def load_data(file_path: str) -> pd.DataFrame:
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


def filter_data_by_quarters(df: pd.DataFrame, first_quarter: int, last_quarter: int) -> pd.DataFrame:
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


def create_output_directory(base_dir: str = DEFAULT_OUTPUT_BASE_DIR) -> str:
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


def save_config_copy(config: 'BacktestConfig', output_dir: str) -> None:
    """
    Save a copy of the configuration to the output directory for traceability.
    
    Parameters:
    - config: BacktestConfig or dict
    - output_dir: Output directory path
    """
    config_data = config.model_dump() if isinstance(config, BacktestConfig) else config
    config_output_path = os.path.join(output_dir, 'config_used.yaml')
    with open(config_output_path, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)
    
    logger.info("Configuration saved to: %s", config_output_path)


def run_stock_selection(input_data: pd.DataFrame, category_scheme: str, category_counts: list[int], category_weights: list[float],
                        sort_by: str = DEFAULT_SORT_BY, min_prob_threshold: Optional[float] = DEFAULT_MIN_PROB_THRESHOLD,
                        selection_type: str = DEFAULT_SELECTION_TYPE, top_k_config: Optional[dict] = None,
                        category_based_weighting_scheme: str = DEFAULT_WEIGHTING_SCHEME,
                        selection_dimension: Optional[str] = None, weighting_dimension: Optional[str] = None) -> pd.DataFrame:
    """
    Run stock selection based on the specified selection type and category scheme.
    
    Parameters:
    - input_data: DataFrame with stock data
    - category_scheme: 'volatility' or 'mcap' (legacy; used as fallback for dimensions)
    - category_counts: [n1, n2, n3] stocks to select per category (category_based only)
    - category_weights: [w1, w2, w3] weights per category (category_based only,
      ignored when category_based_weighting_scheme is 'equal')
    - sort_by: 'probability' (default) or 'risk_adjusted_probability'
    - min_prob_threshold: Minimum probability to consider (None = no filter)
    - selection_type: 'category_based' (default) or 'top_k'
    - top_k_config: dict with 'k' and 'weighting_scheme' (top_k only)
    - category_based_weighting_scheme: 'use_category_weights' (default) or 'equal'
      Controls how capital is allocated for category_based selection.
    - selection_dimension: 'volatility' or 'mcap' — axis to bucket/select by.
      Defaults to category_scheme if None.
    - weighting_dimension: 'volatility' or 'mcap' — axis to assign weights by.
      Defaults to category_scheme if None.
    
    Returns:
    - DataFrame with selected stocks
      - category_based: [quarter, co_name, cat, selection_cat, weight_cat, stock_weight] (+ cat_weight if weighting_scheme='use_category_weights')
      - top_k: [quarter, co_name, stock_weight] (+ cat if category in input)
    """
    if selection_type == 'top_k':
        if top_k_config is None:
            top_k_config = {'k': DEFAULT_TOP_K, 'weighting_scheme': DEFAULT_TOP_K_WEIGHTING}
        
        return select_top_k_stocks(
            input_data,
            k=top_k_config.get('k', DEFAULT_TOP_K),
            sort_by=sort_by,
            min_prob_threshold=min_prob_threshold,
            weighting_scheme=top_k_config.get('weighting_scheme', DEFAULT_TOP_K_WEIGHTING)
        )
    elif selection_type == 'category_based':
        # Resolve dimensions (backward compatible: fall back to category_scheme)
        sel_dim = selection_dimension if selection_dimension is not None else category_scheme
        wgt_dim = weighting_dimension if weighting_dimension is not None else category_scheme
        
        return select_and_weight_stocks(
            input_data,
            selection_dimension=sel_dim,
            weighting_dimension=wgt_dim,
            selection_counts=category_counts,
            category_weights=category_weights,
            sort_by=sort_by,
            min_prob_threshold=min_prob_threshold,
            weighting_scheme=category_based_weighting_scheme,
        )
    else:
        raise ValueError(f"Unknown selection_type: {selection_type}. Use 'category_based' or 'top_k'.")


def validate_preselected_input(df: pd.DataFrame, category_scheme: str, tp_mode: str, sl_mode: str, tp_enabled: bool, sl_enabled: bool, has_tiered_config: bool) -> tuple[pd.DataFrame, bool]:
    """
    Validate and prepare preselected portfolio input.
    
    Parameters:
    - df: Input DataFrame (should have quarter, co_name, stock_weight, and optionally category)
    - category_scheme: 'volatility' or 'mcap' (used for category validation if present)
    - tp_mode: Mode for take profit - 'tiered', 'flat', 'atr', or 'pivot'
    - sl_mode: Mode for stop loss - 'tiered', 'flat', 'atr', or 'pivot'
    - tp_enabled: Whether take profit is enabled
    - sl_enabled: Whether stop loss is enabled
    - has_tiered_config: Whether tiered_config is present in config
    
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
    
    # Validate: if no category and 'tiered' mode is used with TP/SL enabled, need tiered_config
    # Note: 'flat', 'atr', and 'pivot' modes don't require category
    needs_tiered_category = (tp_enabled and tp_mode == 'tiered') or (sl_enabled and sl_mode == 'tiered')
    if not has_category and needs_tiered_category:
        if not has_tiered_config:
            raise ValueError(
                "No 'category' column in preselected input and tp_mode/sl_mode is 'tiered'. "
                "Either:\n"
                "  1. Add 'category' column to input data, or\n"
                "  2. Add 'tiered_config' section to config (with default_tp_pct/default_sl_pct), or\n"
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
        config, 
        input_data: pd.DataFrame, 
        price_data: pd.DataFrame, 
        index_data: pd.DataFrame = None,
        skip_price_data_validation: bool = True) -> Optional[dict]:
    """
    Core backtesting logic.
    
    Parameters:
    - config: BacktestConfig or plain dict (backward compatible with tuning pipeline)
    - input_data: DataFrame with stock data 
    - price_data: DataFrame with OHLCV price data
    - index_data: DataFrame with index data (optional)
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
    # Adapter: accept both BacktestConfig and plain dict
    if isinstance(config, dict):
        config = BacktestConfig(**config)

    try:
        return _backtest_core_impl(config, input_data, price_data, index_data, skip_price_data_validation)
    except Exception as e:
        logger.error("Backtest core failed: %s", e)
        warnings.warn(f"Backtest core failed: {e}", UserWarning)
        return None


def _resolve_tpsl_scheme(config: 'BacktestConfig') -> str:
    """Resolve which dimension scheme (volatility/mcap) drives TP/SL thresholds.
    
    Parameters:
        config: Validated backtest configuration
        
    Returns:
        Dimension name ('volatility' or 'mcap')
        
    Raises:
        ValueError: If tpsl_category_dimension is invalid or unresolvable
    """
    selection_dimension = config.selection_dimension
    weighting_dimension = config.weighting_dimension
    tpsl_category_dimension = config.tpsl_category_dimension

    if tpsl_category_dimension == 'selection':
        tpsl_scheme = selection_dimension
    elif tpsl_category_dimension == 'weighting':
        tpsl_scheme = weighting_dimension
    elif tpsl_category_dimension in ('volatility', 'mcap'):
        tpsl_scheme = tpsl_category_dimension
    else:
        raise ValueError(
            f"tpsl_category_dimension must be 'selection', 'weighting', "
            f"'volatility', or 'mcap', got '{tpsl_category_dimension}'"
        )

    if tpsl_scheme not in (selection_dimension, weighting_dimension):
        raise ValueError(
            f"tpsl_category_dimension resolves to '{tpsl_scheme}', but neither "
            f"selection_dimension ('{selection_dimension}') nor weighting_dimension "
            f"('{weighting_dimension}') uses this dimension. Cannot produce "
            f"correct TP/SL category labels."
        )

    return tpsl_scheme


def _select_stocks(
        config: 'BacktestConfig',
        input_data_filtered: pd.DataFrame,
        price_data: pd.DataFrame,
        first_quarter: int,
        last_quarter: int,
        skip_price_data_validation: bool,
) -> tuple:
    """Run stock selection or validate preselected portfolio.
    
    Returns:
        Tuple of (selected_stocks DataFrame, data_issues DataFrame or None)
    """
    run_stock_selection_flag = config.run_stock_selection
    data_issues = None

    if run_stock_selection_flag:
        logger.info("[2/4] Running stock selection...")

        if not skip_price_data_validation:
            input_data_filtered, data_issues = filter_tradeable_stocks(
                input_data_filtered, price_data,
                first_quarter, last_quarter)
            if data_issues is not None and not data_issues.empty:
                logger.debug(
                    "Filtered %d stock-quarter combinations with price data issues",
                    len(data_issues),
                )

        selected_stocks = run_stock_selection(
            input_data_filtered,
            config.category_scheme,
            config.category_counts,
            config.category_weights,
            sort_by=config.sort_by,
            min_prob_threshold=config.min_prob_threshold,
            selection_type=config.selection_type,
            top_k_config=(
                config.top_k_config.model_dump() if config.top_k_config else None
            ),
            category_based_weighting_scheme=config.category_based_selection_weighting_scheme,
            selection_dimension=config.selection_dimension,
            weighting_dimension=config.weighting_dimension,
        )
        logger.debug(
            "Selected stocks: %d positions across all quarters",
            len(selected_stocks),
        )
    else:
        logger.info("[2/4] Using preselected portfolio...")

        if not skip_price_data_validation:
            data_issues = validate_price_data_coverage_full(
                input_data_filtered, price_data,
                first_quarter, last_quarter)
            if data_issues is not None and not data_issues.empty:
                logger.warning(
                    "Found %d stock-quarter combinations with price data issues",
                    len(data_issues),
                )
                logger.warning(
                    "Issues logged but stocks not filtered since using preselected portfolio"
                )

        selected_stocks, has_category = validate_preselected_input(
            input_data_filtered,
            config.category_scheme,
            config.tp_mode,
            config.sl_mode,
            config.tp_enabled,
            config.sl_enabled,
            config.tiered_config is not None,
        )
        logger.debug(
            "Preselected stocks: %d positions across all quarters",
            len(selected_stocks),
        )
        if has_category:
            logger.debug(
                "Categories found: %s",
                sorted(selected_stocks['cat'].unique().tolist()),
            )
        else:
            logger.debug("No category column in input (using default TP/SL thresholds)")

    return selected_stocks, data_issues


def _simulate_and_compute(
        config: 'BacktestConfig',
        selected_stocks: pd.DataFrame,
        price_data: pd.DataFrame,
        index_data: pd.DataFrame,
        tpsl_scheme: str,
        first_quarter: int,
        last_quarter: int,
    ) -> tuple:
    """Run trade simulation and compute daily portfolio values.
    
    Returns:
        Tuple of (trade_results DataFrame, daily_pf_values DataFrame)
        Either may be None on failure.
    """
    tpsl_category_dimension = config.tpsl_category_dimension
    selection_dimension = config.selection_dimension
    weighting_dimension = config.weighting_dimension
    entry_price_window = config.entry_price_window

    # BUG: may not work as intended for all tiered TP/SL variations
    # Set up tpsl_cat column for TP/SL category lookup
    if 'selection_cat' in selected_stocks.columns:
        # We have category_based selection output with both selection_cat and weight_cat
        if tpsl_category_dimension in ('selection', selection_dimension):
            selected_stocks['tpsl_cat'] = selected_stocks['selection_cat']
        elif tpsl_category_dimension in ('weighting', weighting_dimension):
            selected_stocks['tpsl_cat'] = selected_stocks['weight_cat']
        else:
            # tpsl_category_dimension is a direct dimension name (volatility/mcap)
            # Use the column that matches the resolved tpsl_scheme
            if tpsl_scheme == selection_dimension:
                selected_stocks['tpsl_cat'] = selected_stocks['selection_cat']
            else:  # tpsl_scheme == weighting_dimension
                selected_stocks['tpsl_cat'] = selected_stocks['weight_cat']

    logger.info("[3/4] Simulating trades with TP/SL thresholds...")

    trade_results = simulate_trades(
        selected_stocks,
        price_data,
        tpsl_scheme,
        index_data=index_data,
        tp_mode=config.tp_mode,
        sl_mode=config.sl_mode,
        tp_enabled=config.tp_enabled,
        sl_enabled=config.sl_enabled,
        tiered_config=(
            config.tiered_config.model_dump() if config.tiered_config else None
        ),
        atr_config=config.atr_config.model_dump() if config.atr_config else None,
        pivot_config=(
            config.pivot_config.model_dump() if config.pivot_config else None
        ),
        flat_config=config.flat_config.model_dump() if config.flat_config else None,
        index_exit_config=(
            config.index_exit.model_dump() if config.index_exit else None
        ),
        entry_price_window=entry_price_window,
    )

    if trade_results is None or trade_results.empty:
        return None, None

    logger.debug("Trade simulation complete: %d trades", len(trade_results))

    logger.info("[4/4] Generating daily portfolio values...")

    cash_appreciation_rate = config.cash_appreciation_rate if config.cash_appreciation_rate is not None else RISK_FREE_RATE

    daily_pf_values = compute_portfolio_value_over_quarters(
        trade_results,
        price_data,
        first_quarter,
        last_quarter,
        INITIAL_CAPITAL,
        entry_price_window=entry_price_window,
        risk_free_rate_annual=cash_appreciation_rate,
    )

    if daily_pf_values is not None and not daily_pf_values.empty:
        logger.debug(
            "Daily portfolio values generated: %d days", len(daily_pf_values)
        )

    return trade_results, daily_pf_values


def _backtest_core_impl(
        config: 'BacktestConfig',
        input_data: pd.DataFrame,
        price_data: pd.DataFrame,
        index_data: pd.DataFrame = None,
        skip_price_data_validation: bool = True) -> dict:
    """Inner implementation of backtest_core (unwrapped from try/except)."""

    first_quarter = config.first_quarter
    last_quarter = config.last_quarter
    
    # Resolve TP/SL dimension scheme
    tpsl_scheme = _resolve_tpsl_scheme(config)
    
    # ---------------------------------------------------------------------
    # Filter Input Data by Quarter Range
    # ---------------------------------------------------------------------
    logger.info("[1/4] Filtering data by quarter range...")
    
    if first_quarter is None and last_quarter is None:
        input_data_filtered = input_data.copy()
        first_quarter = int(input_data_filtered['quarter'].min())
        last_quarter = int(input_data_filtered['quarter'].max())
        logger.debug("Using full input data: %d rows", len(input_data_filtered))
        logger.debug("Quarters in data: %s", sorted(input_data_filtered['quarter'].unique().tolist()))
    else:
        input_data_filtered = filter_data_by_quarters(input_data, first_quarter, last_quarter)
        logger.debug("Filtered input data: %d rows", len(input_data_filtered))
        logger.debug("Quarters in data: %s", sorted(input_data_filtered['quarter'].unique().tolist()))
    
    if input_data_filtered.empty:
        return None
    
    # ---------------------------------------------------------------------
    # Run Stock Selection OR Use Preselected Portfolio
    # ---------------------------------------------------------------------
    selected_stocks, data_issues = _select_stocks(
        config, input_data_filtered, price_data,
        first_quarter, last_quarter, skip_price_data_validation,
    )
    
    if selected_stocks is None or selected_stocks.empty:
        return None
    
    # ---------------------------------------------------------------------
    # Simulate Trades and Compute Portfolio Values
    # ---------------------------------------------------------------------
    trade_results, daily_pf_values = _simulate_and_compute(
        config, selected_stocks, price_data, index_data,
        tpsl_scheme, first_quarter, last_quarter,
    )
    
    if trade_results is None:
        return None
    
    return {
        'daily_pf_values': daily_pf_values,
        'trade_results': trade_results,
        'first_quarter': first_quarter,
        'last_quarter': last_quarter,
        'data_issues': data_issues,
    }


def _log_config_summary(config: 'BacktestConfig') -> None:
    """Log a human-readable summary of the loaded configuration."""
    logger.debug("Input data: %s", config.input_data_path)
    logger.debug("Price data: %s", config.price_data_path)
    logger.debug("Index data: %s", config.index_data_path)
    logger.info("Quarter range: %s to %s", config.first_quarter, config.last_quarter)
    logger.debug("Category scheme: %s", config.category_scheme)

    if (config.selection_dimension != config.category_scheme
            or config.weighting_dimension != config.category_scheme):
        logger.debug("Selection dimension: %s", config.selection_dimension)
        logger.debug("Weighting dimension: %s", config.weighting_dimension)
        logger.debug("TP/SL category dimension: %s", config.tpsl_category_dimension)

    logger.debug("Run stock selection: %s", config.run_stock_selection)

    if config.run_stock_selection:
        logger.debug("Selection type: %s", config.selection_type)
        if config.selection_type == 'category_based':
            logger.debug("Category counts: %s", config.category_counts)
            logger.debug("Category weights: %s", config.category_weights)
            logger.debug("Weighting scheme: %s", config.category_based_selection_weighting_scheme)
        elif config.selection_type == 'top_k':
            k = config.top_k_config.k if config.top_k_config else DEFAULT_TOP_K
            weighting = (
                config.top_k_config.weighting_scheme
                if config.top_k_config else DEFAULT_TOP_K_WEIGHTING
            )
            logger.debug("Top k: %s", k)
            logger.debug("Weighting scheme: %s", weighting)
        logger.debug("Sort by: %s", config.sort_by)
        if config.min_prob_threshold is not None:
            logger.debug("Min probability threshold: %s", config.min_prob_threshold)
    else:
        logger.debug("Using preselected portfolio (selection config options ignored)")
        if config.tiered_config is not None:
            logger.debug("Tiered TP/SL config provided")

    logger.debug(
        "TP mode: %s%s", config.tp_mode,
        " (take profit exits disabled)" if not config.tp_enabled else "",
    )
    logger.debug(
        "SL mode: %s%s", config.sl_mode,
        " (stop loss exits disabled)" if not config.sl_enabled else "",
    )
    logger.debug("TP enabled: %s", config.tp_enabled)
    logger.debug("SL enabled: %s", config.sl_enabled)
    logger.debug("Entry price window: %d trading day(s)", config.entry_price_window)


def _save_results_and_report(
        config: 'BacktestConfig',
        results: dict,
        price_data: pd.DataFrame,
        index_data: pd.DataFrame,
) -> str:
    """Create output directory, save artefacts, and optionally generate report.
    
    Returns:
        Path to the output directory.
    """
    trade_results = results['trade_results']
    daily_pf_values = results['daily_pf_values']
    first_quarter = results['first_quarter']
    last_quarter = results['last_quarter']
    data_issues = results.get('data_issues')

    # Create output directory
    output_dir = create_output_directory()
    logger.info("Output directory: %s", output_dir)

    # Add file handler now that output dir exists
    log_path = os.path.join(output_dir, 'backtest_log.txt')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    fh.setFormatter(formatter)
    logging.getLogger('alphaBT').addHandler(fh)
    logger.debug("File logging initialized: %s", log_path)

    # Save config for traceability
    save_config_copy(config, output_dir)

    # Compute comparison data (if index data available)
    comparison_df = None
    if config.index_data_path:
        comparison_df = compute_portfolio_vs_index(
            trade_results, price_data, index_data,
            first_quarter, last_quarter, INITIAL_CAPITAL,
            daily_pf_values=daily_pf_values,
        )
    else:
        logger.info("Skipping index comparison (no index data path specified)")

    # Generate report
    if config.generate_report and daily_pf_values is not None and not daily_pf_values.empty:
        daily_pf = daily_pf_values.reset_index()
        col_map = {
            daily_pf.columns[0]: 'date',
            'Total_Portfolio_Value': 'portfolio_value',
            'Cash_In_Hand': 'cash_in_hand',
            'quarter': 'quarter',
        }
        daily_pf = daily_pf.rename(columns=col_map)
        if 'cash_in_hand' in daily_pf.columns:
            daily_pf['cash_ratio'] = daily_pf['cash_in_hand'] / daily_pf['portfolio_value']

        report_path = os.path.join(output_dir, 'backtest_report.xlsx')
        generate_backtest_report(
            daily_pf=daily_pf,
            trade_results=trade_results,
            comparison_df=comparison_df,
            output_path=report_path,
            sub_periods=config.report_sub_periods,
            input_frequency="daily",
            report_title="Backtest Report",
            data_issues=data_issues,
            first_quarter=first_quarter,
            last_quarter=last_quarter,
        )
    elif config.generate_report:
        logger.info("Skipping report (no equity curve data available)")

    return output_dir


def run_backtest(config_path: str = DEFAULT_CONFIG_PATH) -> Optional[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """
    Main function to run the full backtesting workflow.
    
    Parameters:
    - config_path: Path to the configuration YAML file
    """
    setup_logging(console_level=logging.INFO)
    
    # Record start time
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("BACKTESTING STRATEGY - STARTING")
    logger.info("=" * 60)
    logger.info("Start time: %s", start_time.strftime('%Y-%m-%d %H:%M:%S'))
    
    # -------------------------------------------------------------------------
    # 1. Load Configuration
    # -------------------------------------------------------------------------
    logger.info("[1/2] Loading configuration...")
    config = load_config(config_path)
    _log_config_summary(config)
    
    # -------------------------------------------------------------------------
    # 2. Load Data
    # -------------------------------------------------------------------------
    logger.info("[2/2] Loading data...")
    
    input_data = load_data(config.input_data_path)
    logger.debug("Input data loaded: %d rows", len(input_data))
    
    price_data = load_data(config.price_data_path)
    logger.debug("Price data loaded: %d rows", len(price_data))
    
    index_data = load_data(config.index_data_path)
    logger.debug("Index data loaded: %d rows", len(index_data))
    
    # -------------------------------------------------------------------------
    # 3. Run Backtest Core
    # -------------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("RUNNING BACKTEST CORE")
    logger.info("-" * 60)
    
    results = backtest_core(
        config=config,
        input_data=input_data,
        price_data=price_data,
        index_data=index_data,
    )
    
    if results is None:
        logger.error("Backtest core returned no results.")
        return None, None, None
    
    logger.info("-" * 60)
    
    # -------------------------------------------------------------------------
    # 4. Save Results and Generate Report
    # -------------------------------------------------------------------------
    logger.info("Saving results...")
    output_dir = _save_results_and_report(config, results, price_data, index_data)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    daily_pf_values = results['daily_pf_values']
    trade_results = results['trade_results']

    logger.info("=" * 60)
    logger.info("BACKTESTING COMPLETE")
    logger.info("=" * 60)
    logger.info("All outputs saved to: %s", output_dir)
    logger.info("Files generated:")
    logger.info("  - config_used.yaml (configuration traceability)")
    if config.generate_report and daily_pf_values is not None and not daily_pf_values.empty:
        logger.info("  - backtest_report.xlsx (metrics, charts, trade data — all in one)")
    logger.info("  - backtest_log.txt (full execution log)")
    
    # Calculate and log execution time
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    total_seconds = elapsed_time.total_seconds()
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    logger.info("End time: %s", end_time.strftime('%Y-%m-%d %H:%M:%S'))
    if hours > 0:
        logger.info("Total execution time: %dh %dm %.2fs", int(hours), int(minutes), seconds)
    elif minutes > 0:
        logger.info("Total execution time: %dm %.2fs", int(minutes), seconds)
    else:
        logger.info("Total execution time: %.2fs", seconds)
    
    return output_dir, trade_results, daily_pf_values


if __name__ == '__main__':
    # Hari Om
    output_dir, portfolio_results, daily_pf_values = run_backtest()
