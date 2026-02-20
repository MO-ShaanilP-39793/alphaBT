"""TP/SL Backtesting Module

This module handles trade execution with Take Profit and Stop Loss thresholds.
TP and SL modes can be configured independently, allowing mixed combinations
(e.g., ATR for take profit with pivot points for stop loss).

Supported modes (for each of tp_mode / sl_mode):
- 'tiered': Per-category percentage thresholds (requires tiered_config with category mapping)
- 'flat': Single percentage for all stocks (no category required)
- 'atr': Dynamic thresholds based on Average True Range
- 'pivot': Dynamic thresholds based on pivot point support/resistance levels

Also supports index-guided exit strategies:
- Regime filter: Exit when index falls below moving average
- Volatility adjustment: Widen/tighten thresholds based on market volatility

"""

import pandas as pd

from backtest.dynamic_levels import (
    calculate_atr_thresholds,
    calculate_pivot_thresholds,
    get_volatility_adjustment_multiplier,
    check_regime_exit_signal
)
from config.defaults import (
    DEFAULT_TPSL_FALLBACK_PCT,
    DEFAULT_ATR_TP_MULTIPLIER,
    DEFAULT_ATR_SL_MULTIPLIER,
    DEFAULT_ATR_PERIOD,
    DEFAULT_PIVOT_TP_LEVEL,
    DEFAULT_PIVOT_SL_LEVEL,
    DEFAULT_PIVOT_LOOKBACK_DAYS,
    DEFAULT_FLAT_TP_PCT,
    DEFAULT_FLAT_SL_PCT,
    DEFAULT_TP_MODE,
    DEFAULT_SL_MODE,
    DEFAULT_TP_ENABLED,
    DEFAULT_SL_ENABLED,
    DEFAULT_ENTRY_PRICE_WINDOW,
    DEFAULT_VOL_LOOKBACK,
    DEFAULT_HIGH_VOL_THRESHOLD,
    DEFAULT_LOW_VOL_THRESHOLD,
    DEFAULT_HIGH_VOL_MULTIPLIER,
    DEFAULT_LOW_VOL_MULTIPLIER,
    DEFAULT_REGIME_MA_PERIOD,
    DEFAULT_REGIME_EXIT_THRESHOLD,
)

def get_date_params(quarter_str):
    """
    Parses YYYYMM string to determine entry search start date 
    and mandatory exit cutoff date.
    """
    year = int(quarter_str[:4])
    mm = int(quarter_str[4:])
    
    entry_search_start = None
    mandatory_exit_date = None
    
    if mm == 2:  # 15th Feb to 30th May endpoints inclusive
        entry_search_start = pd.Timestamp(year=year, month=2, day=15)
        mandatory_exit_date = pd.Timestamp(year=year, month=5, day=30)
    elif mm == 5:  # 31st May to 14th August
        entry_search_start = pd.Timestamp(year=year, month=5, day=31)
        mandatory_exit_date = pd.Timestamp(year=year, month=8, day=14)
    elif mm == 8:  # 15th August to 14th November
        entry_search_start = pd.Timestamp(year=year, month=8, day=15)
        mandatory_exit_date = pd.Timestamp(year=year, month=11, day=14)
    elif mm == 11:  # 15th November to 14th Feb (but the next year)
        entry_search_start = pd.Timestamp(year=year, month=11, day=15)
        # Note: Nov quarter exits in Feb of the NEXT year
        mandatory_exit_date = pd.Timestamp(year=year + 1, month=2, day=14)
        
    return entry_search_start, mandatory_exit_date


def calculate_thresholds_tiered(
        entry_price, 
        cat,
        tiered_config):
    """
    Calculate tiered (per-category) percentage-based TP/SL thresholds.
    
    Parameters:
    - entry_price: Entry price of the trade
    - cat: Category name (e.g., 'high_volatility', 'largecap'), or '_default' / None
    - tiered_config: Dict with structure:
        {
            'tp_pct': {category_name: float, ...},
            'sl_pct': {category_name: float, ...},
            'default_tp_pct': float,  # fallback when cat is None/_default
            'default_sl_pct': float,
        }
    
    Returns:
    - Tuple of (tp_price, sl_price, tp_pct, sl_pct)
    """
    if cat is None or cat == '_default':
        # Use default thresholds when no category is provided
        tp_pct = tiered_config.get('default_tp_pct', DEFAULT_TPSL_FALLBACK_PCT)
        sl_pct = tiered_config.get('default_sl_pct', DEFAULT_TPSL_FALLBACK_PCT)
    else:
        tp_pct = tiered_config.get('tp_pct', {}).get(cat, DEFAULT_TPSL_FALLBACK_PCT)
        sl_pct = tiered_config.get('sl_pct', {}).get(cat, DEFAULT_TPSL_FALLBACK_PCT)
    
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)
    
    return tp_price, sl_price, tp_pct, sl_pct


def _compute_single_side(side, mode, price_df, co_name, entry_price, entry_date,
                         category_scheme, cat, *, atr_config, pivot_config,
                         flat_config, tiered_config):
    """
    Compute the threshold price for one side (TP or SL) using the specified mode.
    
    Parameters:
    - side: 'tp' or 'sl'
    - mode: 'tiered', 'flat', 'atr', or 'pivot'
    - price_df, co_name, entry_price, entry_date: trade context
    - category_scheme, cat: category info for tiered-mode lookup
    - atr_config, pivot_config, flat_config: mode-specific configs
    - tiered_config: tiered-mode config dict
    
    Returns:
    - Tuple of (price, side_metadata_dict)
      side_metadata_dict contains keys like 'pct', 'mode', and mode-specific info
    """
    meta = {'mode': mode}
    
    if mode == 'atr':
        tp_price, sl_price, atr = calculate_atr_thresholds(
            price_df, co_name, entry_price, entry_date,
            tp_multiplier=atr_config.get('tp_multiplier', DEFAULT_ATR_TP_MULTIPLIER),
            sl_multiplier=atr_config.get('sl_multiplier', DEFAULT_ATR_SL_MULTIPLIER),
            period=atr_config.get('period', DEFAULT_ATR_PERIOD)
        )
        
        if tp_price is None:
            # Fallback to flat if ATR calculation fails
            if side == 'tp':
                pct = flat_config.get('tp_pct', DEFAULT_FLAT_TP_PCT)
                price = entry_price * (1 + pct)
            else:
                pct = flat_config.get('sl_pct', DEFAULT_FLAT_SL_PCT)
                price = entry_price * (1 - pct)
            meta['fallback'] = 'flat'
            meta['pct'] = pct
        else:
            price = tp_price if side == 'tp' else sl_price
            meta['atr'] = atr
            if side == 'tp':
                meta['pct'] = (tp_price - entry_price) / entry_price
            else:
                meta['pct'] = (entry_price - sl_price) / entry_price
                
    elif mode == 'pivot':
        tp_price, sl_price, pivots = calculate_pivot_thresholds(
            price_df, co_name, entry_price, entry_date,
            tp_level=pivot_config.get('tp_level', DEFAULT_PIVOT_TP_LEVEL),
            sl_level=pivot_config.get('sl_level', DEFAULT_PIVOT_SL_LEVEL),
            lookback_days=pivot_config.get('lookback_days', DEFAULT_PIVOT_LOOKBACK_DAYS)
        )
        
        if tp_price is None:
            # Fallback to flat if pivot calculation fails
            if side == 'tp':
                pct = flat_config.get('tp_pct', DEFAULT_FLAT_TP_PCT)
                price = entry_price * (1 + pct)
            else:
                pct = flat_config.get('sl_pct', DEFAULT_FLAT_SL_PCT)
                price = entry_price * (1 - pct)
            meta['fallback'] = 'flat'
            meta['pct'] = pct
        else:
            price = tp_price if side == 'tp' else sl_price
            meta['pivots'] = pivots
            if side == 'tp':
                meta['pct'] = (tp_price - entry_price) / entry_price
            else:
                meta['pct'] = (entry_price - sl_price) / entry_price
                
    elif mode == 'flat':
        if side == 'tp':
            pct = flat_config.get('tp_pct', DEFAULT_FLAT_TP_PCT)
            price = entry_price * (1 + pct)
        else:
            pct = flat_config.get('sl_pct', DEFAULT_FLAT_SL_PCT)
            price = entry_price * (1 - pct)
        meta['pct'] = pct
    
    elif mode == 'tiered':
        f_tp, f_sl, f_tp_pct, f_sl_pct = calculate_thresholds_tiered(
            entry_price, cat, tiered_config
        )
        if side == 'tp':
            price = f_tp
            meta['pct'] = f_tp_pct
        else:
            price = f_sl
            meta['pct'] = f_sl_pct
    
    else:
        raise ValueError(f"Unknown TP/SL mode: '{mode}'. Use 'tiered', 'flat', 'atr', or 'pivot'.")
    
    return price, meta


def calculate_dynamic_thresholds(price_df, co_name, entry_price, entry_date, 
                                  category_scheme, cat, tp_mode, sl_mode,
                                  index_df=None,
                                  *, atr_config, pivot_config, flat_config,
                                  index_exit_config, tiered_config):
    """
    Calculate TP/SL thresholds using independently specified modes.
    
    TP and SL can each use a different calculation mode, enabling mixed
    strategies (e.g., ATR for take profit with pivot for stop loss).
    
    Parameters:
    - price_df: DataFrame with price data
    - co_name: Stock name
    - entry_price: Entry price of the trade
    - entry_date: Entry date of the trade
    - category_scheme: 'volatility' or 'mcap'
    - cat: Category name
    - tp_mode: Mode for take profit - 'tiered', 'flat', 'atr', or 'pivot'
    - sl_mode: Mode for stop loss - 'tiered', 'flat', 'atr', or 'pivot'
    - index_df: Optional DataFrame with index data for volatility adjustment
    - atr_config: ATR mode config dict
    - pivot_config: Pivot mode config dict
    - flat_config: Flat mode config dict
    - index_exit_config: Index-guided exit config dict
    - tiered_config: Tiered-mode config dict
    
    Returns:
    - Tuple of (tp_price, sl_price, metadata_dict)
    """
    common_kwargs = dict(
        price_df=price_df, co_name=co_name, entry_price=entry_price,
        entry_date=entry_date, category_scheme=category_scheme, cat=cat,
        atr_config=atr_config, pivot_config=pivot_config,
        flat_config=flat_config, tiered_config=tiered_config
    )
    
    tp_price, tp_meta = _compute_single_side('tp', tp_mode, **common_kwargs)
    sl_price, sl_meta = _compute_single_side('sl', sl_mode, **common_kwargs)
    
    metadata = {
        'tp_mode': tp_mode,
        'sl_mode': sl_mode,
        'tp_pct': tp_meta.get('pct'),
        'sl_pct': sl_meta.get('pct'),
    }
    # Merge mode-specific metadata
    for key in ('atr', 'pivots', 'fallback'):
        if key in tp_meta:
            metadata[f'tp_{key}'] = tp_meta[key]
        if key in sl_meta:
            metadata[f'sl_{key}'] = sl_meta[key]
    
    # Apply volatility adjustment if enabled and index data is available
    vol_config = index_exit_config.get('vol_adjustment', {})
    if vol_config.get('enabled', False) and index_df is not None:
        multiplier = get_volatility_adjustment_multiplier(
            index_df, entry_date,
            lookback=vol_config.get('lookback', DEFAULT_VOL_LOOKBACK),
            high_vol_threshold=vol_config.get('high_vol_threshold', DEFAULT_HIGH_VOL_THRESHOLD),
            low_vol_threshold=vol_config.get('low_vol_threshold', DEFAULT_LOW_VOL_THRESHOLD),
            high_vol_multiplier=vol_config.get('high_vol_multiplier', DEFAULT_HIGH_VOL_MULTIPLIER),
            low_vol_multiplier=vol_config.get('low_vol_multiplier', DEFAULT_LOW_VOL_MULTIPLIER)
        )
        
        if multiplier != 1.0:
            # Adjust the distance from entry price by the multiplier
            tp_distance = tp_price - entry_price
            sl_distance = entry_price - sl_price
            
            tp_price = entry_price + (tp_distance * multiplier)
            sl_price = entry_price - (sl_distance * multiplier)
            
            metadata['vol_adjustment_multiplier'] = multiplier
    
    return tp_price, sl_price, metadata


def process_trade(row, price_df, category_scheme, index_df=None,
                   tp_mode=DEFAULT_TP_MODE, sl_mode=DEFAULT_SL_MODE,
                   tp_enabled=DEFAULT_TP_ENABLED, sl_enabled=DEFAULT_SL_ENABLED, entry_price_window=DEFAULT_ENTRY_PRICE_WINDOW,
                   *, tiered_config, atr_config,
                   pivot_config, flat_config, index_exit_config):
    """
    Calculates entry and exit data for a single row from selected_stocks.
    
    Parameters:
    - row: A row from selected_stocks DataFrame
    - price_df: DataFrame with price data
    - category_scheme: 'volatility' or 'mcap' to determine TP/SL config
    - index_df: Optional DataFrame with index data for regime-based exits
    - tp_mode: Mode for take profit - 'tiered', 'flat', 'atr', or 'pivot'
    - sl_mode: Mode for stop loss - 'tiered', 'flat', 'atr', or 'pivot'
    - tp_enabled: Whether take profit is active
    - sl_enabled: Whether stop loss is active
    - entry_price_window: Number of trading days to average for entry price (default: 3)
    - tiered_config: Tiered-mode config dict
    - atr_config: ATR mode config dict
    - pivot_config: Pivot mode config dict
    - flat_config: Flat mode config dict
    - index_exit_config: Index-guided exit config dict
    
    Returns:
    - pandas Series with trade results
    """
    co_name = row['co_name']
    quarter = str(row['quarter'])
    # Use tpsl_cat for TP/SL threshold lookup if present (cross-dimensional mode),
    # otherwise fall back to cat (standard mode)
    cat = row['tpsl_cat'] if 'tpsl_cat' in row.index else row['cat']
    
    # Result columns
    result_cols = ['entry_date', 'exit_date', 'entry_price', 'exit_price', 
                   'holding_period', 'SL_triggered', 'TP_triggered', 
                   'regime_exit', 'tp_pct_used', 'sl_pct_used']
    null_result = pd.Series([None] * len(result_cols), index=result_cols)
    
    # 1. Get Date Boundaries
    entry_start_limit, mandatory_exit_limit = get_date_params(quarter)
    
    # Filter price data for this company
    co_prices = price_df[price_df['co_name'] == co_name].sort_values('date')
    
    # ---------------------------------------------
    # CALCULATE ENTRY
    # ---------------------------------------------
    potential_entry_days = co_prices[co_prices['date'] >= entry_start_limit]
    
    if len(potential_entry_days) < entry_price_window:
        return null_result
    
    entry_window = potential_entry_days.iloc[:entry_price_window]
    entry_price = entry_window['close'].mean()
    entry_date = entry_window.iloc[entry_price_window // 2]['date']
    last_entry_calc_date = entry_window.iloc[-1]['date']
    
    # ---------------------------------------------
    # CALCULATE THRESHOLDS (only if TP or SL is enabled)
    # ---------------------------------------------
    tp_price = None
    sl_price = None
    tp_pct_used = None
    sl_pct_used = None
    
    if tp_enabled or sl_enabled:
        tp_price, sl_price, threshold_metadata = calculate_dynamic_thresholds(
            price_df, co_name, entry_price, entry_date,
            category_scheme, cat, tp_mode, sl_mode, index_df,
            atr_config=atr_config, pivot_config=pivot_config,
            flat_config=flat_config, index_exit_config=index_exit_config,
            tiered_config=tiered_config
        )
        
        if tp_enabled:
            tp_pct_used = threshold_metadata.get('tp_pct', DEFAULT_TPSL_FALLBACK_PCT)
        else:
            tp_price = None  # Disable TP threshold
            
        if sl_enabled:
            sl_pct_used = threshold_metadata.get('sl_pct', DEFAULT_TPSL_FALLBACK_PCT)
        else:
            sl_price = None  # Disable SL threshold
    
    # ---------------------------------------------
    # CALCULATE EXIT
    # ---------------------------------------------
    monitoring_df = co_prices[
        (co_prices['date'] > last_entry_calc_date) & 
        (co_prices['date'] <= mandatory_exit_limit)
    ]
    
    exit_date = None
    exit_price = None
    sl_triggered = False
    tp_triggered = False
    regime_exit = False
    
    # Get regime filter config
    regime_config = index_exit_config.get('regime_filter', {})
    regime_filter_enabled = regime_config.get('enabled', False) and index_df is not None
    
    # Iterate through days to check for TP/SL and regime exits
    for idx, day_data in monitoring_df.iterrows():
        current_date = day_data['date']
        
        # Check regime exit first (market downturn)
        if regime_filter_enabled:
            if check_regime_exit_signal(
                index_df, current_date,
                ma_period=regime_config.get('ma_period', DEFAULT_REGIME_MA_PERIOD),
                exit_threshold=regime_config.get('exit_threshold', DEFAULT_REGIME_EXIT_THRESHOLD)
            ):
                exit_date = current_date
                exit_price = day_data['close']  # Exit at close
                regime_exit = True
                break
        
        # Check Stop Loss (Priority: checked against Low) - only if SL is enabled
        if sl_price is not None and day_data['low'] <= sl_price:
            exit_date = current_date
            exit_price = sl_price
            sl_triggered = True
            break
            
        # Check Take Profit (Checked against High) - only if TP is enabled
        if tp_price is not None and day_data['high'] >= tp_price:
            exit_date = current_date
            exit_price = tp_price
            tp_triggered = True
            break
            
    # If loop finishes without TP/SL/Regime, use Time Exit
    if exit_date is None:
        if not monitoring_df.empty:
            last_day = monitoring_df.iloc[-1]
            exit_date = last_day['date']
            exit_price = last_day['close']
        else:
            return pd.Series([entry_date, None, entry_price, None, None, None, None, None, tp_pct_used, sl_pct_used], 
                           index=result_cols)

    holding_period = (exit_date - entry_date).days

    return pd.Series([entry_date, exit_date, entry_price, exit_price, holding_period, 
                      sl_triggered, tp_triggered, regime_exit, tp_pct_used, sl_pct_used], 
                     index=result_cols)


def simulate_trades(
    selected_stocks: pd.DataFrame, 
    price_data: pd.DataFrame, 
    category_scheme: str, 
    index_data: pd.DataFrame = None,
    tp_mode: str = None,
    sl_mode: str = None,
    tp_enabled: bool = None,
    sl_enabled: bool = None,
    tiered_config: dict = None,
    atr_config: dict = None,
    pivot_config: dict = None,
    flat_config: dict = None,
    index_exit_config: dict = None,
    entry_price_window: int = DEFAULT_ENTRY_PRICE_WINDOW):
    """
    Calculate portfolio performance with TP/SL simulation.
    
    Supports two input modes:
    - Selection mode: selected_stocks has ['quarter', 'co_name', 'cat', 'cat_weight']
    - Preselected mode: selected_stocks has ['quarter', 'co_name', 'stock_weight'] with optional 'cat'
    
    TP and SL modes can be configured independently, enabling mixed strategies
    (e.g., ATR for take profit with pivot points for stop loss).
    
    Parameters:
    - selected_stocks: DataFrame with required columns depending on mode (see above)
    - price_data: ['date', 'co_name', 'open', 'high', 'low', 'close']
    - category_scheme: 'volatility' or 'mcap' to determine which TP/SL config to use
    - index_data: Optional DataFrame with ['date', 'value'] for index-guided exits
    - tp_mode: Take profit mode ('tiered', 'flat', 'atr', 'pivot'). Defaults to 'flat' if None.
    - sl_mode: Stop loss mode ('tiered', 'flat', 'atr', 'pivot'). Defaults to 'flat' if None.
    - tp_enabled: Whether take profit is active. Defaults to True if None.
    - sl_enabled: Whether stop loss is active. Defaults to True if None.
    - tiered_config: Per-category TP/SL config dict with 'tp_pct', 'sl_pct' mappings,
      plus 'default_tp_pct' and 'default_sl_pct' fallbacks
    - atr_config: ATR config dict with 'period', 'tp_multiplier', 'sl_multiplier'
    - pivot_config: Pivot config dict with 'lookback_days', 'tp_level', 'sl_level'
    - flat_config: Flat config dict with 'tp_pct', 'sl_pct'
    - index_exit_config: Index exit config dict with 'regime_filter' and 'vol_adjustment'
    - entry_price_window: Number of trading days to average for entry price (default: 3)
    
    Returns:
    - DataFrame with trade results including entry/exit details
    """
    # Resolve mode flags
    resolved_tp_mode = tp_mode if tp_mode is not None else DEFAULT_TP_MODE
    resolved_sl_mode = sl_mode if sl_mode is not None else DEFAULT_SL_MODE
    use_tp = tp_enabled if tp_enabled is not None else DEFAULT_TP_ENABLED
    use_sl = sl_enabled if sl_enabled is not None else DEFAULT_SL_ENABLED

    # --- Validate required configs based on active modes ---
    active_modes = set()
    if use_tp:
        active_modes.add(resolved_tp_mode)
    if use_sl:
        active_modes.add(resolved_sl_mode)

    # tiered mode requires tiered_config
    if 'tiered' in active_modes:
        if tiered_config is None:
            raise ValueError(
                f"tiered_config is required when tp_mode or sl_mode is 'tiered' but was not provided"
            )

    if 'atr' in active_modes and atr_config is None:
        raise ValueError("atr_config is required when tp_mode or sl_mode is 'atr'")

    if 'pivot' in active_modes and pivot_config is None:
        raise ValueError("pivot_config is required when tp_mode or sl_mode is 'pivot'")

    # flat_config is always needed (used as fallback for atr/pivot failures too)
    if flat_config is None:
        flat_config = {}

    if index_exit_config is None:
        raise ValueError("index_exit_config is required but was not provided")

    # --- Input Validation ---
    if category_scheme not in ['volatility', 'mcap']:
        raise ValueError("category_scheme must be 'volatility' or 'mcap'")
    
    # Base required columns
    required_base_cols = {'quarter', 'co_name'}
    required_price_cols = {'date', 'co_name', 'open', 'high', 'low', 'close'}
    
    if not required_base_cols.issubset(selected_stocks.columns):
        missing_cols = required_base_cols - set(selected_stocks.columns)
        raise ValueError(f"Missing required columns in selected_stocks: {missing_cols}")
    
    # Check for weight column: either cat_weight (selection mode) or stock_weight (preselected mode)
    has_cat_weight = 'cat_weight' in selected_stocks.columns
    has_stock_weight = 'stock_weight' in selected_stocks.columns
    
    if not (has_cat_weight or has_stock_weight):
        raise ValueError("Missing weight column: need either 'cat_weight' or 'stock_weight'")

    if not required_price_cols.issubset(price_data.columns):
        missing_cols = required_price_cols - set(price_data.columns)
        raise ValueError(f"Missing required columns in price_data: {missing_cols}")
    
    # Make a copy of selected_stocks to avoid modifying the original
    selected_stocks = selected_stocks.copy()
    
    # Handle optional 'cat' column: add placeholder if missing
    if 'cat' not in selected_stocks.columns:
        selected_stocks['cat'] = '_default'
    
    # Ensure date column is datetime
    price_data = price_data.copy()
    price_data['date'] = pd.to_datetime(price_data['date'])
    
    # Prepare index data if provided
    index_df = None
    if index_data is not None:
        index_df = index_data.copy()
        index_df['date'] = pd.to_datetime(index_df['date'])

    # Sort the selected_stocks data
    selected_stocks = selected_stocks.sort_values(by=['quarter', 'co_name'], ignore_index=True)
    
    # Apply the logic row by row
    results = selected_stocks.apply(
        lambda row: process_trade(
            row, price_data, category_scheme, index_df,
            resolved_tp_mode, resolved_sl_mode,
            use_tp, use_sl, entry_price_window,
            tiered_config=tiered_config, atr_config=atr_config,
            pivot_config=pivot_config, flat_config=flat_config,
            index_exit_config=index_exit_config),
        axis=1
    )
    
    # Concatenate the generated columns back to original dataframe
    final_df = pd.concat([selected_stocks, results], axis=1)
    
    # Calculate stock return
    final_df['stock_return'] = (final_df['exit_price'] - final_df['entry_price']) / final_df['entry_price']
    
    return final_df
