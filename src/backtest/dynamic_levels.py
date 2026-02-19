"""
Dynamic Levels Module - ATR and Pivot Point Calculations

This module provides functions for calculating dynamic support/resistance levels
used to set adaptive TP/SL thresholds based on each stock's price behavior.
"""

import pandas as pd
import numpy as np
import logging
from config.defaults import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_ATR_TP_MULTIPLIER,
    DEFAULT_ATR_SL_MULTIPLIER,
    ATR_SL_FLOOR_FACTOR,
    DEFAULT_PIVOT_LOOKBACK_DAYS,
    DEFAULT_PIVOT_TP_LEVEL,
    DEFAULT_PIVOT_SL_LEVEL,
    PIVOT_TP_FALLBACK_FACTOR,
    PIVOT_SL_FALLBACK_FACTOR,
    DEFAULT_VOL_LOOKBACK,
    DEFAULT_HIGH_VOL_THRESHOLD,
    DEFAULT_LOW_VOL_THRESHOLD,
    DEFAULT_HIGH_VOL_MULTIPLIER,
    DEFAULT_LOW_VOL_MULTIPLIER,
    DEFAULT_REGIME_MA_PERIOD,
    DEFAULT_REGIME_EXIT_THRESHOLD,
    TRADING_DAYS_PER_YEAR,
)

# Configure module-level logger
logger = logging.getLogger(__name__)


def setup_dynamic_levels_logging(log_file: str = None, level: int = logging.INFO):
    """
    Configure logging for the dynamic_levels module.
    
    Parameters:
    - log_file: Path to log file. If None, logs only to console.
    - level: Logging level (e.g., logging.DEBUG, logging.INFO)
    
    Example:
        setup_dynamic_levels_logging('logs/dynamic_levels.log', logging.DEBUG)
    """
    logger.setLevel(level)
    
    # Formatter with timestamp, level, and context
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def calculate_atr(price_df: pd.DataFrame, co_name: str, end_date: pd.Timestamp, period: int = DEFAULT_ATR_PERIOD) -> float:
    """
    Calculate Average True Range (ATR) for a stock.
    
    ATR measures volatility by decomposing the entire range of an asset price for a period.
    
    Parameters:
    - price_df: DataFrame with columns ['date', 'co_name', 'open', 'high', 'low', 'close']
    - co_name: Stock name to calculate ATR for
    - end_date: Calculate ATR using data up to this date
    - period: Number of periods for ATR calculation (default 14)
    
    Returns:
    - ATR value as a float, or None if insufficient data
    """
    # Filter for the specific stock and dates before end_date
    stock_data = price_df[
        (price_df['co_name'] == co_name) & 
        (price_df['date'] < end_date)
    ].sort_values('date').copy()
    
    if len(stock_data) < period + 1:
        return None
    
    # Take the most recent data needed for ATR
    stock_data = stock_data.tail(period + 1).reset_index(drop=True)
    
    # Calculate True Range components
    stock_data['prev_close'] = stock_data['close'].shift(1)
    stock_data['tr1'] = stock_data['high'] - stock_data['low']
    stock_data['tr2'] = abs(stock_data['high'] - stock_data['prev_close'])
    stock_data['tr3'] = abs(stock_data['low'] - stock_data['prev_close'])
    
    # True Range is the max of the three components
    stock_data['true_range'] = stock_data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # ATR is the average of True Range over the period
    # Skip the first row (no prev_close)
    atr = stock_data['true_range'].iloc[1:].mean()
    
    return atr


def calculate_atr_thresholds(price_df: pd.DataFrame, co_name: str, entry_price: float,
                              end_date: pd.Timestamp, tp_multiplier: float = DEFAULT_ATR_TP_MULTIPLIER,
                              sl_multiplier: float = DEFAULT_ATR_SL_MULTIPLIER, period: int = DEFAULT_ATR_PERIOD) -> tuple:
    """
    Calculate TP/SL thresholds based on ATR.
    
    Parameters:
    - price_df: DataFrame with price data
    - co_name: Stock name
    - entry_price: Entry price of the trade
    - end_date: Date to calculate ATR up to (typically entry date)
    - tp_multiplier: Multiplier for take profit (TP = entry + multiplier * ATR)
    - sl_multiplier: Multiplier for stop loss (SL = entry - multiplier * ATR)
    - period: ATR period
    
    Returns:
    - Tuple of (tp_price, sl_price, atr_value)
    - Returns (None, None, None) if ATR cannot be calculated
    """
    atr = calculate_atr(price_df, co_name, end_date, period)
    
    if atr is None or atr == 0:
        return None, None, None
    
    tp_price = entry_price + (tp_multiplier * atr)
    sl_price = entry_price - (sl_multiplier * atr)
    
    # Ensure SL is not negative
    sl_price = max(sl_price, entry_price * ATR_SL_FLOOR_FACTOR)  # Floor at 50% of entry
    
    return tp_price, sl_price, atr


def calculate_pivot_points(price_df: pd.DataFrame, co_name: str, 
                           end_date: pd.Timestamp, lookback_days: int = DEFAULT_PIVOT_LOOKBACK_DAYS) -> dict:
    """
    Calculate classic pivot points for support/resistance levels.
    
    Pivot Point = (High + Low + Close) / 3
    R1 = 2 × Pivot - Low
    R2 = Pivot + (High - Low)
    R3 = High + 2 × (Pivot - Low)
    S1 = 2 × Pivot - High
    S2 = Pivot - (High - Low)
    S3 = Low - 2 × (High - Pivot)
    
    Parameters:
    - price_df: DataFrame with price data
    - co_name: Stock name
    - end_date: Calculate pivots from data before this date
    - lookback_days: Number of trading days to look back for H/L/C
    
    Returns:
    - Dictionary with keys: 'pivot', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3'
    - Returns None if insufficient data
    """
    # Filter for the specific stock and dates before end_date
    stock_data = price_df[
        (price_df['co_name'] == co_name) & 
        (price_df['date'] < end_date)
    ].sort_values('date').copy()
    
    if len(stock_data) < lookback_days:
        # Use whatever data is available if less than lookback
        if len(stock_data) < 5:  # Need at least 5 days
            return None
    else:
        stock_data = stock_data.tail(lookback_days)
    
    # Get the high, low, close for pivot calculation
    high = stock_data['high'].max()
    low = stock_data['low'].min()
    close = stock_data['close'].iloc[-1]
    
    # Calculate pivot point
    pivot = (high + low + close) / 3
    
    # Calculate resistance levels
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    # Calculate support levels
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'pivot': pivot,
        'R1': r1,
        'R2': r2,
        'R3': r3,
        'S1': s1,
        'S2': s2,
        'S3': s3,
        'period_high': high,
        'period_low': low,
        'period_close': close
    }


def calculate_pivot_thresholds(price_df: pd.DataFrame, co_name: str, entry_price: float,
                                end_date: pd.Timestamp, tp_level: str = DEFAULT_PIVOT_TP_LEVEL,
                                sl_level: str = DEFAULT_PIVOT_SL_LEVEL, lookback_days: int = DEFAULT_PIVOT_LOOKBACK_DAYS) -> tuple:
    """
    Calculate TP/SL thresholds based on pivot points.
    
    Parameters:
    - price_df: DataFrame with price data
    - co_name: Stock name
    - entry_price: Entry price of the trade
    - end_date: Date to calculate pivots up to
    - tp_level: Which resistance level to use for TP ('R1', 'R2', 'R3')
    - sl_level: Which support level to use for SL ('S1', 'S2', 'S3')
    - lookback_days: Days to look back for pivot calculation
    
    Returns:
    - Tuple of (tp_price, sl_price, pivot_dict)
    - pivot_dict includes 'fallback_info' key with metadata about any fallbacks used
    - Returns (None, None, None) if pivots cannot be calculated
    """
    pivots = calculate_pivot_points(price_df, co_name, end_date, lookback_days)
    
    if pivots is None:
        logger.warning(f"PIVOT_CALC_FAILED | {co_name} | date={end_date.date()} | "
                      f"Insufficient data for pivot calculation")
        return None, None, None
    
    tp_price = pivots.get(tp_level)
    sl_price = pivots.get(sl_level)
    original_tp = tp_price
    original_sl = sl_price
    
    # Track fallback metadata
    fallback_info = {
        'tp_fallback_used': False,
        'tp_fallback_reason': None,
        'tp_original_level': tp_level,
        'tp_final_level': tp_level,
        'sl_fallback_used': False,
        'sl_fallback_reason': None,
        'sl_original_level': sl_level,
        'sl_final_level': sl_level,
    }
    
    # Validate: TP should be above entry, SL should be below entry
    if tp_price is not None and tp_price <= entry_price:
        # If calculated TP is below entry, use next level up
        level_order = ['R1', 'R2', 'R3']
        current_idx = level_order.index(tp_level) if tp_level in level_order else 0
        for i in range(current_idx + 1, len(level_order)):
            if pivots[level_order[i]] > entry_price:
                tp_price = pivots[level_order[i]]
                fallback_info['tp_final_level'] = level_order[i]
                logger.info(f"TP_LEVEL_ESCALATION | {co_name} | date={end_date.date()} | "
                           f"entry={entry_price:.2f} | {tp_level}={original_tp:.2f} <= entry | "
                           f"escalated to {level_order[i]}={tp_price:.2f}")
                break
        # If still no valid TP, use a percentage fallback
        if tp_price <= entry_price:
            tp_price = entry_price * PIVOT_TP_FALLBACK_FACTOR  # 5% fallback
            fallback_info['tp_fallback_used'] = True
            fallback_info['tp_fallback_reason'] = 'all_resistance_below_entry'
            fallback_info['tp_final_level'] = 'PCT_FALLBACK_5'
            
            logger.warning(
                f"TP_PERCENTAGE_FALLBACK | {co_name} | date={end_date.date()} | "
                f"entry={entry_price:.2f} | R1={pivots['R1']:.2f} R2={pivots['R2']:.2f} R3={pivots['R3']:.2f} | "
                f"All resistance levels <= entry | Using 5% fallback: TP={tp_price:.2f} | "
                f"period_high={pivots['period_high']:.2f} period_low={pivots['period_low']:.2f}"
            )
    
    if sl_price is not None and sl_price >= entry_price:
        # If calculated SL is above entry, use next level down
        level_order = ['S1', 'S2', 'S3']
        current_idx = level_order.index(sl_level) if sl_level in level_order else 0
        for i in range(current_idx + 1, len(level_order)):
            if pivots[level_order[i]] < entry_price:
                sl_price = pivots[level_order[i]]
                fallback_info['sl_final_level'] = level_order[i]
                logger.info(f"SL_LEVEL_ESCALATION | {co_name} | date={end_date.date()} | "
                           f"entry={entry_price:.2f} | {sl_level}={original_sl:.2f} >= entry | "
                           f"escalated to {level_order[i]}={sl_price:.2f}")
                break
        # If still no valid SL, use a percentage fallback
        if sl_price >= entry_price:
            sl_price = entry_price * PIVOT_SL_FALLBACK_FACTOR  # 5% fallback
            fallback_info['sl_fallback_used'] = True
            fallback_info['sl_fallback_reason'] = 'all_support_above_entry'
            fallback_info['sl_final_level'] = 'PCT_FALLBACK_5'
            
            logger.warning(
                f"SL_PERCENTAGE_FALLBACK | {co_name} | date={end_date.date()} | "
                f"entry={entry_price:.2f} | S1={pivots['S1']:.2f} S2={pivots['S2']:.2f} S3={pivots['S3']:.2f} | "
                f"All support levels >= entry | Using 5% fallback: SL={sl_price:.2f} | "
                f"period_high={pivots['period_high']:.2f} period_low={pivots['period_low']:.2f}"
            )
    
    # Add fallback info to pivots dict for downstream analysis
    pivots['fallback_info'] = fallback_info
    
    return tp_price, sl_price, pivots


def calculate_index_volatility(index_df: pd.DataFrame, date: pd.Timestamp, 
                                lookback: int = DEFAULT_VOL_LOOKBACK) -> float:
    """
    Calculate rolling volatility of the index.
    
    Parameters:
    - index_df: DataFrame with columns ['date', 'value']
    - date: Calculate volatility as of this date
    - lookback: Number of trading days for volatility calculation
    
    Returns:
    - Annualized volatility as a decimal (e.g., 0.20 for 20%)
    - Returns None if insufficient data
    """
    idx_data = index_df[index_df['date'] <= date].sort_values('date').copy()
    
    if len(idx_data) < lookback + 1:
        return None
    
    idx_data = idx_data.tail(lookback + 1)
    idx_data['returns'] = idx_data['value'].pct_change()
    
    daily_vol = idx_data['returns'].std()
    annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return annualized_vol


def calculate_index_ma(index_df: pd.DataFrame, date: pd.Timestamp, 
                       ma_period: int = DEFAULT_REGIME_MA_PERIOD) -> tuple:
    """
    Calculate moving average of the index and current value.
    
    Parameters:
    - index_df: DataFrame with columns ['date', 'value']
    - date: Calculate MA as of this date
    - ma_period: Period for moving average
    
    Returns:
    - Tuple of (current_value, ma_value, pct_from_ma)
    - Returns (None, None, None) if insufficient data
    """
    idx_data = index_df[index_df['date'] <= date].sort_values('date').copy()
    
    if len(idx_data) < ma_period:
        return None, None, None
    
    current_value = idx_data['value'].iloc[-1]
    ma_value = idx_data['value'].tail(ma_period).mean()
    pct_from_ma = (current_value - ma_value) / ma_value
    
    return current_value, ma_value, pct_from_ma


def get_volatility_adjustment_multiplier(index_df: pd.DataFrame, date: pd.Timestamp,
                                          lookback: int = DEFAULT_VOL_LOOKBACK,
                                          high_vol_threshold: float = DEFAULT_HIGH_VOL_THRESHOLD,
                                          low_vol_threshold: float = DEFAULT_LOW_VOL_THRESHOLD,
                                          high_vol_multiplier: float = DEFAULT_HIGH_VOL_MULTIPLIER,
                                          low_vol_multiplier: float = DEFAULT_LOW_VOL_MULTIPLIER) -> float:
    """
    Get a multiplier to adjust TP/SL based on market volatility.
    
    In high volatility markets, we widen TP/SL to avoid getting stopped out.
    In low volatility markets, we tighten TP/SL to capture smaller moves.
    
    Parameters:
    - index_df: DataFrame with index data
    - date: Date to evaluate volatility
    - lookback: Days for volatility calculation
    - high_vol_threshold: Annualized vol above this is "high" (default 25%)
    - low_vol_threshold: Annualized vol below this is "low" (default 15%)
    - high_vol_multiplier: Multiplier in high vol regime
    - low_vol_multiplier: Multiplier in low vol regime
    
    Returns:
    - Multiplier to apply to TP/SL thresholds (1.0 if normal vol)
    """
    vol = calculate_index_volatility(index_df, date, lookback)
    
    if vol is None:
        return 1.0
    
    if vol >= high_vol_threshold:
        return high_vol_multiplier
    elif vol <= low_vol_threshold:
        return low_vol_multiplier
    else:
        return 1.0


def check_regime_exit_signal(index_df: pd.DataFrame, date: pd.Timestamp,
                              ma_period: int = DEFAULT_REGIME_MA_PERIOD, exit_threshold: float = DEFAULT_REGIME_EXIT_THRESHOLD) -> bool:
    """
    Check if market regime suggests exiting positions.
    
    Returns True if the index is significantly below its moving average,
    indicating a potential market downturn.
    
    Parameters:
    - index_df: DataFrame with index data
    - date: Date to check
    - ma_period: Period for moving average
    - exit_threshold: Exit if index is below MA by this percentage (e.g., -0.02 = 2% below)
    
    Returns:
    - True if exit signal is triggered, False otherwise
    """
    current, ma, pct_from_ma = calculate_index_ma(index_df, date, ma_period)
    
    if pct_from_ma is None:
        return False
    
    return pct_from_ma < exit_threshold
