# Dynamic Levels Module Documentation

## Overview

`dynamic_levels.py` provides utility functions for calculating dynamic support/resistance levels used to set adaptive TP/SL thresholds. This module is used internally by `backtest/tpsl.py` when running in ATR or Pivot mode.

---

## Functions

### ATR (Average True Range) Functions

#### `calculate_atr(price_df, co_name, end_date, period=14)`

Calculates the Average True Range for a stock, measuring its typical price volatility.

**Parameters:**
- `price_df`: DataFrame with columns `['date', 'co_name', 'open', 'high', 'low', 'close']`
- `co_name`: Stock identifier
- `end_date`: Calculate ATR using data before this date
- `period`: Number of periods for ATR (default: 14)

**Returns:** ATR value as float, or `None` if insufficient data

**Example:**
```python
from dynamic_levels import calculate_atr
import pandas as pd

atr = calculate_atr(price_data, 'RELIANCE', pd.Timestamp('2024-03-15'), period=14)
print(f"14-day ATR: ₹{atr:.2f}")
```

---

#### `calculate_atr_thresholds(price_df, co_name, entry_price, end_date, tp_multiplier=2.0, sl_multiplier=1.5, period=14)`

Calculates TP and SL prices based on ATR.

**Formula:**
```
TP Price = Entry Price + (tp_multiplier × ATR)
SL Price = Entry Price - (sl_multiplier × ATR)
```

**Returns:** Tuple of `(tp_price, sl_price, atr_value)` or `(None, None, None)` if calculation fails

**Example:**
```python
tp, sl, atr = calculate_atr_thresholds(
    price_data, 'RELIANCE', 
    entry_price=2500.0, 
    end_date=pd.Timestamp('2024-03-15'),
    tp_multiplier=2.0,
    sl_multiplier=1.5
)
print(f"Entry: ₹2500 | TP: ₹{tp:.2f} | SL: ₹{sl:.2f}")
```

---

### Pivot Point Functions

#### `calculate_pivot_points(price_df, co_name, end_date, lookback_days=60)`

Calculates classic pivot points with support and resistance levels.

**Formulas:**
```
Pivot = (High + Low + Close) / 3

R1 = 2 × Pivot - Low
R2 = Pivot + (High - Low)
R3 = High + 2 × (Pivot - Low)

S1 = 2 × Pivot - High
S2 = Pivot - (High - Low)
S3 = Low - 2 × (High - Pivot)
```

**Returns:** Dictionary with keys `['pivot', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', 'period_high', 'period_low', 'period_close']`

**Example:**
```python
pivots = calculate_pivot_points(price_data, 'INFY', pd.Timestamp('2024-03-15'), lookback_days=60)
print(f"Pivot: ₹{pivots['pivot']:.2f}")
print(f"R1: ₹{pivots['R1']:.2f} | R2: ₹{pivots['R2']:.2f}")
print(f"S1: ₹{pivots['S1']:.2f} | S2: ₹{pivots['S2']:.2f}")
```

---

#### `calculate_pivot_thresholds(price_df, co_name, entry_price, end_date, tp_level='R1', sl_level='S1', lookback_days=60)`

Gets TP/SL prices from pivot levels.

**Returns:** Tuple of `(tp_price, sl_price, pivots_dict)`

**Automatic Adjustment:** If the selected level is invalid (e.g., R1 is below entry price), the function automatically selects the next appropriate level or falls back to a 5% threshold.

---

### Index-Based Functions

#### `calculate_index_volatility(index_df, date, lookback=20)`

Calculates rolling volatility of the market index.

**Parameters:**
- `index_df`: DataFrame with columns `['date', 'value']`
- `date`: Calculate volatility as of this date
- `lookback`: Rolling window in trading days

**Returns:** Annualized volatility as decimal (e.g., 0.20 = 20%)

---

#### `calculate_index_ma(index_df, date, ma_period=20)`

Calculates moving average of the index.

**Returns:** Tuple of `(current_value, ma_value, pct_from_ma)`

---

#### `get_volatility_adjustment_multiplier(index_df, date, lookback=20, high_vol_threshold=0.25, low_vol_threshold=0.15, high_vol_multiplier=1.5, low_vol_multiplier=0.8)`

Returns a multiplier to adjust TP/SL based on current market volatility.

**Logic:**
- If volatility > `high_vol_threshold`: return `high_vol_multiplier`
- If volatility < `low_vol_threshold`: return `low_vol_multiplier`
- Otherwise: return `1.0`

**Example:**
```python
multiplier = get_volatility_adjustment_multiplier(
    index_data, 
    pd.Timestamp('2024-03-15'),
    high_vol_threshold=0.25,
    high_vol_multiplier=1.5
)

# Adjust thresholds
adjusted_tp = base_tp * multiplier
adjusted_sl = base_sl * multiplier
```

---

#### `check_regime_exit_signal(index_df, date, ma_period=20, exit_threshold=-0.02)`

Checks if market regime suggests exiting positions.

**Returns:** `True` if index is below MA by more than `exit_threshold`, otherwise `False`

**Example:**
```python
should_exit = check_regime_exit_signal(
    index_data,
    pd.Timestamp('2024-03-15'),
    ma_period=20,
    exit_threshold=-0.02  # Exit if 2% below MA
)

if should_exit:
    print("Market downturn detected - exit signal triggered")
```

---

## Usage in backtest/tpsl.py

These functions are called automatically based on the `tp_mode` and `sl_mode` settings:

```yaml
# strategy_config.yaml
tp_mode: 'atr'    # Uses calculate_atr_thresholds() for take-profit
sl_mode: 'atr'    # Uses calculate_atr_thresholds() for stop-loss
# OR
tp_mode: 'pivot'  # Uses calculate_pivot_thresholds() for take-profit
sl_mode: 'pivot'  # Uses calculate_pivot_thresholds() for stop-loss
```

**Note**: `tp_mode` and `sl_mode` can be set independently (e.g., `tp_mode: 'atr'` with `sl_mode: 'pivot'`).

The index functions are used when `index_exit` settings are enabled.

---

## Data Requirements

### Price Data Format
```
date       | co_name  | open   | high   | low    | close
2024-03-01 | RELIANCE | 2450.0 | 2480.0 | 2440.0 | 2475.0
2024-03-02 | RELIANCE | 2475.0 | 2490.0 | 2460.0 | 2485.0
...
```

### Index Data Format
```
date       | value
2024-03-01 | 22100.50
2024-03-02 | 22250.75
...
```

---

## Error Handling

All functions return `None` (or tuples with `None` values) when:
- Insufficient historical data for calculation
- Stock not found in price data
- Date range has no valid trading days

The `backtest/tpsl.py` module handles these cases by falling back to flat percentage thresholds.
