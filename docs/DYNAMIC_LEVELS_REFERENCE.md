# Dynamic Levels Reference

> Consolidated reference for the `backtest/dynamic_levels.py` module — ATR-based, pivot-based, and index-based TP/SL calculations.

---

## Table of Contents

1. [Overview](#overview)
2. [ATR Mode](#atr-mode)
   - [What is ATR?](#what-is-atr)
   - [True Range Calculation](#true-range-calculation)
   - [ATR Calculation](#atr-calculation)
   - [Using ATR for TP/SL](#using-atr-for-tpsl)
   - [ATR Characteristics](#atr-characteristics)
   - [API Reference: ATR Functions](#api-reference-atr-functions)
3. [Pivot Mode](#pivot-mode)
   - [Support and Resistance](#support-and-resistance)
   - [What are Pivot Points?](#what-are-pivot-points)
   - [Classic Pivot Point Calculation](#classic-pivot-point-calculation)
   - [Using Pivot Points for TP/SL](#using-pivot-points-for-tpsl)
   - [Pivot Point Variations](#pivot-point-variations)
   - [Handling Edge Cases](#handling-edge-cases)
   - [API Reference: Pivot Functions](#api-reference-pivot-functions)
4. [Index-Based Functions](#index-based-functions)
5. [Data Requirements](#data-requirements)
6. [Error Handling and Fallbacks](#error-handling-and-fallbacks)
7. [Configuration](#configuration)

---

## Overview

`backtest/dynamic_levels.py` provides utility functions for calculating dynamic support/resistance levels used to set adaptive TP/SL thresholds. This module is used internally by `backtest/tpsl.py` when running in **ATR** or **Pivot** mode.

The module contains three families of functions:

| Family | Purpose | Used When |
|--------|---------|-----------|
| **ATR** | Volatility-based adaptive thresholds | `tp_mode: 'atr'` or `sl_mode: 'atr'` |
| **Pivot** | Support/resistance-based thresholds | `tp_mode: 'pivot'` or `sl_mode: 'pivot'` |
| **Index** | Market regime detection & volatility adjustment | `index_exit` settings enabled |

> **Note**: `tp_mode` and `sl_mode` can be set independently (e.g., `tp_mode: 'atr'` with `sl_mode: 'pivot'`).

---

## ATR Mode

### What is ATR?

**Average True Range (ATR)** is a technical indicator that measures market **volatility** by analyzing the range of price movement over a period. It was developed by J. Welles Wilder Jr. and introduced in his 1978 book *"New Concepts in Technical Trading Systems."*

> ATR tells you "how much does this stock typically move in a day?"

- A stock with ATR of ₹10 typically moves ₹10 per day
- A stock with ATR of ₹50 typically moves ₹50 per day

This is useful because:
- A ₹5 move means very different things for these two stocks
- For the first stock, ₹5 is a significant 50% of normal movement
- For the second stock, ₹5 is just 10% of normal movement — basically noise

**ATR normalizes volatility**, allowing you to:
- Set stop losses that respect a stock's natural "breathing room"
- Compare volatility across different stocks and price levels
- Avoid being stopped out by normal price fluctuations

---

### True Range Calculation

The simple daily range (High - Low) misses **gaps** — overnight jumps between one day's close and the next day's open. True Range captures this.

True Range (TR) is the **largest** of these three values:

| Component | Formula | What It Captures |
|-----------|---------|------------------|
| **TR1** | High - Low | Normal intraday range |
| **TR2** | \|High - Previous Close\| | Gap up scenarios |
| **TR3** | \|Low - Previous Close\| | Gap down scenarios |

```
True Range = MAX(TR1, TR2, TR3)
```

#### Example: Normal Day vs Gap Day

```
Scenario A: Normal Day (no gap)
Yesterday Close: 100
Today: High=105, Low=98

TR1 = 105 - 98 = 7
TR2 = |105 - 100| = 5
TR3 = |98 - 100| = 2
True Range = MAX(7, 5, 2) = 7 ← Same as simple range
```

```
Scenario B: Gap Up Day
Yesterday Close: 100
Today: High=115, Low=108 (opened at 110)

TR1 = 115 - 108 = 7
TR2 = |115 - 100| = 15  ← Captures the gap!
TR3 = |108 - 100| = 8
True Range = MAX(7, 15, 8) = 15 ← Much larger than simple range
```

---

### ATR Calculation

ATR is the **average of True Range** over N periods (typically 14 days).

Our implementation in `dynamic_levels.py` uses the **simple average** method:

```python
def calculate_atr(price_df, co_name, end_date, period=14):
    stock_data['prev_close'] = stock_data['close'].shift(1)
    stock_data['tr1'] = stock_data['high'] - stock_data['low']
    stock_data['tr2'] = abs(stock_data['high'] - stock_data['prev_close'])
    stock_data['tr3'] = abs(stock_data['low'] - stock_data['prev_close'])
    stock_data['true_range'] = stock_data[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr = stock_data['true_range'].iloc[1:].mean()
    return atr
```

#### Step-by-Step Example

5-day ATR calculation:

| Day | High | Low | Close | Prev Close | TR1 | TR2 | TR3 | **True Range** |
|-----|------|-----|-------|------------|-----|-----|-----|----------------|
| 1 | 105 | 98 | 102 | — | 7 | — | — | — |
| 2 | 108 | 100 | 106 | 102 | 8 | 6 | 2 | **8** |
| 3 | 112 | 104 | 110 | 106 | 8 | 6 | 2 | **8** |
| 4 | 107 | 99 | 101 | 110 | 8 | 3 | 11 | **11** |
| 5 | 105 | 97 | 103 | 101 | 8 | 4 | 4 | **8** |
| 6 | 115 | 108 | 112 | 103 | 7 | 12 | 5 | **12** |

**5-Day ATR (Days 2-6)** = (8 + 8 + 11 + 8 + 12) / 5 = **9.4** — the stock typically moves about ₹9.4 per day.

#### Common ATR Periods

| Period | Use Case |
|--------|----------|
| **7-day** | Short-term traders, more responsive |
| **14-day** | Standard (Wilder's original), balanced |
| **20-day** | Approximately one trading month |
| **50-day** | Longer-term perspective |

---

### Using ATR for TP/SL

If a stock typically moves ₹10/day (ATR = 10), setting a stop loss ₹5 away is likely to get hit by normal fluctuations. Instead:

- **Take Profit** = Entry Price + (M × ATR)
- **Stop Loss** = Entry Price - (N × ATR)

Common multipliers:

| Multiplier | Use Case |
|------------|----------|
| 1.0 × ATR | Tight stop, quick exits |
| 1.5 × ATR | Standard stop loss |
| 2.0 × ATR | Wider stop, more room |
| 2.5–3.0 × ATR | Swing trading targets |

#### Example

```
Stock: XYZ
Entry Price: ₹500
14-day ATR: ₹20

Stop Loss (1.5 × ATR): 500 - (1.5 × 20) = ₹470
Take Profit (2.0 × ATR): 500 + (2.0 × 20) = ₹540

Risk: ₹30 (6%)
Reward: ₹40 (8%)
Risk:Reward = 1:1.33
```

---

### ATR Characteristics

| What ATR Tells You | What ATR Does NOT Tell You |
|---|---|
| Volatility magnitude — how much price typically moves | Direction — will price go up or down? |
| Position sizing — adjust size based on volatility | Trend — high ATR can occur in any direction |
| Stop placement — respect natural price movement | Overbought/Oversold — purely a volatility measure |

| Market Condition | ATR Behavior |
|-----------------|--------------|
| Trending strongly | ATR typically increases |
| Consolidating/ranging | ATR typically decreases |
| Before breakout | Often low (calm before storm) |
| After breakout | Spikes higher |
| Market crash | Spikes dramatically |

---

### API Reference: ATR Functions

#### `calculate_atr(price_df, co_name, end_date, period=14)`

Calculates the Average True Range for a stock.

**Parameters:**
- `price_df` — DataFrame with columns `['date', 'co_name', 'open', 'high', 'low', 'close']`
- `co_name` — Stock identifier
- `end_date` — Calculate ATR using data before this date
- `period` — Number of periods for ATR (default: 14)

**Returns:** ATR value as `float`, or `None` if insufficient data.

```python
from backtest.dynamic_levels import calculate_atr

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

**Returns:** Tuple of `(tp_price, sl_price, atr_value)` or `(None, None, None)` if calculation fails.

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

## Pivot Mode

### Support and Resistance

**Support** and **Resistance** are price levels where buying or selling pressure historically concentrates, causing price to pause, reverse, or consolidate.

> **Support** = A "floor" where buyers step in  
> **Resistance** = A "ceiling" where sellers step in

Why these levels exist:

| Factor | How It Creates S/R |
|--------|-------------------|
| **Round numbers** | ₹100, ₹500, ₹1000 attract orders (psychological anchoring) |
| **Previous highs/lows** | Traders remember where price reversed before |
| **Entry points** | Traders who bought at ₹100 may sell if price returns there |
| **Stop losses** | Clusters of stop orders create self-fulfilling levels |
| **Institutional orders** | Large orders often sit at key levels |

---

### What are Pivot Points?

**Pivot Points** are a mathematical method to calculate support and resistance levels using the previous period's High, Low, and Close prices.

> Pivot Points answer: "Based on recent trading, where are the key levels?"

```
        R3    ─────  Strong Resistance
        R2_R3 ─────  Midpoint (R2–R3)
        R2    ─────  Moderate Resistance  
        R1_R2 ─────  Midpoint (R1–R2)
        R1    ─────  First Resistance
        P     ═════  PIVOT (equilibrium)
        S1    ─────  First Support
        S1_S2 ─────  Midpoint (S1–S2)
        S2    ─────  Moderate Support
        S2_S3 ─────  Midpoint (S2–S3)
        S3    ─────  Strong Support
```

Why they work:
1. **Self-fulfilling prophecy** — Many traders use them, so orders cluster at these levels
2. **Objective calculation** — Everyone calculates the same levels
3. **Adaptability** — Automatically adjusts as price action changes
4. **Historical basis** — Derived from actual traded prices

---

### Classic Pivot Point Calculation

```
Given: H = High, L = Low, C = Close

P  = (H + L + C) / 3

R1    = 2P - L           S1    = 2P - H
R1_R2 = (R1 + R2) / 2    S1_S2 = (S1 + S2) / 2
R2    = P + (H - L)      S2    = P - (H - L)
R2_R3 = (R2 + R3) / 2    S2_S3 = (S2 + S3) / 2
R3    = H + 2(P - L)     S3    = L - 2(H - P)
```

#### Step-by-Step Example

**Previous Period:** High = ₹520, Low = ₹480, Close = ₹510

```
P = (520 + 480 + 510) / 3 = ₹503.33

R1 = 2 × 503.33 - 480 = ₹526.67
R2 = 503.33 + (520 - 480) = ₹543.33
R3 = 520 + 2 × (503.33 - 480) = ₹566.67

S1 = 2 × 503.33 - 520 = ₹486.67
S2 = 503.33 - (520 - 480) = ₹463.33
S3 = 480 - 2 × (520 - 503.33) = ₹446.67
```

| Level | Price | Distance from Pivot |
|-------|-------|---------------------|
| R3 | ₹566.67 | +12.6% |
| R2 | ₹543.33 | +7.9% |
| R1 | ₹526.67 | +4.6% |
| **P** | **₹503.33** | **0%** |
| S1 | ₹486.67 | -3.3% |
| S2 | ₹463.33 | -7.9% |
| S3 | ₹446.67 | -11.3% |

#### Probability of Reaching Each Level

Based on historical analysis, approximately:

| Level | Probability |
|-------|-------------|
| R1/S1 | ~70-80% |
| R1_R2 / S1_S2 | Midpoint between R1/S1 and R2/S2 |
| R2/S2 | ~40-50% |
| R2_R3 / S2_S3 | Midpoint between R2/S2 and R3/S3 |
| R3/S3 | ~15-25% |

---

### Using Pivot Points for TP/SL

#### For Long Positions

| Element | Level to Use | Rationale |
|---------|--------------|-----------|
| **Entry** | Near P or S1 | Buy at support/equilibrium |
| **Stop Loss** | S1, S1_S2, S2, S2_S3, or S3 | Exit if support breaks (configurable) |
| **Take Profit** | R1, R1_R2, R2, R2_R3, or R3 | Target resistance (configurable) |

#### Example Trade Setup

```
Entry: ₹505 (near Pivot)
Take Profit = R1 = ₹526.67  (+4.3%)
Stop Loss = S1 = ₹486.67    (-3.6%)
Risk:Reward = 1:1.2
```

### Our Implementation

In `dynamic_levels.py`, we calculate pivot points using a **lookback period** rather than just the previous day:

```python
def calculate_pivot_points(price_df, co_name, end_date, lookback_days=60):
    high = stock_data['high'].max()      # Highest high in period
    low = stock_data['low'].min()        # Lowest low in period
    close = stock_data['close'].iloc[-1] # Most recent close
    
    pivot = (high + low + close) / 3
    # ... R1-R3, S1-S3 from classic formulas; R1_R2, R2_R3, S1_S2, S2_S3 = midpoints
```

| Lookback | Best For |
|----------|----------|
| 1 day | Day trading, intraday |
| 5 days | Swing trading |
| ~20 days | Position trading |
| **60 days** | **Quarterly strategies** (our default) |

For quarterly holding periods, 60-day lookback captures more significant support/resistance levels that are likely to hold over weeks/months.

---

### Pivot Point Variations

| Variant | Pivot Formula | Notes |
|---------|---------------|-------|
| **Classic** (ours) | `(H + L + C) / 3` | Standard, widely used |
| Woodie's | `(H + L + 2C) / 4` | More weight to close |
| Fibonacci | Classic P + Fib ratios for S/R | R1 = P + 0.382×(H-L) |
| Camarilla | Close-based with fixed multipliers | Tighter levels, mean reversion |

---

### Handling Edge Cases

Our implementation includes logic for when pivot levels don't make sense relative to the entry price:

**TP below entry:**
```python
if tp_price <= entry_price:
    # Try next levels: R1_R2, R2, R2_R3, R3
    for level in ['R1_R2', 'R2', 'R2_R3', 'R3']:
        if pivots[level] > entry_price:
            tp_price = pivots[level]
            break
    # Fallback to 5% above entry
    if tp_price <= entry_price:
        tp_price = entry_price * 1.05
```

**SL above entry:**
```python
if sl_price >= entry_price:
    # Try next levels: S1_S2, S2, S2_S3, S3
    for level in ['S1_S2', 'S2', 'S2_S3', 'S3']:
        if pivots[level] < entry_price:
            sl_price = pivots[level]
            break
    # Fallback to 5% below entry
    if sl_price >= entry_price:
        sl_price = entry_price * 0.95
```

---

### API Reference: Pivot Functions

#### `calculate_pivot_points(price_df, co_name, end_date, lookback_days=60)`

Calculates classic pivot points with support and resistance levels.

**Returns:** Dictionary with keys `['pivot', 'R1', 'R1_R2', 'R2', 'R2_R3', 'R3', 'S1', 'S1_S2', 'S2', 'S2_S3', 'S3', 'period_high', 'period_low', 'period_close']`

```python
from backtest.dynamic_levels import calculate_pivot_points

pivots = calculate_pivot_points(price_data, 'INFY', pd.Timestamp('2024-03-15'), lookback_days=60)
print(f"Pivot: ₹{pivots['pivot']:.2f}")
print(f"R1: ₹{pivots['R1']:.2f} | R2: ₹{pivots['R2']:.2f}")
print(f"S1: ₹{pivots['S1']:.2f} | S2: ₹{pivots['S2']:.2f}")
```

---

#### `calculate_pivot_thresholds(price_df, co_name, entry_price, end_date, tp_level='R1', sl_level='S1', lookback_days=60)`

Gets TP/SL prices from pivot levels with automatic adjustment if the selected level is invalid.

- **tp_level**: `'R1'`, `'R1_R2'`, `'R2'`, `'R2_R3'`, or `'R3'`
- **sl_level**: `'S1'`, `'S1_S2'`, `'S2'`, `'S2_S3'`, or `'S3'`

**Returns:** Tuple of `(tp_price, sl_price, pivots_dict)`

---

## Index-Based Functions

These functions detect market regime and adjust thresholds accordingly. Used when `index_exit` settings are enabled in the config.

#### `calculate_index_volatility(index_df, date, lookback=20)`

Calculates rolling annualized volatility of the market index.

**Returns:** Annualized volatility as decimal (e.g., 0.20 = 20%).

---

#### `calculate_index_ma(index_df, date, ma_period=20)`

Calculates moving average of the index.

**Returns:** Tuple of `(current_value, ma_value, pct_from_ma)`

---

#### `get_volatility_adjustment_multiplier(index_df, date, lookback=20, high_vol_threshold=0.25, low_vol_threshold=0.15, high_vol_multiplier=1.5, low_vol_multiplier=0.8)`

Returns a multiplier to adjust TP/SL based on current market volatility.

**Logic:**
- Volatility > `high_vol_threshold` → return `high_vol_multiplier`
- Volatility < `low_vol_threshold` → return `low_vol_multiplier`
- Otherwise → return `1.0`

```python
multiplier = get_volatility_adjustment_multiplier(
    index_data, pd.Timestamp('2024-03-15'),
    high_vol_threshold=0.25, high_vol_multiplier=1.5
)
adjusted_tp = base_tp * multiplier
adjusted_sl = base_sl * multiplier
```

---

#### `check_regime_exit_signal(index_df, date, ma_period=20, exit_threshold=-0.02)`

Checks if market regime suggests exiting positions.

**Returns:** `True` if index is below MA by more than `exit_threshold`, otherwise `False`.

```python
should_exit = check_regime_exit_signal(
    index_data, pd.Timestamp('2024-03-15'),
    ma_period=20, exit_threshold=-0.02
)
```

---

## Data Requirements

### Price Data Format

```
date       | co_name  | open   | high   | low    | close
2024-03-01 | RELIANCE | 2450.0 | 2480.0 | 2440.0 | 2475.0
2024-03-02 | RELIANCE | 2475.0 | 2490.0 | 2460.0 | 2485.0
```

### Index Data Format

```
date       | value
2024-03-01 | 22100.50
2024-03-02 | 22250.75
```

---

## Error Handling and Fallbacks

All functions return `None` (or tuples with `None` values) when:
- Insufficient historical data for calculation
- Stock not found in price data
- Date range has no valid trading days

The `backtest/tpsl.py` module handles these cases by falling back to flat percentage thresholds.

---

## Configuration

ATR and pivot modes are activated via `strategy_config.yaml`:

```yaml
# ATR mode
tp_mode: 'atr'
sl_mode: 'atr'
atr_config:
  period: 14
  tp_multiplier: 2.0
  sl_multiplier: 1.5

# Pivot mode
tp_mode: 'pivot'
sl_mode: 'pivot'
pivot_config:
  lookback_days: 60
  tp_level: 'R1'
  sl_level: 'S1'

# Index exit (optional, works with any mode)
index_exit:
  enabled: true
  ma_period: 20
  exit_threshold: -0.02
```

See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for the full parameter reference.

---

## References

- Source: `src/backtest/dynamic_levels.py`
- Consumer: `src/backtest/tpsl.py` → `calculate_dynamic_thresholds()`
- Config: `src/strategy_config.yaml`
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*
- Potential improvements: [ATR_POTENTIAL_IMPROVEMENTS.md](ATR_POTENTIAL_IMPROVEMENTS.md)

*Last updated: March 2026*
