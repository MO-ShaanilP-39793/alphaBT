# ATR Threshold Analysis: Static vs Rolling

## Overview

This document analyzes how ATR-based Take Profit (TP) and Stop Loss (SL) thresholds are calculated in the backtesting system, and provides recommendations for potential improvements.

---

## Current Implementation: Static Thresholds

The current implementation in `src/TP_SL_bt.py` computes ATR-based thresholds **once at entry** and keeps them **static throughout the trade**.

### How It Works

1. **Entry Calculation**: The entry price is calculated as the 3-day average at the start of the quarter window.

2. **Threshold Calculation**: `calculate_dynamic_thresholds()` is called once with the entry date:
   ```python
   tp_price, sl_price, threshold_metadata = calculate_dynamic_thresholds(
       price_df, co_name, entry_price, entry_date,
       category_scheme, cat, tpsl_mode, index_df
   )
   ```

3. **Static Monitoring**: The monitoring loop checks each day against the **same** `tp_price` and `sl_price` values:
   ```python
   for idx, day_data in monitoring_df.iterrows():
       if day_data['low'] <= sl_price:    # Static SL
           # Exit triggered
       if day_data['high'] >= tp_price:   # Static TP
           # Exit triggered
   ```

The ATR is calculated using price data **up to the entry date** and never recalculated as the trade progresses.

---

## Should Thresholds Be Rolling/Continuously Updated?

### Arguments FOR Rolling ATR Thresholds

| Benefit | Rationale |
|---------|-----------|
| **Adaptive to volatility changes** | A stock may become more/less volatile after entry. Static thresholds ignore this evolution. |
| **Trailing stop capability** | Rolling ATR allows SL to "lock in" gains as price advances, protecting profits. |
| **Better risk management** | In a calming market, tighter stops preserve more profit; in volatile markets, wider stops prevent premature exits. |
| **Industry standard** | Chandelier Exit and ATR Trailing Stops are well-known strategies that rely on rolling ATR calculations. |

### Arguments AGAINST Rolling Thresholds

| Concern | Rationale |
|---------|-----------|
| **Whipsaw risk** | Constantly tightening stops may shake you out on normal price retracements. |
| **Original thesis violation** | Entry was based on a certain risk/reward profile; changing thresholds mid-trade alters the original bet. |
| **Complexity & overfitting** | More parameters (update frequency, lookback, ratchet rules) create more opportunities to overfit to historical data. |
| **Backtesting integrity** | Rolling calculations require careful implementation to avoid look-ahead bias. |

---

## Recommendation: Hybrid Approach

A balanced approach is to **keep TP static** while implementing a **trailing SL** based on ATR:

### Take Profit: Keep Static
- You identified a price target at entry based on market conditions
- Rolling TP upward constantly chases a moving target
- May cause premature exits during consolidation before the target is reached

### Stop Loss: Implement Trailing (Ratchet-Up Only)
- Recalculate `SL = Current Close - (N × ATR)` daily
- Only update SL if the new value is **higher** than the previous SL (never lower)
- This locks in gains as the trade progresses while respecting current volatility

### Summary Table

| Aspect | Current | Recommendation |
|--------|---------|----------------|
| **TP** | Static (computed once) | Keep static |
| **SL** | Static (computed once) | Rolling, ratchet-up only |
| **ATR Recalc** | Once at entry | Daily for SL |

---

## Note: ATR Calculation Method

The current implementation uses **Simple Moving Average (SMA)** of True Range, not Wilder's original exponential smoothing (`ATR_today = ((ATR_prev × (N-1)) + TR_today) / N`).

**This is fine for static thresholds** because:
- No previous ATR exists at entry (SMA is cleaner for point-in-time calculations)
- SMA is more stable, not overweighting recent spikes

**If implementing trailing stops**, consider switching to Wilder's EMA for faster response to changing volatility.

---

## Implementation Concept

### Trailing Stop Loss Function

```python
def calculate_trailing_sl(current_price, price_df, co_name, current_date, 
                          sl_multiplier, atr_period, current_sl):
    """
    Calculate ATR-based trailing stop, only moving UP (never down).
    
    Parameters:
    - current_price: Current closing price
    - price_df: DataFrame with price data
    - co_name: Stock name
    - current_date: Date for ATR calculation
    - sl_multiplier: ATR multiplier for SL distance
    - atr_period: Period for ATR calculation
    - current_sl: Current stop loss level
    
    Returns:
    - Updated stop loss (only higher, never lower)
    """
    atr = calculate_atr(price_df, co_name, current_date, atr_period)
    
    if atr is None:
        return current_sl  # Keep existing SL if ATR unavailable
    
    new_sl = current_price - (sl_multiplier * atr)
    
    # Ratchet: only update if new SL is higher (tighter)
    return max(new_sl, current_sl)
```

### Modified Monitoring Loop

```python
for idx, day_data in monitoring_df.iterrows():
    current_date = day_data['date']
    
    # Update trailing SL (only moves up, never down)
    sl_price = calculate_trailing_sl(
        day_data['close'], price_df, co_name, current_date,
        sl_multiplier=ATR_CONFIG.get('sl_multiplier', 1.5),
        atr_period=ATR_CONFIG.get('period', 14),
        current_sl=sl_price
    )
    
    # Check Stop Loss (with potentially updated SL)
    if day_data['low'] <= sl_price:
        exit_date = current_date
        exit_price = sl_price
        sl_triggered = True
        break
        
    # Check Take Profit (static TP)
    if day_data['high'] >= tp_price:
        exit_date = current_date
        exit_price = tp_price
        tp_triggered = True
        break
```

---

## Configuration Options

If implementing trailing stops, consider adding these config parameters:

```yaml
atr_config:
  period: 14
  tp_multiplier: 2.0
  sl_multiplier: 1.5
  
  # Trailing stop configuration
  trailing_sl:
    enabled: true
    update_frequency: 'daily'  # 'daily' or 'weekly'
    ratchet_only: true         # Only move SL up, never down
    min_profit_to_trail: 0.0   # Only start trailing after X% profit (optional)
```

---

## Testing Considerations

Before implementing rolling thresholds in production:

1. **A/B Backtest**: Compare static vs trailing SL across multiple quarters
2. **Parameter Sensitivity**: Test different ATR periods and multipliers
3. **Market Regime Analysis**: Evaluate performance in trending vs ranging markets
4. **Transaction Cost Impact**: More frequent exits may increase slippage/costs

---

## Conclusion

The current static threshold implementation is simpler but doesn't adapt to changing volatility during a trade. A hybrid approach with static TP and trailing SL (ratchet-up only) offers a balanced improvement that:

- Protects accumulated gains
- Adapts to volatility changes
- Maintains the original profit target
- Avoids over-complication

This enhancement would be particularly valuable for the quarterly holding periods used in this strategy, where volatility can change significantly over 2-3 months.
