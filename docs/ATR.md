# Average True Range (ATR) Explained

## What is ATR?

**Average True Range (ATR)** is a technical indicator that measures market **volatility** by analyzing the range of price movement over a period. It was developed by J. Welles Wilder Jr. and introduced in his 1978 book *"New Concepts in Technical Trading Systems."*

### The Core Intuition

> ATR tells you "how much does this stock typically move in a day?"

- A stock with ATR of ₹10 typically moves ₹10 per day
- A stock with ATR of ₹50 typically moves ₹50 per day

This is incredibly useful because:
- A ₹5 move means very different things for these two stocks
- For the first stock, ₹5 is a significant 50% of normal movement
- For the second stock, ₹5 is just 10% of normal movement—basically noise

**ATR normalizes volatility**, allowing you to:
- Set stop losses that respect a stock's natural "breathing room"
- Compare volatility across different stocks and price levels
- Avoid being stopped out by normal price fluctuations

---

## The Problem ATR Solves

### Why Not Just Use High - Low?

The simple daily range (High - Low) misses an important scenario: **gaps**.

Consider these two days:

| Day | Open | High | Low | Close |
|-----|------|------|-----|-------|
| Day 1 | 100 | 105 | 98 | 104 |
| Day 2 | 110 | 115 | 108 | 112 |

**Day 2's range**: 115 - 108 = ₹7

But wait—the stock jumped from 104 to 110 overnight! The *actual* movement from Day 1's close was much larger. The simple range of ₹7 understates the true volatility.

**True Range captures this** by considering the previous close.

---

## True Range Calculation

True Range (TR) is the **largest** of these three values:

| Component | Formula | What It Captures |
|-----------|---------|------------------|
| **TR1** | High - Low | Normal intraday range |
| **TR2** | \|High - Previous Close\| | Gap up scenarios |
| **TR3** | \|Low - Previous Close\| | Gap down scenarios |

```
True Range = MAX(TR1, TR2, TR3)
```

### Visual Intuition

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

## ATR Calculation

ATR is simply the **average of True Range** over N periods (typically 14 days).

### Method 1: Simple Moving Average (SMA)

```
ATR = (TR₁ + TR₂ + TR₃ + ... + TRₙ) / N
```

### Method 2: Exponential/Wilder's Smoothing (Original)

Wilder used a smoothing method that gives more weight to recent values:

```
ATR_today = ((ATR_yesterday × (N-1)) + TR_today) / N
```

This is equivalent to an exponential moving average with α = 1/N.

### Our Implementation

In `dynamic_levels.py`, we use the simple average method:

```python
def calculate_atr(price_df, co_name, end_date, period=14):
    # Calculate True Range components
    stock_data['prev_close'] = stock_data['close'].shift(1)
    stock_data['tr1'] = stock_data['high'] - stock_data['low']
    stock_data['tr2'] = abs(stock_data['high'] - stock_data['prev_close'])
    stock_data['tr3'] = abs(stock_data['low'] - stock_data['prev_close'])
    
    # True Range is the max of the three
    stock_data['true_range'] = stock_data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # ATR is the average
    atr = stock_data['true_range'].iloc[1:].mean()
    return atr
```

---

## Step-by-Step Example

Let's calculate 5-day ATR for a stock:

| Day | High | Low | Close | Prev Close | TR1 | TR2 | TR3 | **True Range** |
|-----|------|-----|-------|------------|-----|-----|-----|----------------|
| 1 | 105 | 98 | 102 | — | 7 | — | — | — |
| 2 | 108 | 100 | 106 | 102 | 8 | 6 | 2 | **8** |
| 3 | 112 | 104 | 110 | 106 | 8 | 6 | 2 | **8** |
| 4 | 107 | 99 | 101 | 110 | 8 | 3 | 11 | **11** |
| 5 | 105 | 97 | 103 | 101 | 8 | 4 | 4 | **8** |
| 6 | 115 | 108 | 112 | 103 | 7 | 12 | 5 | **12** |

**5-Day ATR (Days 2-6)** = (8 + 8 + 11 + 8 + 12) / 5 = **9.4**

This tells us the stock typically moves about ₹9.4 per day.

---

## Using ATR for TP/SL

### The Logic

If a stock typically moves ₹10/day (ATR = 10), setting a stop loss ₹5 away is likely to get hit by normal fluctuations. Instead:

- **Stop Loss** = Entry Price - (N × ATR)
- **Take Profit** = Entry Price + (M × ATR)

Common multipliers:
| Multiplier | Use Case |
|------------|----------|
| 1.0 × ATR | Tight stop, quick exits |
| 1.5 × ATR | Standard stop loss |
| 2.0 × ATR | Wider stop, more room |
| 2.5-3.0 × ATR | Swing trading targets |

### Example

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

## ATR Characteristics

### What ATR Tells You

✅ **Volatility magnitude** — How much price typically moves  
✅ **Position sizing** — Adjust position size based on volatility  
✅ **Stop placement** — Set stops that respect natural price movement  

### What ATR Does NOT Tell You

❌ **Direction** — ATR doesn't indicate if price will go up or down  
❌ **Trend** — High ATR can occur in uptrends, downtrends, or sideways markets  
❌ **Overbought/Oversold** — It's purely a volatility measure  

### ATR Behavior Patterns

| Market Condition | ATR Behavior |
|-----------------|--------------|
| Trending strongly | ATR typically increases |
| Consolidating/ranging | ATR typically decreases |
| Before breakout | Often low (calm before storm) |
| After breakout | Spikes higher |
| Market crash | Spikes dramatically |

---

## Common ATR Periods

| Period | Use Case |
|--------|----------|
| **7-day** | Short-term traders, more responsive |
| **14-day** | Standard (Wilder's original), balanced |
| **20-day** | Approximately one trading month |
| **50-day** | Longer-term perspective |

Shorter periods = More responsive to recent volatility  
Longer periods = Smoother, less reactive to short-term spikes

---

## Key Takeaways

1. **ATR measures volatility, not direction** — It tells you *how much* a stock moves, not *which way*

2. **True Range captures gaps** — Unlike simple High-Low range, ATR accounts for overnight gaps

3. **ATR is in price units** — If a stock is at ₹500 with ATR of ₹15, that's 3% daily volatility

4. **Use multipliers for TP/SL** — Typically 1.5-2× ATR for stops, 2-3× for targets

5. **ATR adapts to each stock** — A ₹50 stock and a ₹5000 stock get appropriately sized thresholds

6. **ATR changes over time** — Recalculating periodically can improve adaptiveness (see `atr_threshold_analysis.md`)

---

## References

- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*
- Implementation: `src/dynamic_levels.py` → `calculate_atr()`
- Usage: `src/TP_SL_bt.py` → `calculate_atr_thresholds()`
