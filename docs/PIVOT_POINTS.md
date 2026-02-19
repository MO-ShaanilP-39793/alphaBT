# Pivot Points, Support & Resistance Explained

## What are Support and Resistance?

**Support** and **Resistance** are price levels where buying or selling pressure historically concentrates, causing price to pause, reverse, or consolidate.

### The Core Intuition

> **Support** = A "floor" where buyers step in  
> **Resistance** = A "ceiling" where sellers step in

Think of it like a bouncing ball:
- The floor (support) stops the ball from falling further
- The ceiling (resistance) stops the ball from rising further

### Why Do These Levels Exist?

They exist because of **market psychology and memory**:

| Factor | How It Creates S/R |
|--------|-------------------|
| **Round numbers** | ₹100, ₹500, ₹1000 attract orders (psychological anchoring) |
| **Previous highs/lows** | Traders remember where price reversed before |
| **Entry points** | Traders who bought at ₹100 may sell if price returns there |
| **Stop losses** | Clusters of stop orders create self-fulfilling levels |
| **Institutional orders** | Large orders often sit at key levels |

### Visual Example

```
Price
  ↑
 120 ─────────────── Resistance (price rejected here twice)
     \    /\    /
      \  /  \  /
       \/    \/
  100 ─────────────── Support (price bounced here twice)
  
  Time →
```

---

## What are Pivot Points?

**Pivot Points** are a mathematical method to calculate support and resistance levels using the previous period's High, Low, and Close prices.

### The Core Intuition

> Pivot Points answer: "Based on yesterday's trading, where are the key levels today?"

The central **Pivot Point (P)** represents the "fair value" or equilibrium price. Levels above are resistance (R1, R2, R3), levels below are support (S1, S2, S3).

```
        R3  ─────  Strong Resistance
        R2  ─────  Moderate Resistance  
        R1  ─────  First Resistance
        P   ═════  PIVOT (equilibrium)
        S1  ─────  First Support
        S2  ─────  Moderate Support
        S3  ─────  Strong Support
```

### Why Pivot Points Work

1. **Self-fulfilling prophecy** — Many traders use them, so orders cluster at these levels
2. **Objective calculation** — No subjectivity; everyone calculates the same levels
3. **Adaptability** — Automatically adjusts as price action changes
4. **Historical basis** — Derived from actual traded prices, not arbitrary lines

---

## Classic Pivot Point Calculation

### Step 1: Calculate the Pivot Point

```
Pivot (P) = (High + Low + Close) / 3
```

This is the average of the three key prices—the "center of gravity" of the previous period.

### Step 2: Calculate Resistance Levels

| Level | Formula | Interpretation |
|-------|---------|----------------|
| **R1** | 2 × P - Low | First resistance; most likely to be tested |
| **R2** | P + (High - Low) | Second resistance; adds the full range above pivot |
| **R3** | High + 2 × (P - Low) | Third resistance; extreme bullish target |

### Step 3: Calculate Support Levels

| Level | Formula | Interpretation |
|-------|---------|----------------|
| **S1** | 2 × P - High | First support; most likely to hold |
| **S2** | P - (High - Low) | Second support; subtracts full range from pivot |
| **S3** | Low - 2 × (High - P) | Third support; extreme bearish level |

### Formula Summary

```
Given: H = High, L = Low, C = Close

P  = (H + L + C) / 3

R1 = 2P - L       S1 = 2P - H
R2 = P + (H - L)  S2 = P - (H - L)
R3 = H + 2(P - L) S3 = L - 2(H - P)
```

---

## Step-by-Step Example

### Given Data (Previous Period)

```
High  = ₹520
Low   = ₹480
Close = ₹510
```

### Calculate Pivot Point

```
P = (520 + 480 + 510) / 3
P = 1510 / 3
P = ₹503.33
```

### Calculate Resistance Levels

```
R1 = 2 × 503.33 - 480 = 1006.67 - 480 = ₹526.67
R2 = 503.33 + (520 - 480) = 503.33 + 40 = ₹543.33
R3 = 520 + 2 × (503.33 - 480) = 520 + 46.67 = ₹566.67
```

### Calculate Support Levels

```
S1 = 2 × 503.33 - 520 = 1006.67 - 520 = ₹486.67
S2 = 503.33 - (520 - 480) = 503.33 - 40 = ₹463.33
S3 = 480 - 2 × (520 - 503.33) = 480 - 33.33 = ₹446.67
```

### Result Summary

| Level | Price | Distance from Pivot |
|-------|-------|---------------------|
| R3 | ₹566.67 | +12.6% |
| R2 | ₹543.33 | +7.9% |
| R1 | ₹526.67 | +4.6% |
| **P** | **₹503.33** | **0%** |
| S1 | ₹486.67 | -3.3% |
| S2 | ₹463.33 | -7.9% |
| S3 | ₹446.67 | -11.3% |

---

## Visual Representation

```
Price
  ↑
₹567 ─── R3 ─────────────────── Extreme bullish
         
₹543 ─── R2 ─────────────────── Strong resistance
         
₹527 ─── R1 ─────────────────── First resistance
         
₹503 ═══ P ══════════════════ PIVOT (equilibrium)
         
₹487 ─── S1 ─────────────────── First support
         
₹463 ─── S2 ─────────────────── Strong support
         
₹447 ─── S3 ─────────────────── Extreme bearish
  ↓
```

---

## Using Pivot Points for TP/SL

### For Long Positions (Buying)

| Element | Level to Use | Rationale |
|---------|--------------|-----------|
| **Entry** | Near P or S1 | Buy at support/equilibrium |
| **Stop Loss** | Below S1 or S2 | Exit if support breaks |
| **Take Profit** | R1 or R2 | Target resistance levels |

### For Short Positions (Selling)

| Element | Level to Use | Rationale |
|---------|--------------|-----------|
| **Entry** | Near P or R1 | Sell at resistance/equilibrium |
| **Stop Loss** | Above R1 or R2 | Exit if resistance breaks |
| **Take Profit** | S1 or S2 | Target support levels |

### Example Trade Setup

```
Stock enters position at: ₹505 (near Pivot)
Strategy: Long

Take Profit = R1 = ₹526.67  (+4.3%)
Stop Loss = S1 = ₹486.67    (-3.6%)

Risk:Reward = 3.6% : 4.3% = 1:1.2
```

---

## Our Implementation

In `dynamic_levels.py`, we calculate pivot points using a lookback period rather than just the previous day:

```python
def calculate_pivot_points(price_df, co_name, end_date, lookback_days=60):
    """
    Uses the HIGH, LOW of the entire lookback period
    and the CLOSE of the most recent day
    """
    high = stock_data['high'].max()      # Highest high in period
    low = stock_data['low'].min()        # Lowest low in period  
    close = stock_data['close'].iloc[-1] # Most recent close
    
    pivot = (high + low + close) / 3
    
    # Resistance levels
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    # Support levels
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
```

### Why Use a Longer Lookback?

| Approach | Lookback | Best For |
|----------|----------|----------|
| **Daily pivots** | 1 day | Day trading, intraday |
| **Weekly pivots** | 5 days | Swing trading |
| **Monthly pivots** | ~20 days | Position trading |
| **Extended (60 days)** | 60 days | Quarterly strategies ✓ |

For quarterly holding periods, using 60-day lookback captures more significant support/resistance levels that are likely to hold over weeks/months.

---

## Pivot Point Variations

### Classic (What We Use)
```
P = (H + L + C) / 3
```

### Woodie's Pivots
Gives more weight to the close:
```
P = (H + L + 2C) / 4
```

### Camarilla Pivots
Tighter levels, good for mean reversion:
```
R4 = C + (H-L) × 1.1/2
R3 = C + (H-L) × 1.1/4
...
```

### Fibonacci Pivots
Uses Fibonacci ratios for S/R distances:
```
R1 = P + 0.382 × (H - L)
R2 = P + 0.618 × (H - L)
R3 = P + 1.000 × (H - L)
```

---

## Characteristics of Pivot Levels

### Probability of Price Reaching Each Level

Based on historical analysis, approximately:

| Level | Probability of Being Reached |
|-------|------------------------------|
| R1/S1 | ~70-80% |
| R2/S2 | ~40-50% |
| R3/S3 | ~15-25% |

This is why:
- **R1/S1** are used for conservative targets
- **R2/S2** are used for moderate targets
- **R3/S3** represent extreme moves

### How Price Behaves at Pivot Levels

| Scenario | What Happens |
|----------|--------------|
| **Bounce** | Price reverses at the level (support holds / resistance rejects) |
| **Break** | Price moves through the level; it often becomes the opposite (broken resistance → new support) |
| **Consolidation** | Price hovers around the level before deciding direction |

---

## Support/Resistance Strength Indicators

Not all S/R levels are equal. Stronger levels have:

| Factor | Why It Matters |
|--------|----------------|
| **Multiple touches** | More tests = more traders aware of the level |
| **Higher volume** | More participants = stronger level |
| **Round numbers** | Psychological significance (₹500 > ₹503) |
| **Confluence** | Multiple methods pointing to same level |
| **Recent** | More recent levels are more relevant |

### Confluence Example

If multiple methods give similar levels, that level is stronger:

```
Classic Pivot S1:     ₹486
Fibonacci 38.2%:      ₹488  
Previous swing low:   ₹485
Round number:         ₹490

→ Strong support zone: ₹485-490
```

---

## Handling Edge Cases

Our implementation includes logic for when pivot levels don't make sense:

### Problem: TP Below Entry

If the calculated resistance (R1) is below our entry price, we need to use a higher level:

```python
if tp_price <= entry_price:
    # Try R2, then R3
    for level in ['R2', 'R3']:
        if pivots[level] > entry_price:
            tp_price = pivots[level]
            break
    # Fallback to 5% above entry
    if tp_price <= entry_price:
        tp_price = entry_price * 1.05
```

### Problem: SL Above Entry

Similarly for support levels:

```python
if sl_price >= entry_price:
    # Try S2, then S3
    for level in ['S2', 'S3']:
        if pivots[level] < entry_price:
            sl_price = pivots[level]
            break
    # Fallback to 5% below entry
    if sl_price >= entry_price:
        sl_price = entry_price * 0.95
```

---

## Key Takeaways

1. **Support = floor, Resistance = ceiling** — Price levels where buying/selling pressure concentrates

2. **Pivot Point = equilibrium** — The "fair value" derived from previous H, L, C

3. **Levels are self-fulfilling** — They work partly because many traders use them

4. **R1/S1 most likely to be reached** — Use for conservative targets; R3/S3 are extreme

5. **Longer lookback = stronger levels** — 60-day pivots are more significant than daily pivots for quarterly strategies

6. **Confluence strengthens levels** — Multiple methods pointing to the same price = stronger S/R

7. **Broken levels flip** — Broken resistance often becomes support (and vice versa)

---

## References

- Implementation: `src/dynamic_levels.py` → `calculate_pivot_points()`, `calculate_pivot_thresholds()`
- Usage: `src/backtest/tpsl.py` → `calculate_dynamic_thresholds()` (pivot mode)
- Configuration: `src/strategy_config.yaml` → `pivot_config` section
