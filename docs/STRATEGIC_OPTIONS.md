# Strategic Options Guide

This guide explains all the strategic choices available in CCQPF and helps you decide which strategies to use for your portfolio.

---

## Table of Contents

- [Introduction: The Quarterly Rebalancing Concept](#introduction-the-quarterly-rebalancing-concept)
- [Trading Concepts Primer](#trading-concepts-primer)
- [Stock Selection Strategies](#stock-selection-strategies)
- [Exit Strategies](#exit-strategies)
- [Advanced Exit Features](#advanced-exit-features)
- [Strategy Selection Matrix](#strategy-selection-matrix)

---

## Introduction: The Quarterly Rebalancing Concept

CCQPF operates on a **quarterly rebalancing** schedule, mirroring how many institutional portfolios work:

```
Quarter 1 (Feb-May)  → Select stocks → Trade → Exit → Evaluate
Quarter 2 (May-Aug)  → Select stocks → Trade → Exit → Evaluate
Quarter 3 (Aug-Nov)  → Select stocks → Trade → Exit → Evaluate
Quarter 4 (Nov-Feb)  → Select stocks → Trade → Exit → Evaluate
```

**Each quarter**:
1. Your ML model provides probability scores for all available stocks
2. The **selection strategy** chooses which stocks to buy (20-50 typically)
3. Positions are entered over the first N trading days (configurable via `entry_price_window`, default 3)
4. The **exit strategy** determines when to sell each stock
5. All remaining positions are closed at the quarter's mandatory cutoff
6. Process repeats for the next quarter (fresh slate)

**Why quarterly?** Balances:
- ✅ Enough time for moves to materialize (vs monthly churn)
- ✅ Responsive to market changes (vs annual "set and forget")
- ✅ Aligned with earnings seasons and institutional flows

---

## Trading Concepts Primer

If you're coming from a pure ML/data science background, here are essential trading concepts explained simply:

### Take Profit (TP)

**What**: A price threshold that triggers an exit *when a stock gains value*.

**Example**: You buy a stock at ₹100 with TP = 10%
- Stock rises to ₹110 → **Automatically sell** and lock in +10% profit
- Prevents "giving back" gains if the stock later drops

**Analogy**: Like setting a sell order at your target price on a trading app.

### Stop Loss (SL)

**What**: A price threshold that triggers an exit *when a stock loses value*.

**Example**: You buy a stock at ₹100 with SL = 5%
- Stock falls to ₹95 → **Automatically sell** to prevent further losses
- Limits downside risk on any single position

**Analogy**: Your emergency exit – "if I'm wrong, I lose only 5%, not 50%"

**Why it matters**: Even the best ML models are wrong sometimes. Stop losses prevent catastrophic losses on bad predictions.

### Volatility

**What**: How much a stock's price fluctuates day-to-day.

**Measurement**: Standard deviation of daily returns, annualized.
- High volatility stock: ₹100 → ₹115 → ₹92 → ₹108 (swings wildly)
- Low volatility stock: ₹100 → ₹102 → ₹99 → ₹101 (stable)

**Why it matters**: High-volatility stocks need wider TP/SL thresholds (or they'll trigger prematurely). The framework can adapt thresholds to volatility automatically.

### Drawdown

**What**: The decline from a portfolio's peak value to its lowest point before recovering.

**Example**: 
- Portfolio grows: ₹100 → ₹150 (peak)
- Market crash: ₹150 → ₹120 (trough)
- **Max Drawdown** = (120 - 150) / 150 = **-20%**

**Why it matters**: Drawdowns measure pain – how much you'd have lost if you invested at the worst possible time. Lower is better.

### Risk-Adjusted Return

**What**: Performance that accounts for how much risk you took to achieve returns.

**Example**:
- Strategy A: +30% return, 40% volatility → Sharpe Ratio = 0.75
- Strategy B: +20% return, 10% volatility → Sharpe Ratio = 2.0
- **Strategy B is better** (less roller-coaster for similar outcome)

**Common metrics**: Sharpe Ratio, Sortino Ratio, Calmar Ratio (see [OUTPUTS_AND_METRICS.md](OUTPUTS_AND_METRICS.md) for details)

---

## Stock Selection Strategies

Every quarter, you need to decide **which stocks to buy** from the universe your ML model scored. CCQPF offers two main approaches:

### Strategy 1: Category-Based Selection

**Concept**: Diversify across buckets (volatility or market cap) to reduce concentration risk.

#### Volatility-Based Categories

Divides stocks into three groups based on recent volatility:

```yaml
selection_type: 'category_based'
category_scheme: 'volatility'
category_counts: [10, 10, 10]      # 10 from each bucket
category_weights: [0.33, 0.33, 0.34]  # Equal capital allocation
```

**How it works**:
1. Each quarter, calculate volatility for all stocks
2. Find 33rd and 66th percentiles (dynamic terciles)
3. Label stocks: `high_volatility`, `medium_volatility`, `low_volatility`
4. Select top 10 (by ML probability) from *each* category
5. Allocate 33% of capital to each category

**Advantages**:
- ✅ Prevents portfolio from being 100% high-vol small-caps (or 100% boring large-caps)
- ✅ Adapts to market conditions (what's "high volatility" changes over time)
- ✅ No lookahead bias (terciles calculated per quarter using past data only)

**When to use**:
- Your ML model tends to favor one volatility regime
- You want systematic diversification across risk profiles
- Historical backtests show concentration in a single volatility zone

**Required data**: Your inference data must include a `volatility` column (e.g., annualized std dev).

#### Market Cap-Based Categories

Divides stocks by size: large-cap, mid-cap, small-cap.

```yaml
selection_type: 'category_based'
category_scheme: 'mcap'
category_counts: [5, 10, 15]       # More small-caps
category_weights: [0.2, 0.3, 0.5]  # More capital to small-caps
```

**How it works**:
1. Your data includes a `category` column pre-labeled: `largecap`, `midcap`, `smallcap`
2. Select top 5 large-caps, top 10 mid-caps, top 15 small-caps (by ML probability)
3. Allocate 20% capital to large, 30% to mid, 50% to small

**Advantages**:
- ✅ Control over size exposure (e.g., tilt toward small-caps for alpha)
- ✅ Aligned with index definitions (Nifty 50 = large, Next 50 = mid, etc.)
- ✅ Clearer regulatory/liquidity buckets

**When to use**:
- Your strategy thesis targets specific market cap segments
- You have pre-labeled market cap data
- You want to benchmark against cap-specific indices

**Required data**: Inference data must include a `category` column with exact values: `largecap`, `midcap`, `smallcap`.

#### Weighting Schemes

**Within each category**, how should capital be distributed?

**Option A: Use Category Weights** (`'use_category_weights'`)
```yaml
category_based_selection_weighting_scheme: 'use_category_weights'
category_counts: [10, 10, 10]
category_weights: [0.33, 0.33, 0.34]
```

- Divide capital by category first (33% + 33% + 34%)
- Then equal-weight *within* each category
- Result: Each category gets its allocated capital share

**Option B: Equal Weight** (`'equal'`)
```yaml
category_based_selection_weighting_scheme: 'equal'
```

- Ignore `category_weights` parameter
- Allocate equal capital to *all* selected stocks regardless of category
- Result: 30 stocks selected → each gets 1/30th of capital

**Which to choose?**
- Use category weights if you want **strategic tilts** (e.g., more capital to high-conviction categories)
- Use equal weights for **pure diversification** (risk management trumps category exposure)

---

### Strategy 2: Top-K Selection

**Concept**: Simple rank-based selection – pick the top N stocks by score.

```yaml
selection_type: 'top_k'
top_k_config:
  k: 30
  weighting_scheme: 'equal'
selection_method: 'probability'
```

**How it works**:
1. Rank all stocks by selection criterion (see below)
2. Select top `k` stocks (e.g., 30)
3. Allocate capital equally: each gets 1/30th

**Advantages**:
- ✅ **Simplest** approach – no need for category labels
- ✅ Pure "trust the model" – highest probabilities get largest positions
- ✅ Fewer configuration parameters

**When to use**:
- Your ML model is well-calibrated and you trust the probability rankings
- You don't have volatility or market cap labels
- You want to start simple before adding complexity

**Disadvantages**:
- ❌ Can concentrate in one segment (e.g., all small-caps if model favors them)
- ❌ No built-in diversification – relies entirely on model variety

---

### Selection Methods (Ranking Criterion)

**Both** category-based and top-k strategies need to rank stocks within their selection pool. Two methods:

#### Probability Ranking

```yaml
selection_method: 'probability'
```

**What**: Rank stocks directly by ML model probability (higher = better).

**When to use**:
- Your model's probability scores are well-calibrated
- You want to maximize signal strength
- You trust that P(success) = 0.9 is genuinely better than 0.7

**Example**: Stock A (prob=0.85), Stock B (prob=0.72) → A ranks higher.

#### Risk-Adjusted Ranking

```yaml
selection_method: 'risk_adjusted'
```

**What**: Rank by `probability / volatility` ratio.

**When to use**:
- You want to balance conviction with risk
- High-volatility stocks need higher probabilities to justify selection
- You prefer a 0.7 probability stable stock over a 0.75 probability wild stock

**Example**: 
- Stock A: prob=0.80, vol=0.40 → score = 2.0
- Stock B: prob=0.75, vol=0.20 → score = 3.75
- **Stock B ranks higher** despite lower probability

**Required**: Your inference data must include a `volatility` column.

---

### Minimum Probability Threshold

**Optional filter** applied *before* selection:

```yaml
min_prob_threshold: 0.5  # Only consider stocks with prob >= 0.5
# OR
min_prob_threshold: null  # No filtering (use all stocks)
```

**Use case**: If your model outputs probabilities for 500 stocks but many have low scores (<0.4), you may want to exclude "low conviction" predictions from consideration entirely.

**Effect**: Reduces selection universe before ranking/selection happens.

---

## Exit Strategies

Once you've bought stocks, you need a plan for **when to sell**. CCQPF offers five main approaches:

### Strategy 1: Hold Until Quarter End

**Configuration**:
```yaml
tp_enabled: false
sl_enabled: false
```

**How it works**: Buy at quarter start, sell at mandatory cutoff. No early exits.

**Advantages**:
- ✅ **Simplest** – zero parameters to tune
- ✅ Gives stocks maximum time to realize potential
- ✅ Avoids premature exits from noise

**Disadvantages**:
- ❌ No downside protection (hold losers for full quarter)
- ❌ No profit-taking (give back gains if stock surges early then crashes)
- ❌ Higher drawdowns in volatile markets

**When to use**:
- Your ML model has strong multi-month predictive power
- You're benchmarking "pure signal" performance
- You want to establish a baseline before adding exit complexity

---

### Strategy 2: Fixed Percentage TP/SL

**Configuration**:
```yaml
tp_enabled: true
sl_enabled: true
tpsl_mode: 'fixed'

TP_CONFIG:
  volatility:  # or 'mcap' depending on your categories
    high_volatility: 0.10     # 10% TP for high-vol stocks
    medium_volatility: 0.05   # 5% TP for medium-vol
    low_volatility: 0.02      # 2% TP for low-vol

SL_CONFIG:
  volatility:
    high_volatility: 0.10     # 10% SL for high-vol
    medium_volatility: 0.10   # 10% SL for medium-vol (same for all)
    low_volatility: 0.10
```

**How it works**: Each stock gets **category-specific** TP/SL thresholds at entry.

**Example**:
- Buy high-vol stock at ₹100 → TP = ₹110, SL = ₹90
- Buy low-vol stock at ₹500 → TP = ₹510, SL = ₹450

**Advantages**:
- ✅ **Predictable** – you know exact exit prices at entry
- ✅ **Customizable** by category (wider bands for volatile stocks)
- ✅ Easy to backtest and understand

**Disadvantages**:
- ❌ Requires category labels (or use `default_tpsl` for preselected mode)
- ❌ Static – doesn't adapt to changing market conditions
- ❌ May be too tight/loose for specific stocks

**When to use**:
- You have clear category definitions (volatility or mcap)
- You want different risk management for different stock types
- You've backtested optimal thresholds per category

**Note**: If running without categories (e.g., preselected stocks), set `default_tpsl` in config:
```yaml
default_tpsl:
  tp_pct: 0.05
  sl_pct: 0.05
```

---

### Strategy 3: Flat Percentage TP/SL

**Configuration**:
```yaml
tp_enabled: true
sl_enabled: true
tpsl_mode: 'flat'

flat_config:
  tp_pct: 0.05   # 5% TP for ALL stocks
  sl_pct: 0.05   # 5% SL for ALL stocks
```

**How it works**: Every stock gets the **same** TP/SL percentage, regardless of category.

**Advantages**:
- ✅ **Simplest TP/SL approach** – only 2 parameters
- ✅ No category column required
- ✅ Uniform risk management across portfolio

**Disadvantages**:
- ❌ One-size-fits-all (volatile stocks may trigger SL too easily)
- ❌ Doesn't adapt to stock-specific characteristics

**When to use**:
- You don't have category labels or want category-agnostic strategy
- You want a baseline TP/SL test
- Your portfolio is already homogeneous (e.g., all large-caps)

**Tip**: Start here if unsure, then evolve to fixed (category-based) or ATR (volatility-adaptive) if needed.

---

### Strategy 4: ATR-Based TP/SL (Volatility-Adaptive)

**Configuration**:
```yaml
tp_enabled: true
sl_enabled: true
tpsl_mode: 'atr'

atr_config:
  period: 14           # Use 14-day ATR
  tp_multiplier: 2.0   # TP = entry + (2.0 × ATR)
  sl_multiplier: 1.5   # SL = entry - (1.5 × ATR)
```

**What is ATR?** Average True Range – a measure of volatility calculated from recent High-Low ranges. Higher ATR = more volatile stock.

**How it works**:
1. At entry, calculate 14-day ATR for the stock
2. Set TP = `entry_price + (2.0 × ATR)`
3. Set SL = `entry_price - (1.5 × ATR)`

**Example**:
- Stock A (calm): entry = ₹100, ATR = ₹2 → TP = ₹104, SL = ₹97
- Stock B (wild): entry = ₹100, ATR = ₹8 → TP = ₹116, SL = ₹88
- **Wider bands for volatile stocks automatically**

**Advantages**:
- ✅ **Adapts to each stock's volatility** dynamically
- ✅ No category labels needed
- ✅ Prevents premature exits on volatile stocks
- ✅ Rooted in technical analysis (widely used by traders)

**Disadvantages**:
- ❌ More parameters to tune (period, multipliers)
- ❌ Requires sufficient price history for ATR calculation
- ❌ Can create very wide/narrow bands in extreme cases

**When to use**:
- Your portfolio includes stocks with varying volatility profiles
- You want technical-analysis-backed thresholds
- You're willing to optimize ATR parameters (period 7-21, multipliers 1.5-3.5)

**Further reading**: See [atr_explained.md](atr_explained.md) for ATR calculation details and intuition.

---

### Strategy 5: Pivot-Based TP/SL (Support/Resistance)

**Configuration**:
```yaml
tp_enabled: true
sl_enabled: true
tpsl_mode: 'pivot'

pivot_config:
  lookback_days: 60    # Use 60 days of history
  tp_level: 'R1'       # Take profit at Resistance 1
  sl_level: 'S1'       # Stop loss at Support 1
```

**What are Pivots?** Classic technical analysis levels calculated from historical High/Low/Close:
- **Pivot Point (P)** = (High + Low + Close) / 3
- **Resistance levels**: R1, R2, R3 (above pivot)
- **Support levels**: S1, S2, S3 (below pivot)

**How it works**:
1. At entry, calculate pivot levels using `lookback_days` historical data
2. Use specified resistance level as TP (e.g., R1, R2, R3)
3. Use specified support level as SL (e.g., S1, S2, S3)

**Advantages**:
- ✅ Based on **support/resistance** theory (where buyers/sellers congregate)
- ✅ No category labels needed
- ✅ Multiple levels (R1/R2/R3) let you tune aggressiveness

**Disadvantages**:
- ❌ Assumes technical analysis validity (not all quants agree)
- ❌ May set illogical levels if recent range is very wide/narrow
- ❌ Requires price history for calculation

**When to use**:
- You believe in technical analysis / support-resistance
- You want TP/SL levels grounded in historical price action
- You're backtesting multiple pivot strategies (e.g., R1 vs R2 vs R3)

**Further reading**: See [pivot_points_explained.md](pivot_points_explained.md) for formulas and theory.

---

### Independent TP/SL Control

You can **enable/disable TP and SL independently**:

```yaml
# Scenario A: Only stop losses (no profit taking)
tp_enabled: false
sl_enabled: true
# → Limits downside, lets winners run

# Scenario B: Only take profits (no stop losses)
tp_enabled: true
sl_enabled: false
# → Locks in gains, accepts unlimited downside risk (risky!)

# Scenario C: Neither (hold til quarter end)
tp_enabled: false
sl_enabled: false

# Scenario D: Both (most common)
tp_enabled: true
sl_enabled: true
```

**Use case for TP-only**: If your model predicts short-term spikes, you might want to capture them without stopping out on dips.

**Use case for SL-only**: If you trust your model long-term but want to cut disasters, "let winners run" while protecting downside.

---

## Advanced Exit Features

Beyond basic TP/SL, CCQPF offers **index-guided exits** that react to broader market conditions:

### Regime Filter (Market Downturn Protection)

**Configuration**:
```yaml
index_exit:
  regime_filter:
    enabled: true
    ma_period: 20         # 20-day moving average
    exit_threshold: -0.02 # Exit if index < MA by 2%
```

**How it works**:
1. Calculate a moving average of the benchmark index (e.g., Nifty 50's 20-day MA)
2. Each day, check: `(index_value - MA) / MA`
3. If index is 2% or more below its MA → **Exit ALL positions at close**
4. Interpretation: Market is in a downtrend, get to cash

**Example**:
- Index value = 11,760
- 20-day MA = 12,000
- Deviation = (11,760 - 12,000) / 12,000 = **-2.0%**
- **Trigger activated** → Sell all stocks

**Advantages**:
- ✅ **Protects in bear markets** (avoids holding through crashes)
- ✅ Based on momentum (price < MA = downtrend)
- ✅ Systematic (no emotional decisions)

**Disadvantages**:
- ❌ Can whipsaw (false signals in choppy markets)
- ❌ Exits winning positions too early if they buck the trend
- ❌ Re-entry not automatic (must wait for next quarter)

**When to use**:
- Your strategy struggles in bear markets
- You want macro-level risk management
- You're okay with going to cash during market weakness

**Parameters to tune**:
- `ma_period`: Shorter (5-10) = more sensitive, Longer (50-100) = fewer signals
- `exit_threshold`: Tighter (-0.01) = exit earlier, Looser (-0.05) = tolerate more drawdown

---

### Volatility Adjustment (Dynamic TP/SL Scaling)

**Configuration**:
```yaml
index_exit:
  vol_adjustment:
    enabled: true
    lookback: 20              # 20-day vol calculation
    high_vol_threshold: 0.25  # Annualized vol > 25% = high
    low_vol_threshold: 0.15   # Annualized vol < 15% = low
    high_vol_multiplier: 1.5  # Widen TP/SL by 50%
    low_vol_multiplier: 0.8   # Tighten TP/SL by 20%
```

**How it works**:
1. Calculate index volatility over `lookback` days (annualized std dev)
2. Classify market state:
   - **High volatility**: vol > 25% → Apply 1.5× multiplier to TP/SL
   - **Low volatility**: vol < 15% → Apply 0.8× multiplier to TP/SL
   - **Normal volatility**: Between thresholds → No adjustment (1.0×)
3. Scale thresholds accordingly

**Example**:
- Base TP = 5%, Base SL = 5%
- **Calm market** (vol = 12%): TP = 4%, SL = 4% (tighter)
- **Volatile market** (vol = 30%): TP = 7.5%, SL = 7.5% (wider)

**Advantages**:
- ✅ **Adapts to market regime** (wider bands in crisis, tighter in stability)
- ✅ Reduces premature exits in volatile periods
- ✅ Captures smaller moves in calm periods

**Disadvantages**:
- ❌ More parameters to tune
- ❌ Can amplify losses in high-vol if you widen SL too much
- ❌ Requires thoughtful threshold/multiplier selection

**When to use**:
- Market volatility varies significantly over your backtest period
- You want to avoid getting stopped out during market-wide volatility spikes
- You're testing "when VIX is high, widen stops" type strategies

---

### Combining Regime Filter + Volatility Adjustment

You can enable **both simultaneously**:

```yaml
index_exit:
  regime_filter:
    enabled: true
    ma_period: 20
    exit_threshold: -0.02
  vol_adjustment:
    enabled: true
    lookback: 20
    high_vol_threshold: 0.25
    low_vol_threshold: 0.15
    high_vol_multiplier: 1.5
    low_vol_multiplier: 0.8
```

**Effect**: Volatility adjustment modifies TP/SL levels daily, while regime filter can force exits if market crashes.

**Exit priority** (checked daily):
1. **Stop Loss** (widened/tightened by vol adjustment)
2. **Regime Exit** (if index trigger fires)
3. **Take Profit** (widened/tightened by vol adjustment)
4. **Quarter End Cutoff**

---

## Strategy Selection Matrix

Use this table to choose strategies based on your goals:

| Your Situation | Recommended Selection | Recommended Exit | Rationale |
|----------------|----------------------|------------------|-----------|
| **Just getting started** | Top-K (k=30) | Flat TP/SL (5%/5%) | Simplest, no labels needed |
| **Have volatility data** | Category-based (volatility) | ATR-based | Diversify by risk, adaptive exits |
| **Have market cap labels** | Category-based (mcap) | Fixed TP/SL by category | Sector-style allocation |
| **Model is very confident** | Top-K (probability) | Hold til quarter end | Trust signals, max time horizon |
| **Model has false positives** | Top-K with `min_prob_threshold` | Fixed or Flat TP/SL | Filter weak signals, manage risk |
| **Portfolio concentrates in one segment** | Category-based (equal weights) | Fixed TP/SL by category | Force diversification |
| **Stocks have very different volatility** | Top-K (risk-adjusted) | ATR-based | Balance risk across positions |
| **Bear markets kill performance** | Any selection | Enable regime filter | Tactical exits in downturns |
| **Market volatility varies a lot** | Any selection | Enable vol adjustment | Adapt bands to conditions |
| **Want technical analysis edge** | Any selection | Pivot-based | Support/resistance levels |
| **Optimizing for Sharpe ratio** | Category-based (volatility) | ATR + vol adjustment | Risk-adjusted throughout |
| **Optimizing for pure returns** | Top-K (probability) | Wide TP, tight SL | Ride winners, cut losers fast |
| **Minimizing drawdowns** | Category-based (equal weights) | Flat TP/SL + regime filter | Diversify + macro protection |
| **Testing pure ML model skill** | Top-K (probability) | Hold til quarter end | Remove exit strategy noise |

---

## Quick Reference: Decision Tree

```
START: Do you have category labels?
│
├─ NO → Use Top-K selection
│   │
│   ├─ Trust model probabilities? 
│   │   ├─ YES → selection_method: 'probability'
│   │   └─ NO → selection_method: 'risk_adjusted' (needs volatility column)
│   │
│   └─ Exit strategy?
│       ├─ Simple → tpsl_mode: 'flat'
│       ├─ Volatility-aware → tpsl_mode: 'atr'
│       ├─ Technical → tpsl_mode: 'pivot'
│       └─ Pure signal → tp_enabled/sl_enabled: false
│
└─ YES → Use Category-Based selection
    │
    ├─ What kind of categories?
    │   ├─ Volatility → category_scheme: 'volatility' (needs volatility column)
    │   └─ Market cap → category_scheme: 'mcap' (needs category: largecap/midcap/smallcap)
    │
    ├─ How to weight?
    │   ├─ Strategic tilt → category_based_selection_weighting_scheme: 'use_category_weights'
    │   └─ Pure diversification → category_based_selection_weighting_scheme: 'equal'
    │
    └─ Exit strategy?
        ├─ Category-specific → tpsl_mode: 'fixed' (requires TP_CONFIG/SL_CONFIG)
        ├─ Uniform → tpsl_mode: 'flat'
        ├─ Volatility-aware → tpsl_mode: 'atr'
        └─ Technical → tpsl_mode: 'pivot'
```

---

## Next Steps

- **Configure your strategy**: See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for detailed parameter syntax
- **Run experiments**: Use [TUNING_GUIDE.md](TUNING_GUIDE.md) to systematically test multiple strategies
- **Interpret results**: See [OUTPUTS_AND_METRICS.md](OUTPUTS_AND_METRICS.md) to understand which metrics matter

---

*Last updated: February 2026*
