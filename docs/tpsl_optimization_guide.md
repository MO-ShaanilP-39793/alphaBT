# TP/SL Optimization Guide

This guide explains how to optimize Take Profit (TP) and Stop Loss (SL) parameters in the backtesting framework. It covers three main capabilities:

1. **Parameter Optimization** - Find optimal TP/SL values through grid search
2. **Dynamic TP/SL Modes** - Set thresholds based on stock-specific volatility or technical levels
3. **Index-Guided Exits** - Use market conditions to inform exit decisions

---

## Table of Contents

- [Quick Start](#quick-start)
- [TP/SL Modes](#tpsl-modes)
  - [Fixed Mode](#fixed-mode-default)
  - [ATR Mode](#atr-mode)
  - [Pivot Mode](#pivot-mode)
- [Index-Guided Exits](#index-guided-exits)
  - [Regime Filter](#regime-filter)
  - [Volatility Adjustment](#volatility-adjustment)
- [Parameter Optimization](#parameter-optimization)
  - [Running Optimization](#running-optimization)
  - [Interpreting Results](#interpreting-results)
  - [Optimization Metrics](#optimization-metrics)
- [Configuration Reference](#configuration-reference)
- [Best Practices](#best-practices)

---

## Quick Start

### 1. Run with Default Fixed TP/SL
No changes needed—just run the backtest:
```bash
cd src
python backtest_strategy.py
```

### 2. Switch to ATR-Based Dynamic TP/SL
Edit `strategy_config.yaml`:
```yaml
tpsl_mode: 'atr'
atr_config:
  period: 14
  tp_multiplier: 2.0
  sl_multiplier: 1.5
```

### 3. Find Optimal Parameters
```bash
cd src
python optimize_tpsl.py --metric sharpe
```

---

## TP/SL Modes

The `tpsl_mode` setting in `strategy_config.yaml` determines how TP/SL thresholds are calculated.

### Fixed Mode (Default)

**What it does:** Uses static percentage thresholds defined per category.

**Configuration:**
```yaml
tpsl_mode: 'fixed'

TP_CONFIG:
  volatility:
    high_volatility: 0.10    # 10% take profit
    medium_volatility: 0.05  # 5% take profit
    low_volatility: 0.02     # 2% take profit
  mcap:
    largecap: 0.04
    midcap: 0.06
    smallcap: 0.08

SL_CONFIG:
  volatility:
    high_volatility: 0.10
    medium_volatility: 0.10
    low_volatility: 0.10
  mcap:
    largecap: 0.02
    midcap: 0.03
    smallcap: 0.04
```

**When to use:**
- As a baseline for comparison
- When you want consistent, predictable thresholds
- After optimization to lock in optimal values

---

### ATR Mode

**What it does:** Calculates TP/SL based on each stock's Average True Range (ATR), a measure of volatility. This adapts thresholds to each stock's typical price movement.

**Formula:**
```
TP Price = Entry Price + (tp_multiplier × ATR)
SL Price = Entry Price - (sl_multiplier × ATR)
```

**Configuration:**
```yaml
tpsl_mode: 'atr'

atr_config:
  period: 14            # Days for ATR calculation
  tp_multiplier: 2.0    # TP at 2x ATR above entry
  sl_multiplier: 1.5    # SL at 1.5x ATR below entry
```

**Example:**
- Stock entry price: ₹100
- 14-day ATR: ₹5
- TP = 100 + (2.0 × 5) = ₹110 (10% gain)
- SL = 100 - (1.5 × 5) = ₹92.50 (7.5% loss)

**When to use:**
- For volatility-adaptive position management
- When stocks in your universe have widely varying volatilities
- To avoid getting stopped out on volatile stocks while capturing moves on stable ones

**Recommended multipliers:**
| Style | TP Multiplier | SL Multiplier | Risk:Reward |
|-------|---------------|---------------|-------------|
| Conservative | 1.5 | 1.0 | 1.5:1 |
| Balanced | 2.0 | 1.5 | 1.3:1 |
| Aggressive | 3.0 | 1.5 | 2:1 |

---

### Pivot Mode

**What it does:** Uses classic pivot point analysis to set TP at resistance levels and SL at support levels.

**Pivot Point Calculation:**
```
Pivot = (High + Low + Close) / 3

Resistance Levels:
  R1 = 2 × Pivot - Low
  R2 = Pivot + (High - Low)
  R3 = High + 2 × (Pivot - Low)

Support Levels:
  S1 = 2 × Pivot - High
  S2 = Pivot - (High - Low)
  S3 = Low - 2 × (High - Pivot)
```

**Configuration:**
```yaml
tpsl_mode: 'pivot'

pivot_config:
  lookback_days: 60     # Historical period for H/L/C calculation
  tp_level: 'R1'        # Use R1 for take profit (R1, R2, or R3)
  sl_level: 'S1'        # Use S1 for stop loss (S1, S2, or S3)
```

**When to use:**
- For technically-driven exit strategies
- When you believe stocks respect historical support/resistance
- For swing trading approaches

**Level Selection:**
| Level | Distance | Use Case |
|-------|----------|----------|
| R1/S1 | Nearest | Conservative, higher hit rate |
| R2/S2 | Medium | Balanced approach |
| R3/S3 | Farthest | Aggressive, lower hit rate |

---

## Index-Guided Exits

These features use benchmark index data to inform exit decisions.

### Regime Filter

**What it does:** Exits all positions early when the market enters a downturn (index falls below its moving average).

**Configuration:**
```yaml
index_exit:
  regime_filter:
    enabled: true
    ma_period: 20           # 20-day moving average
    exit_threshold: -0.02   # Exit when index is 2% below MA
```

**How it works:**
1. Each trading day, check if index is below its N-day moving average
2. If index is more than `exit_threshold` below MA, trigger exit
3. Exit at the day's closing price

**When to use:**
- To protect against market crashes
- In momentum-based strategies that underperform in downtrends
- When preserving capital is a priority

**Considerations:**
- May cause premature exits in choppy markets
- Increases trade count and transaction costs
- Works best with trend-following strategies

---

### Volatility Adjustment

**What it does:** Dynamically widens or tightens TP/SL based on market volatility.

**Logic:**
- **High volatility market:** Widen thresholds to avoid premature stops
- **Low volatility market:** Tighten thresholds to capture smaller moves

**Configuration:**
```yaml
index_exit:
  vol_adjustment:
    enabled: true
    lookback: 20                  # Days to measure volatility
    high_vol_threshold: 0.25      # 25% annualized vol = "high"
    low_vol_threshold: 0.15       # 15% annualized vol = "low"
    high_vol_multiplier: 1.5      # Widen by 50% in high vol
    low_vol_multiplier: 0.8       # Tighten by 20% in low vol
```

**Example:**
- Base TP/SL: 5% / 5%
- High volatility detected: TP/SL becomes 7.5% / 7.5%
- Low volatility detected: TP/SL becomes 4% / 4%

**When to use:**
- In strategies that suffer from volatile market whipsaws
- When you want adaptive risk management
- Combined with fixed or ATR mode for additional adaptation

---

## Parameter Optimization

The optimization module systematically tests TP/SL combinations to find optimal values.

### Running Optimization

**Basic usage:**
```bash
cd src
python optimize_tpsl.py
```

**With custom parameters:**
```bash
python optimize_tpsl.py \
    --metric sharpe \
    --tp-range 0.02 0.04 0.06 0.08 0.10 0.12 \
    --sl-range 0.02 0.04 0.06 0.08 \
    --output-dir ../optimization_results
```

**Command-line options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `strategy_config.yaml` | Config file path |
| `--metric` | `sharpe` | Metric to optimize |
| `--tp-range` | `0.02 0.04 0.06 0.08 0.10 0.12 0.15` | TP values to test |
| `--sl-range` | `0.02 0.04 0.06 0.08 0.10` | SL values to test |
| `--output-dir` | `../optimization_results` | Output directory |
| `--no-parallel` | (flag) | Disable parallel execution |

### Interpreting Results

The optimizer generates several outputs:

**1. `optimization_results.csv`**
Raw results for all tested combinations:
```
tp,sl,total_return,sharpe,sortino,max_drawdown,win_rate,...
0.02,0.02,12.5,0.85,1.2,-8.5,55.3,...
0.02,0.04,14.2,0.92,1.4,-7.2,52.1,...
...
```

**2. Heatmaps (`heatmap_*.png`)**
Visual representation of the parameter surface. Warmer colors = better performance.

![Heatmap Example](heatmap_sharpe.png)

**3. `optimal_params_summary.yaml`**
Best parameters for each metric:
```yaml
sharpe:
  optimal_tp: 0.06
  optimal_sl: 0.04
  metric_value: 1.23
  total_return: 18.5
  max_drawdown: -12.3
```

### Optimization Metrics

| Metric | Description | Optimize For |
|--------|-------------|--------------|
| `sharpe` | Risk-adjusted return (excess return / volatility) | Best overall risk/reward |
| `sortino` | Like Sharpe but only penalizes downside volatility | Upside capture with downside protection |
| `total_return` | Absolute percentage return | Maximum profit |
| `calmar` | Return / Max Drawdown | Capital preservation |
| `win_rate` | Percentage of profitable trades | Consistency |

**Recommendation:** Start with `sharpe` for balanced optimization, then compare with `calmar` if drawdown control is important.

---

## Configuration Reference

Complete `strategy_config.yaml` settings for TP/SL:

```yaml
# =============================================================================
# TP/SL MODE CONFIGURATION
# =============================================================================

# Mode selection: 'fixed', 'atr', or 'pivot'
tpsl_mode: 'fixed'

# ATR configuration
atr_config:
  period: 14
  tp_multiplier: 2.0
  sl_multiplier: 1.5

# Pivot configuration
pivot_config:
  lookback_days: 60
  tp_level: 'R1'      # R1, R2, R3
  sl_level: 'S1'      # S1, S2, S3

# =============================================================================
# INDEX-GUIDED EXIT CONFIGURATION
# =============================================================================

index_exit:
  regime_filter:
    enabled: false
    ma_period: 20
    exit_threshold: -0.02
  
  vol_adjustment:
    enabled: false
    lookback: 20
    high_vol_threshold: 0.25
    low_vol_threshold: 0.15
    high_vol_multiplier: 1.5
    low_vol_multiplier: 0.8

# =============================================================================
# FIXED TP/SL THRESHOLDS
# =============================================================================

TP_CONFIG:
  volatility:
    high_volatility: 0.10
    medium_volatility: 0.05
    low_volatility: 0.02
  mcap:
    largecap: 0.04
    midcap: 0.06
    smallcap: 0.08

SL_CONFIG:
  volatility:
    high_volatility: 0.10
    medium_volatility: 0.10
    low_volatility: 0.10
  mcap:
    largecap: 0.02
    midcap: 0.03
    smallcap: 0.04
```

---

## Best Practices

### 1. Start with Optimization
Before using dynamic modes, run optimization to understand your baseline:
```bash
python optimize_tpsl.py --metric sharpe
```

### 2. Avoid Overfitting
- Use out-of-sample testing (optimize on quarters 1-4, test on 5-6)
- Prefer robust parameter regions over single optimal points
- If optimal is TP=6%, values from 4-8% should also perform reasonably

### 3. Match Mode to Strategy
| Strategy Type | Recommended Mode |
|---------------|------------------|
| Momentum | ATR (captures trends) |
| Mean Reversion | Fixed or Pivot (known levels) |
| High Frequency | Fixed (consistent execution) |
| Swing Trading | Pivot (technical levels) |

### 4. Combine Features Thoughtfully
Good combinations:
- ATR mode + Volatility adjustment (doubly adaptive)
- Fixed mode + Regime filter (simple with crash protection)

Avoid:
- Pivot mode + Volatility adjustment (conflicting logic)
- Multiple aggressive settings together

### 5. Document Your Choices
Each backtest run saves `config_used.yaml` for reproducibility. Review this when comparing results.

---

## Troubleshooting

**"ATR calculation failed, using fallback"**
- Not enough price history before entry date
- Increase `atr_config.period` or ensure sufficient data

**"Pivot thresholds invalid"**
- Entry price is outside the pivot range
- System automatically adjusts to next level or uses percentage fallback

**Optimization runs slowly**
- Reduce parameter ranges
- Use `--no-parallel` if memory issues
- Consider testing fewer quarters first

**Results vary significantly between runs**
- Check for data quality issues
- Ensure consistent quarter ranges
- Look for stocks with missing price data
