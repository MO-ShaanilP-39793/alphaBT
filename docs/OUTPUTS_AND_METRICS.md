# Outputs and Metrics Guide

Complete guide to understanding alphaBT backtest results, reports, and performance metrics.

---

## Table of Contents

- [Output Directory Structure](#output-directory-structure)
- [Core Output Files](#core-output-files)
- [Backtest Report Workbook](#backtest-report-workbook)
- [Performance Metrics Explained](#performance-metrics-explained)
- [Benchmark Metrics Explained](#benchmark-metrics-explained)
- [Interpreting Results](#interpreting-results)
- [Common Patterns and Red Flags](#common-patterns-and-red-flags)

---

## Output Directory Structure

Every backtest creates a timestamped folder in `backtesting_results/`:

```
bacektesting_results/
  run_20260211_094957/          ← Timestamp: YYYYMMDD_HHMMSS
    ├── config_used.yaml        ← Full configuration (traceability)
    ├── backtest_log.txt        ← Complete execution log
    └── backtest_report.xlsx    ← Consolidated workbook (if generate_report: true)
```

**Timestamp format**: `YYYYMMDD_HHMMSS` (year-month-day_hour-minute-second)

**Example**: `run_20260211_094957` = Run on Feb 11, 2026 at 09:49:57

---

## Core Output Files

### 1. config_used.yaml

**Purpose**: Exact configuration used for this backtest (reproducibility).

**Contents**: Complete YAML with all parameters (data paths, selection params, TP/SL settings, etc.)

**Why it matters**: 
- 6 months later, you can reproduce exact results
- Compare configurations across runs
- Verify what settings were actually used

**Tip**: If a backtest performs exceptionally well, save this file with a descriptive name for future reference.

---

### 2. backtest_log.txt

**Purpose**: Complete execution log (console output saved to file).

**Contents**:
- Data loading messages
- Quarter-by-quarter progress
- Stock selection counts
- Trade simulation status
- Warnings (e.g., "Stocks filtered due to insufficient data")
- Final summary statistics

**Example excerpt**:
```
Loading input data from: ../data/inference_data.csv
Loading price data from: ../data/ohlcv.parquet
Loading index data from: ../data/nifty500.csv

Running backtest from quarter 202002 to 202411...

Quarter 202002:
  Selected 30 stocks
  Simulating trades...
  Completed 30 trades (Entry: Feb 15-17, Exit: various)

Quarter 202405:
  Selected 30 stocks
  Simulating trades...
  Completed 30 trades

...

Backtest complete. Results saved.
```

**When to check**: 
- Debugging issues (errors appear here)
- Verifying data was loaded correctly
- Understanding quarter-by-quarter behavior

---

### 3. backtest_report.xlsx

**Generated when**: `generate_report: true` in config (default: `true`)

**Purpose**: Single consolidated Excel workbook containing all metrics, charts, trade data, and analysis. This replaces the previous multi-file output system.

See [REPORT_SHEETS.md](REPORT_SHEETS.md) for detailed documentation of each sheet.

**Sheets overview**:

| Sheet | Contents |
|-------|----------|
| `portfolio_metrics` | CAGR, Sharpe, Sortino, Calmar, Max Drawdown, VaR, etc. |
| `benchmark_metrics` | Alpha, Beta, Tracking Error, Information Ratio, Correlation |
| `periodic_returns` | Summary stats since inception + sub-period breakdowns |
| `rolling_returns` | 1Y, 3Y, 5Y rolling return statistics with probability buckets |
| `calendar_returns` | Calendar year cumulative returns |
| `up_down_months` | Count of positive vs negative months |
| `trailing_returns` | Point-in-time trailing returns (1m, 3m, 6m, 1y, 3y, 5y, 10y) |
| `quarter_analysis` | Per-quarter stock counts, TP/SL counts, category returns |
| `quarterly_alpha` | Per-quarter outperformance vs benchmark |
| `crisis_regimes` | Returns during pre-defined crisis periods (GFC, Covid, etc.) |
| `market_regimes` | Returns during bull/bear/recovery phases |
| `stock_counts_by_mcap` | Stocks per quarter by largecap/midcap/smallcap |
| `charts` | Embedded visualizations (equity curve, drawdown, heatmaps, distributions) |
| `trade_results` | Raw trade-level data (entry/exit prices, returns, TP/SL triggers) |
| `daily_portfolio_values` | Raw daily equity curve time series |
| `portfolio_vs_index` | Daily portfolio vs benchmark comparison |
| `data_quality_issues` | Data validation warnings (if any) |

**Quick Health Check**:
1. Open `backtest_report.xlsx` → `periodic_returns` sheet
2. Look for:
   - **CAGR** (Compound Annual Growth Rate): Is it positive? Higher than the index?
   - **Max Drawdown**: How much did the portfolio fall from peak? (Lower is better)
   - **Sharpe Ratio**: Risk-adjusted return (>1.0 is good, >2.0 is excellent)

---

## Performance Metrics Explained

### Total Return

**Formula**: 
```
Total Return = (Final Value - Initial Value) / Initial Value
```

**Example**: Started with ₹100 Cr, ended with ₹175 Cr → Total Return = 75%

**Interpretation**:
- **Positive**: Portfolio grew
- **Negative**: Portfolio declined
- **vs Index**: Compare to benchmark's total return

**Limitation**: Doesn't account for time (1 year vs 5 years).

---

### CAGR (Compound Annual Growth Rate)

**Formula**:
```
CAGR = (Final Value / Initial Value)^(1 / Years) - 1
```

**Example**: 
- Grew from ₹100 Cr to ₹175 Cr over 4 years
- CAGR = (175/100)^(1/4) - 1 = 0.150 = **15.0% per year**

**Interpretation**:
- **CAGR > Index CAGR**: Outperforming
- **CAGR > 15%** (Indian context): Strong performance
- **CAGR > 25%**: Exceptional (rare to sustain)

**What it means**: "If I compounded at X% per year, I'd reach the final value."

**Why it matters**: Apples-to-apples comparison across different time periods.

---

### Volatility (Annualized)

**Formula**:
```
Daily Volatility = Std Dev of daily returns
Annualized Volatility = Daily Volatility × √252
```
(252 = trading days per year)

**Example**: Daily std dev = 1.5% → Annualized = 1.5% × √252 ≈ **23.8%**

**Interpretation**:
- **Low volatility (<15%)**: Stable, defensive
- **Medium volatility (15-30%)**: Typical equity fund
- **High volatility (>30%)**: Aggressive, risky

**What it means**: "Typical year-to-year fluctuation magnitude"

**Trade-off**: Higher volatility often (but not always) comes with higher returns.

---

### Sharpe Ratio

**Formula**:
```
Sharpe Ratio = (CAGR - Risk-Free Rate) / Volatility
```

**Risk-Free Rate**: **6.5%** (hardcoded for Indian market, represents 10-year G-Sec yield)

**Example**:
- CAGR = 18%, Volatility = 20%, Risk-Free = 6.5%
- Sharpe = (0.18 - 0.065) / 0.20 = **0.575**

**Interpretation**:

| Sharpe Ratio | Meaning |
|--------------|---------|
| **< 0** | Underperforming risk-free rate (bad) |
| **0 to 1** | Positive but mediocre risk-adjusted return |
| **1 to 2** | Good risk-adjusted performance |
| **> 2** | Excellent (institutional quality) |
| **> 3** | Exceptional (rarely sustained) |

**What it means**: "How much extra return am I getting per unit of volatility risk?"

**Why it matters**: Adjusts for risk – a 30% return with 50% vol could be worse than 15% return with 10% vol.

---

### Sortino Ratio

**Formula**:
```
Sortino Ratio = (CAGR - Risk-Free Rate) / Downside Deviation
```

**Downside Deviation**: Standard deviation of **negative returns only** (ignores upside volatility).

**Example**:
- CAGR = 18%, Downside Dev = 12%, Risk-Free = 6.5%
- Sortino = (0.18 - 0.065) / 0.12 = **0.958**

**Interpretation**: Similar to Sharpe, but **higher** (since downside dev < total vol).

**Why it's better than Sharpe**: 
- Sharpe penalizes both upside and downside volatility
- Sortino only penalizes downside (investors don't mind upside volatility)

**Good range**: >1.5 is strong, >2.5 is excellent.

---

### Maximum Drawdown (MDD)

**Formula**:
```
Drawdown_t = (Portfolio_t - Peak_t) / Peak_t
Max Drawdown = min(Drawdown across all t)
```

**Example**:
- Peak: ₹150 Cr (March 2020)
- Trough: ₹105 Cr (April 2020)
- MDD = (105 - 150) / 150 = **-30%**

**Interpretation**:

| Max Drawdown | Assessment |
|--------------|------------|
| **-5% to -10%** | Very conservative (bond-like) |
| **-10% to -20%** | Moderate (defensive equity) |
| **-20% to -40%** | Typical equity strategy |
| **> -40%** | High risk (needs justification) |

**What it means**: "Worst peak-to-trough decline an investor would have experienced."

**Why it matters**: 
- Psychological – can investors stomach a 40% drop?
- Risk management – position sizing depends on drawdown tolerance

---

### Calmar Ratio

**Formula**:
```
Calmar Ratio = CAGR / |Max Drawdown|
```

**Example**:
- CAGR = 18%, MDD = -30%
- Calmar = 0.18 / 0.30 = **0.6**

**Interpretation**:

| Calmar Ratio | Meaning |
|--------------|---------|
| **< 0.5** | High drawdown relative to return (risky) |
| **0.5 to 1.0** | Acceptable for aggressive strategies |
| **1.0 to 2.0** | Good balance |
| **> 2.0** | Excellent (high return, controlled drawdown) |
| **> 5.0** | Exceptional (rare) |

**What it means**: "How many % CAGR do I get per % of max drawdown?"

**Why optimize Calmar**: Balances return AND risk (drawdown), suitable for most strategies.

---

### VaR (Value at Risk) 95%

**Formula**: 5th percentile of daily return distribution.

**Example**: VaR 95% = -2.3% means:
- 95% of days: Loss is less than 2.3%
- 5% of days: Loss exceeds 2.3% (tail risk)

**Interpretation**:
- **VaR > -1%**: Very low daily risk
- **VaR -1% to -2%**: Moderate
- **VaR < -3%**: High tail risk

**Use case**: "On a bad day (1 in 20), I could lose up to X%."

---

## Benchmark Metrics Explained

### Beta

**Formula** (regression):
```
Beta = Cov(Portfolio Returns, Index Returns) / Var(Index Returns)
```

**Interpretation**:

| Beta | Meaning |
|------|---------|
| **< 0.5** | Very defensive (moves less than market) |
| **0.5 to 0.8** | Defensive |
| **0.8 to 1.2** | Market-like |
| **> 1.2** | Aggressive (amplifies market moves) |

**Example**: Beta = 1.5 means:
- Market up 10% → Portfolio expected to be up 15%
- Market down 10% → Portfolio expected to be down 15%

**What it means**: Sensitivity to market movements.

**Context**: 
- Low beta + high alpha = great (independent returns)
- High beta + low alpha = market exposure without skill

---

### Tracking Error

**Formula**:
```
Tracking Error = Annualized Std Dev(Portfolio Return - Index Return)
```

**Example**: Daily excess returns have 1% std dev → TE = 1% × √252 ≈ **15.9%**

**Interpretation**:

| Tracking Error | Strategy Type |
|----------------|---------------|
| **< 2%** | Index hugger (closet indexing) |
| **2% to 5%** | Moderate active management |
| **5% to 10%** | Active management |
| **> 10%** | Very active / high conviction |

**What it means**: "How much does my portfolio's return deviate from the index?"

**Trade-off**: Higher TE = more independent bets (can outperform OR underperform significantly).

---

### Information Ratio (IR)

**Formula**:
```
IR = Annualized Alpha / Tracking Error
```

**Example**:
- Annual alpha = 8%, Tracking Error = 10%
- IR = 0.08 / 0.10 = **0.8**

**Interpretation**:

| Information Ratio | Meaning |
|-------------------|---------|
| **< 0** | Negative alpha (destroying value) |
| **0 to 0.5** | Weak skill |
| **0.5 to 1.0** | Good active management |
| **> 1.0** | Excellent (top-tier manager) |
| **> 1.5** | Exceptional (very rare) |

**What it means**: "How much alpha do I generate per unit of active risk?"

**Why it matters**: 
- IR > 0.5 for 3+ years = genuine skill (not luck)
- Industry standard for evaluating active managers

---

### Correlation

**Formula**: Pearson correlation between portfolio returns and index returns.

**Range**: -1.0 to +1.0

**Interpretation**:

| Correlation | Meaning |
|-------------|---------|
| **> 0.9** | Almost identical to index (closet indexing) |
| **0.7 to 0.9** | High correlation (market-driven) |
| **0.3 to 0.7** | Moderate correlation (some independence) |
| **< 0.3** | Low correlation (independent strategy) |
| **< 0** | Negative correlation (hedge fund-like) |

**Example**: Correlation = 0.65 means moves are somewhat aligned with market but not identical.

---

### Alpha (CAPM)

**Formula**:
```
Portfolio Return = Risk-Free Rate + Beta × (Index Return - Risk-Free Rate) + Alpha
Rearranged:
Alpha = Portfolio Return - [Risk-Free Rate + Beta × (Index Return - Risk-Free Rate)]
```

**Example**:
- Portfolio Return = 18%, Index Return = 12%, Beta = 1.2, Risk-Free = 6.5%
- Expected Return = 6.5% + 1.2 × (12% - 6.5%) = 13.1%
- Alpha = 18% - 13.1% = **+4.9%**

**Interpretation**:
- **Alpha > 0**: Outperformance (skill / strategy edge)
- **Alpha ≈ 0**: Fair compensation for beta risk
- **Alpha < 0**: Underperformance (would be better off in index fund)

**What it means**: "Return attributable to skill, after accounting for market exposure."

---

## Charts (Embedded in backtest_report.xlsx)

The `charts` sheet in `backtest_report.xlsx` contains embedded PNG visualizations:

### Portfolio vs Index Chart

Line plot showing portfolio value vs index fund value over time.

**Ideal pattern**: Portfolio line consistently above index = outperformance; widening gap = increasing alpha.

---

### Drawdown Chart

**X-axis**: Date | **Y-axis**: Drawdown (%)

**How to read**:
- **Y = 0%**: At peak (all-time high)
- **Y < 0%**: Underwater (in drawdown)

**Good pattern**: Shallow drawdowns (< -20%), quick recoveries.
**Bad pattern**: Deep drawdowns (< -40%), prolonged underwater periods.

---

### Monthly Returns Heatmap

**Layout**: Calendar heatmap (years × months grid)

**Colors**: Green = positive months, Red = negative months, Intensity = magnitude.

**Use**: Spot seasonality, identify crisis periods, assess consistency.

---

### Return Distribution

**Type**: Histogram of monthly returns.

**Good pattern**: Right-skewed (more big wins than big losses), positive mean, no extreme left tail.

---

### Additional Charts

- **Growth of Wealth**: Cumulative growth of investment
- **Calendar Year Heatmap**: Year-by-year returns
- **Correlation Heatmap**: Portfolio vs index return correlations
- **Box-Whisker Plot**: Monthly return distribution by period

---

## Interpreting Results

### The Complete Health Check

After a backtest, systematically review:

#### ✅ Step 1: Absolute Performance
- **CAGR**: Is it positive? Higher than risk-free rate (6.5%)?
- **Total Return**: Did capital grow?
- **Max Drawdown**: Can you stomach it? (< -40% ideally)

#### ✅ Step 2: Risk-Adjusted Performance
- **Sharpe Ratio**: > 1.0 at minimum
- **Calmar Ratio**: > 1.0 ideally
- **Sortino Ratio**: Should be higher than Sharpe

#### ✅ Step 3: Benchmark Comparison
- **Alpha**: Positive?
- **CAGR vs Index CAGR**: Outperforming?
- **Information Ratio**: > 0.5 minimum

#### ✅ Step 4: Consistency
- **Quarter Analysis**: Win rate > 50%?
- **Monthly Returns Heatmap**: More months green than red?
- **Rolling Returns**: 12-month rolling stable or erratic?

#### ✅ Step 5: Risk Metrics
- **Volatility**: Acceptable for your risk tolerance?
- **VaR 95%**: Daily worst-case loss manageable?
- **Beta**: Does it match your market exposure goal?

#### ✅ Step 6: Trade-Level Analysis
- **Avg holding period**: Too short (premature TP/SL)? Too long (not exiting)?
- **TP vs SL trigger rates**: Balanced or lopsided?
- **Category performance** (if applicable): One category dragging down?

---

### Example: "Is This Backtest Good?"

**Scenario**: 
- CAGR: 22%
- Sharpe: 1.35
- Calmar: 2.8
- Max DD: -18%
- Alpha: +6%
- IR: 0.85

**Assessment**: **Excellent**

**Why**:
- ✅ CAGR (22%) >> Index (assumed 12%) >> Risk-Free (6.5%)
- ✅ Sharpe (1.35) solid, Calmar (2.8) excellent
- ✅ Drawdown (-18%) manageable
- ✅ Positive alpha (+6%) with strong IR (0.85)
- **Action**: Proceed to production, monitor out-of-sample

---

**Scenario 2**:
- CAGR: 8%
- Sharpe: 0.4
- Calmar: 0.3
- Max DD: -35%
- Alpha: -2%
- IR: -0.3

**Assessment**: **Poor**

**Why**:
- ❌ CAGR (8%) barely above risk-free (6.5%), below index
- ❌ Low Sharpe (0.4), terrible Calmar (0.3)
- ❌ Large drawdown (-35%) for low return
- ❌ Negative alpha (-2%), negative IR → destroying value
- **Action**: Don't use. Review strategy, ML model, or data quality

---

## Common Patterns and Red Flags

### ✅ Good Patterns

**Steady Growth with Shallow Drawdowns**:
- Equity curve: smooth upward slope
- Drawdown chart: frequent returns to 0%, shallow dips
- **Interpretation**: Consistent strategy, good risk management

**High Win Rate with Controlled Losses**:
- Trade results: 60-70% profitable
- Avg loss < avg win (positive expectancy)
- **Interpretation**: Edge is real, TP/SL working

**Outperformance in Multiple Regimes**:
- Crisis regime analysis: Positive alpha in most periods
- Market regime analysis: Works in bull AND bear
- **Interpretation**: Robust strategy, not lucky

**Positive Skew in Return Distribution**:
- More big winners than big losers
- Right tail extends (occasional large gains)
- **Interpretation**: "Cut losses, let winners run" working

---

### 🚩 Red Flags

**Hockey Stick Equity Curve**:
- Flat/declining for most of period, then sudden spike at end
- **Warning**: Likely overfitting or data error (check recent trades)
- **Action**: Inspect the `trade_results` sheet in `backtest_report.xlsx` for anomalies

**Frequent Deep Drawdowns**:
- Drawdown chart shows repeated -30% dips
- **Warning**: High volatility, poor risk management
- **Action**: Tighten SL, add regime filter, or reduce concentration

**Negative Alpha with High Volatility**:
- Underperforming index while taking more risk
- **Warning**: Strategy has no edge, just noise
- **Action**: Abandon or fundamentally redesign

**Lopsided Exit Triggers**:
- 90% SL triggers, 5% TP triggers
- **Warning**: TP too tight or SL too loose (losing on most trades)
- **Action**: Rebalance TP/SL thresholds

**Extreme Concentration**:
- Category performance: One category = 80% of returns
- **Warning**: Not diversified, single-factor risk
- **Action**: Reweight categories or filter stocks

**Too-Good-to-Be-True Metrics**:
- CAGR > 100%, Sharpe > 5, Calmar > 20
- **Warning**: Data error, lookahead bias, or survivorship bias
- **Action**: Audit data quality, check for bugs

---

### Interpreting Crisis Performance

**COVID-19 (Mar-Apr 2020)**:
- Did strategy protect capital (small loss) or crash with market (large loss)?
- **Good**: MDD < -15% when index fell -30%
- **Bad**: MDD > -50% (worse than buy-and-hold)

**Recovery from Crisis**:
- How long to recover to pre-crisis peak?
- **Good**: < 3 months
- **Slow**: > 6 months

**Alpha During Crisis**:
- Positive alpha = defensive strategy
- Negative alpha = vulnerable to macro shocks

---

## Practical Workflow

### After Each Backtest

1. **Open `backtest_report.xlsx`** → `periodic_returns` sheet
2. **Check CAGR, Sharpe, Calmar** (30-second assessment)
3. **If promising**: Review drawdown chart (in `charts` sheet), crisis analysis
4. **If excellent**: Check `trade_results` sheet, analyze category performance
5. **Document findings**: Save config_used.yaml with notes

### Comparing Multiple Backtests

**Create comparison spreadsheet**:

| Config | CAGR | Sharpe | Calmar | Max DD | Alpha | Notes |
|--------|------|--------|--------|--------|-------|-------|
| Run 1 (top-k, flat) | 18% | 1.2 | 2.1 | -22% | +5% | Baseline |
| Run 2 (category, ATR) | 22% | 1.5 | 2.8 | -18% | +6% | Better! |
| Run 3 (pivot) | 15% | 0.9 | 1.2 | -28% | +2% | Worse |

**Winner**: Run 2 (category + ATR) → Use for production.

---

## Next Steps

- **Optimize parameters**: See [TUNING_GUIDE.md](TUNING_GUIDE.md)
- **Understand strategies**: See [STRATEGIC_OPTIONS.md](STRATEGIC_OPTIONS.md)
- **Master configuration**: See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)
- **Get started**: Back to [USER_GUIDE.md](USER_GUIDE.md)

---

*Last updated: February 2026*
