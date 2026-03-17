# Mid-Quarter Cash Redeployment

> Identifying opportune dates for redeploying idle cash within a quarter when aggressive TP/SL strategies leave the portfolio sitting mostly in cash.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Design Constraints](#design-constraints)
- [Redeployment Date Identification Framework](#redeployment-date-identification-framework)
- [Retrospective Oracle Analysis](#retrospective-oracle-analysis)
- [Configuration Parameters](#configuration-parameters)
- [Future Work](#future-work)

---

## Problem Statement

### The Idle Cash Problem

alphaBT deploys capital at the start of each quarter and parks exit proceeds in a risk-free vehicle until the next quarter begins (see [CASH_APPRECIATION.md](CASH_APPRECIATION.md) for mechanics). This works well for strategies where positions survive deep into the quarter. However, for strategies with aggressive TP/SL thresholds, the capital utilization profile looks very different.

Consider a strategy with 25 stocks, tight take-profit, and tight stop-loss. A typical quarter unfolds like this:

```
Quarter Start (e.g., Feb 15)
  |
  |-- Day 1-3: Entry phase (positions opened at avg close)
  |
  |-- Day 4-25: Rapid exits
  |     Most stocks hit TP or SL within the first month.
  |     Cash ratio climbs from 0% to 60-80%.
  |
  |-- Day 25-75: Idle phase
  |     Majority of capital sits in cash, earning ~6.5% annualized.
  |     A handful of survivors ride to mandatory exit.
  |
Quarter End (e.g., May 30)
```

By roughly 30 trading days into the quarter, the portfolio is dominated by cash. The remaining ~45 trading days — more than half the quarter — see that cash earning the risk-free rate instead of being put to work in equities.

### Opportunity Cost

The gap between the risk-free rate and equity market returns represents the opportunity cost of idle cash. In a market returning 12-15% annualized, the forgone return on 60-80% of capital for ~2 months is material.

A rough illustration:

| Metric | Value |
|--------|-------|
| Portfolio size | 100 Cr |
| Avg cash ratio after month 1 | 70% |
| Idle capital | ~70 Cr |
| Remaining days in quarter | ~45 trading days |
| Risk-free daily rate (6.5% ann.) | ~0.025% |
| Risk-free return on idle capital | ~0.79 Cr |
| Hypothetical equity return (12% ann.) on same capital | ~1.47 Cr |
| **Opportunity gap per quarter** | **~0.68 Cr** |

Over 36 quarters (9 years), even a partial capture of this gap compounds significantly.

---

## Design Constraints

The redeployment mechanism is governed by the following constraints:

1. **At most one redeployment per quarter.** Multiple rounds would add complexity with diminishing returns — the second tranche itself starts hitting TP/SL, and the remaining runway shrinks with each redeployment.

2. **The redeployment date varies quarter to quarter.** It is not a fixed calendar offset. Market conditions and exit patterns differ across quarters, so the optimal entry point for the second tranche shifts.

3. **Some quarters may have no redeployment at all.** If exit patterns are slow, the market is in a downturn, or there is insufficient runway remaining, we skip the quarter entirely. Doing nothing is a valid outcome.

4. **Sufficient runway must remain.** Redeployed positions need enough time to work. If only 15 trading days remain, aggressive TP/SL will barely have time to trigger, and the overhead is not justified.

5. **Scope boundary.** This document covers only how to *identify the redeployment date*. Two critical downstream questions — how to generate signals for the second tranche and how to extend the simulation framework to model it — are deferred to future work.

---

## Redeployment Date Identification Framework

The redeployment date for a quarter is defined as the **first trading day** on which three criteria are simultaneously satisfied. If no such day exists within the quarter, no redeployment occurs.

```
Redeployment Date = first day where:
    Criterion 1 (Cash Readiness)   AND
    Criterion 2 (Time Window)      AND
    Criterion 3 (Market Condition)
```

### Criterion 1: Cash Readiness

**Purpose:** Ensure a meaningful amount of capital is available before attempting redeployment.

The daily cash ratio is defined as:

$$\text{cash\_ratio}(t) = \frac{\text{Cash\_In\_Hand}(t)}{\text{Total\_Portfolio\_Value}(t)}$$

Both values are already computed by `generate_quarter_equity_curve` in `backtest/simulation.py` for every trading day of every quarter. No new data is needed.

Redeployment becomes eligible on the first day `t` where:

$$\text{cash\_ratio}(t) \geq \theta_{\text{cash}}$$

where $\theta_{\text{cash}}$ is a configurable threshold (e.g., 0.50 to 0.60).

**Rationale for 50-60%:** At 50%, half the portfolio has exited. Below this, the redeployable amount may not justify the effort. Above 60%, we may be waiting unnecessarily long when positions are already done exiting. The exact threshold is a tuning parameter — the retrospective analysis (Section 4) will inform calibration.

**Edge case:** If a quarter has unusually slow exits (e.g., a calm market where few positions hit TP or SL early), the cash ratio may never reach the threshold. This is intentional — the quarter is skipped.

### Criterion 2: Time Window

**Purpose:** Prevent redeployment too early (before positions have had a chance to exit) or too late (insufficient runway for new positions).

The eligible window is defined by two offsets from the quarter boundaries:

| Parameter | Symbol | Description | Suggested Default |
|-----------|--------|-------------|-------------------|
| Earliest eligible day | $d_{\text{earliest}}$ | Trading days after quarter entry start | 15-20 |
| Latest eligible day | $d_{\text{latest\_before\_end}}$ | Trading days before mandatory quarter exit | 25-30 |

Given that quarters span roughly 75 trading days:

```
Quarter:    |---entry---|-------eligible window---------|---buffer---|
Day:        0    3     15-20                         45-50          75
            ^          ^                              ^              ^
         entry      earliest                       latest        mandatory
         window     eligible                       eligible      exit
```

**Why not before day 15-20?** The entry phase itself occupies the first 3 trading days. Most aggressive strategies see the bulk of their TP/SL exits between days 5 and 25. Deploying a second tranche on day 10, when exits are still streaming in, would mean the cash readiness criterion is moving target.

**Why not after day 45-50?** With only 25-30 trading days remaining, the second tranche has limited runway. For a strategy that normally takes ~25 days to trigger TP/SL, deploying on day 55 leaves only ~20 days — positions would disproportionately hit mandatory time exits rather than TP/SL, undermining the strategy's edge.

### Criterion 3: Market Condition

**Purpose:** Avoid deploying into a falling market. Wait for conditions that historically correlate with favorable short-term equity returns.

This is the criterion where judgement matters most. Below are four candidate approaches, each leveraging the existing index analysis infrastructure in `backtest/dynamic_levels.py`. They are not mutually exclusive — the composite option combines them.

#### Option A: Index Mean Reversion

Deploy when the index has pulled back from a recent high and is showing signs of recovery.

```
Conditions (all must hold on day t):
  1. index_value(t) < rolling_max(index, 20 days) × (1 - pullback_threshold)
     i.e., index has pulled back at least X% from its 20-day high
  2. index_value(t) > SMA(index, 5 days)
     i.e., index is recovering (above its short-term average)
```

| Parameter | Description | Suggested Range |
|-----------|-------------|-----------------|
| `pullback_threshold` | Min pullback from 20-day high | 0.02 - 0.05 (2-5%) |
| `recovery_ma_period` | Short MA to confirm recovery | 5 - 10 days |

**Intuition:** Buying after a dip that is already reversing. The pullback creates a more favorable entry price, and the MA crossover confirms the dip is not deepening into a crash.

**Risk:** Whipsaws in choppy markets. The index might dip 3%, bounce briefly above the 5-day MA, then continue falling.

#### Option B: Regime Confirmation

Deploy only when the index is in an uptrend, defined by being above its longer-term moving average.

```
Condition on day t:
  index_value(t) > SMA(index, N days)
```

This is the inverse of the existing `check_regime_exit_signal` in `dynamic_levels.py`, which fires when the index is *below* its MA by more than a threshold. Here we require the index to be *above* its MA.

| Parameter | Description | Suggested Range |
|-----------|-------------|-----------------|
| `regime_ma_period` | MA period for trend confirmation | 20 - 50 days |
| `min_pct_above_ma` | Optional: require index to be at least X% above MA | 0.0 - 0.02 |

**Intuition:** Simple trend-following. If the broad market is in an uptrend, new positions are more likely to benefit from the tailwind.

**Risk:** In a slow grind down that stays near the MA, this criterion might fire repeatedly on false signals. It also does not capture entry at attractive valuations — it could trigger at all-time highs.

Existing infrastructure: `calculate_index_ma(index_df, date, ma_period)` returns `(current_value, ma_value, pct_from_ma)` directly.

#### Option C: Volatility Normalization

Deploy when market volatility has settled from a high-vol regime to normal or low-vol.

```
Conditions on day t:
  1. index_volatility(t) < high_vol_threshold
     i.e., we are NOT in a high-volatility regime
  2. (optional) index_volatility(t - lookback) >= high_vol_threshold
     i.e., we recently WERE in a high-vol regime (vol has come down)
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vol_lookback` | Days for vol calculation | 20 |
| `high_vol_threshold` | Annualized vol considered "high" | 0.25 (25%) |

**Intuition:** High-volatility periods often coincide with drawdowns. Waiting for volatility to normalize means the worst of the turbulence has passed. Entry after a vol spike often captures the recovery.

**Risk:** Volatility can remain low during a slow, grinding decline. Low vol does not guarantee positive returns.

Existing infrastructure: `calculate_index_volatility(index_df, date, lookback)` returns annualized volatility.

#### Option D: Composite

Combine elements from the above to create a higher-conviction signal at the cost of fewer triggers.

**Example composite rule:**

```
Deploy on day t if ALL of:
  1. Index has pulled back >= 3% from 20-day high
  2. Index is above its 10-day MA (short-term recovery)
  3. 20-day annualized volatility < 25% (not in crisis)
```

This filters out both "buying into a crash" (condition 3 eliminates high-vol periods) and "buying a dead cat bounce" (condition 2 requires actual recovery momentum). The pullback requirement (condition 1) ensures we are not buying at the absolute top.

The trade-off is that in quiet bull markets where the index never pulls back 3%, no redeployment trigger fires. This is acceptable — in those quarters, the market condition did not present an identifiable entry opportunity, and the cash continues to earn the risk-free rate.

### Putting It Together

For a given quarter, the evaluation proceeds day by day:

```
for each trading_day t in quarter:
    if t < entry_start + d_earliest:
        continue                          # too early
    if t > mandatory_exit - d_latest_before_end:
        break                             # too late, skip this quarter

    cash_ratio = Cash_In_Hand(t) / Total_Portfolio_Value(t)
    if cash_ratio < theta_cash:
        continue                          # not enough cash freed yet

    if market_condition(t) is True:
        return t                          # redeployment date found

return None                               # no suitable date this quarter
```

The **first** qualifying day is chosen. We do not look for the *best* day — that would require lookahead. The first-qualifying-day rule is implementable in a live setting where each day's decision must be made with information available only up to that point.

---

## Retrospective Oracle Analysis

Before locking in thresholds for the criteria above, we can use the 9-year backtest data to calibrate them. This analysis has two phases: an oracle study (what *would* have been optimal in hindsight) and a forward-looking criterion test (does our rule approximate the oracle).

### Phase 1: Oracle Date Identification

For each quarter in the backtest period:

1. **Run the existing backtest** to produce trade results and daily equity curves.
2. **Extract the cash ratio time series** from the equity curve (`Cash_In_Hand / Total_Portfolio_Value`).
3. **Define the eligible window**: day $d_{\text{earliest}}$ through day $d_{\text{latest\_before\_end}}$ of the quarter, restricted to days where the cash ratio exceeds $\theta_{\text{cash}}$.
4. **For each eligible day $t$**, compute a proxy return: the return of the Nifty 500 index from day $t$ to the quarter's mandatory exit date.

$$R_{\text{proxy}}(t) = \frac{\text{index\_value}(\text{quarter\_end})}{\text{index\_value}(t)} - 1$$

This proxy represents the return the redeployed cash *could* have earned if invested in the broad market. It is deliberately simple — we do not yet have signals for stock selection, so the index return is the best available benchmark.

5. **The oracle date** for the quarter is the eligible day $t^*$ that maximizes $R_{\text{proxy}}(t)$. This is the day we *wish* we had redeployed.

### Phase 2: Feature Analysis of Oracle Dates

Once oracle dates are identified for all quarters, record the following market features on each oracle date:

| Feature | Source |
|---------|--------|
| Index value relative to 20-day MA | `calculate_index_ma(index_df, t, 20)` |
| Index value relative to 50-day MA | `calculate_index_ma(index_df, t, 50)` |
| Pullback from 20-day rolling high | Computed from index data |
| 20-day annualized volatility | `calculate_index_volatility(index_df, t, 20)` |
| Days since quarter entry start | Calendar arithmetic |
| Cash ratio on oracle date | From equity curve |

Analyze the distribution of these features across all oracle dates:

- **Central tendency:** What is the median pullback from the 20-day high on oracle dates? Are oracle dates clustered around a certain volatility regime?
- **Consistency:** Do oracle dates consistently occur when the index is above or below its MA? Or is the pattern noisy?
- **Timing:** Do oracle dates cluster around a particular day offset within the quarter (e.g., always around day 25-35)?

### Phase 3: Criterion Calibration

Use the Phase 2 analysis to set concrete thresholds:

- If oracle dates consistently show pullbacks of 3-5% from the 20-day high, set `pullback_threshold = 0.03`.
- If oracle dates are mostly in low/normal volatility regimes, set `high_vol_threshold = 0.25` and require vol to be below it.
- If oracle dates cluster in days 20-35, tighten the time window accordingly.

Then backtest the rule-based criterion: for each quarter, find the first qualifying day and compare $R_{\text{proxy}}$ at that day against $R_{\text{proxy}}$ at the oracle date. The gap measures how much return we leave on the table with a non-clairvoyant rule.

A rule that captures 60-80% of the oracle's proxy return across most quarters is a good starting point.

### Quarters With No Good Entry

Some quarters will have negative $R_{\text{proxy}}$ for every eligible day (market fell through the second half of the quarter). In these cases, the oracle analysis confirms that redeployment would have *hurt* returns. A well-calibrated market condition criterion should naturally skip these quarters — the market condition signal should not fire when the market is in sustained decline.

Track the false-positive rate: how often does the criterion trigger redeployment in a quarter where the oracle return is negative? This should be minimized.

---

## Configuration Parameters

When this feature is eventually implemented, the following configuration block would live in `strategy_config.yaml`:

```yaml
cash_redeployment:
  enabled: false

  # How many times per quarter can cash be redeployed (capped at 1 for v1)
  max_redeployments_per_quarter: 1

  # Criterion 1: Cash Readiness
  cash_ratio_threshold: 0.60          # Redeployment eligible when 60%+ of portfolio is cash

  # Criterion 2: Time Window
  earliest_day_offset: 20             # No earlier than 20 trading days after quarter start
  latest_day_before_end: 25           # No later than 25 trading days before quarter end

  # Criterion 3: Market Condition
  market_condition: 'composite'       # One of: 'mean_reversion', 'regime', 'volatility', 'composite'
  market_condition_params:
    # Mean reversion parameters (used by 'mean_reversion' and 'composite')
    pullback_threshold: 0.03          # 3% pullback from 20-day high
    pullback_lookback: 20             # Rolling window for recent high
    recovery_ma_period: 10            # Short MA to confirm recovery

    # Regime parameters (used by 'regime' and 'composite')
    regime_ma_period: 20              # MA period for trend confirmation
    min_pct_above_ma: 0.0             # Require index to be at least this % above MA

    # Volatility parameters (used by 'volatility' and 'composite')
    vol_lookback: 20                  # Days for volatility calculation
    high_vol_threshold: 0.25          # Annualized vol above this blocks redeployment
```

These parameters would flow through the `BacktestConfig` schema in `config/schema.py` and be validated by Pydantic, consistent with how all other configuration is handled.

---

## Future Work

The following topics are out of scope for this document and will be addressed separately once the date identification framework is validated:

- **Signal generation for the second tranche.** How to select stocks for redeployment. Options include re-using the same quarterly ML scores (already available in input data), running a fresh inference pass with updated features, or applying a simpler momentum/mean-reversion screen to the existing universe. The choice has data pipeline implications.

- **Position sizing for redeployment.** How much of the idle cash to deploy. Deploying 100% of `Cash_In_Hand` maximizes capital utilization but concentrates risk. A partial deployment (e.g., 70-80% of idle cash) provides a buffer.

- **TP/SL calibration for the second tranche.** With less time remaining in the quarter, the original TP/SL levels may be too wide. Tighter thresholds or a different TP/SL mode (e.g., switching from ATR to flat for the second tranche) may be appropriate.

- **Simulation framework extension.** `generate_quarter_equity_curve` in `backtest/simulation.py` currently models a single entry phase per quarter. Supporting redeployment requires introducing a second entry phase mid-quarter — with its own position sizing, entry price calculation, and daily tracking — while maintaining a unified cash pool and equity curve.

- **Optuna integration.** The redeployment parameters (cash threshold, time window offsets, market condition choice and its sub-parameters) should be added to the hyperparameter search space in `run_tuning.py`, allowing systematic optimization alongside existing TP/SL and selection parameters.

---

See also:
- [`CASH_APPRECIATION.md`](CASH_APPRECIATION.md) — how idle cash is modelled today
- [`STRATEGIC_OPTIONS.md`](STRATEGIC_OPTIONS.md) — overview of all strategy choices
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system architecture and data flow

*Last updated: March 2026*
