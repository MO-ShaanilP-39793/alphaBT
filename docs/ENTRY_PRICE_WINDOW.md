# Entry Price Window

> How the entry price window works in alphaBT, how it ties to the ML target, and why splitting entry across three trading days aligns the strategy with the quarterly signal and reduces short-term noise.

---

## 1. Purpose

The **entry price window** is the number of trading days at the start of each quarter over which the strategy’s **entry price** is computed. The default is **3**: the entry price is the **mean of the closing prices** over the first three trading days of the quarter (the first three days after the SEBI result-posting deadline).

This matters because:

- The ML model is trained on targets built from that same “entry” level (first 3 days’ average close). Backtest and live must use the same convention for train–deploy consistency.
- The signal is **quarterly** (fundamentals + momentum over a quarter). The entry price should reflect a level we can trade at without being overly driven by **short-term sentiment or single-day noise**. Using a multi-day average smooths that out and aligns the realized entry with the quarterly nature of the signal.

This document explains how the entry window is implemented, how it links to target construction, and the rationale for using three days.

---

## 2. How It Works in the Codebase

### Configuration

- **Config key**: `entry_price_window` in the strategy YAML (e.g. [`strategy_config_template.yaml`](../src/strategy_config_template.yaml), lines 104–111).
- **Default**: `3` (defined in [`config/defaults.py`](../src/config/defaults.py) as `DEFAULT_ENTRY_PRICE_WINDOW`).
- **Validation**: Must be an integer ≥ 1 ([`config/schema.py`](../src/config/schema.py)); any value ≥ 1 is valid.

Example:

```yaml
# Number of trading days used to calculate the entry price.
# Entry price = mean closing price over the first N trading days of the quarter.
entry_price_window: 3
```

### Entry resolution

Entry is computed in [`src/backtest/tpsl.py`](../src/backtest/tpsl.py) by `_resolve_entry()` (lines 299–316):

1. **Quarter start**: The “quarter” start date is the first trading day after the SEBI filing deadline for that period (see [`utils/quarter.py`](../src/utils/quarter.py)). That date is `entry_start_limit`.
2. **Window**: The code takes the first **N** trading days where `date >= entry_start_limit`, with `N = entry_price_window`.
3. **Entry price**: `entry_price = mean(close)` over those N days.
4. **Entry date**: The calendar date assigned to the trade is the **middle day** of the window: `entry_window.iloc[entry_price_window // 2]['date']`. For a 3-day window, that is day 2.
5. **Last entry calc date**: `last_entry_calc_date` is the last day in the window (day 3 when N=3). TP/SL monitoring starts the **day after** this date.

So “splitting across three days” is implemented as a **single** entry price (the average of the first three closes), not as three separate order fills. The strategy assumes we can achieve something close to that average by trading over the window; the backtest and target both use this same reference level.

### When TP/SL monitoring starts

In `process_trade()`, the exit logic only considers prices **after** the entry window:

- `monitoring_df = co_prices[(date > last_entry_calc_date) & (date <= mandatory_exit_limit)]`
- So for a 3-day window, TP/SL is checked from day 4 onward until mandatory exit.

### Portfolio valuation during the entry window

In [`src/backtest/simulation.py`](../src/backtest/simulation.py), `generate_quarter_equity_curve()` treats the first `entry_price_window` days as an **entry phase**:

- For each of those days, each position is valued at **entry_price × shares**, not at that day’s market close.
- After the entry phase, positions are marked to market (close × shares) until exit.

So the daily equity curve does not show intra-window price moves; it reflects the assumption that we are “filling” at the average price over the window.

---

## 3. Link to Target Construction

The ML target methodology is described in [MODEL_TRAINING_TARGETS.md](MODEL_TRAINING_TARGETS.md). The important link is:

- **`entry`** in the target pipeline is defined as the **average closing price of the first 3 trading days of the quarter** — the same as backtest when `entry_price_window = 3`.
- Performance metrics are **from that entry level**: e.g. `(high5 − entry)/entry`, `(close5 − entry)/entry`, and their Sharpe versions. Targets are binary labels from **within-quarter percentile rank** of those metrics (e.g. top 20% or 40%).
- So the model is trained to predict: *“From this 3-day-average entry level, which stocks will be in the top percentile by quarter-end performance?”*

That target is **post-result** (all filings in) and **relative** (cross-sectional). For backtest and live to match the training setup, they must use the same entry definition. Using `entry_price_window: 3` ensures that the “price we get” in simulation and production is the same “entry” the model was trained on.

This train–deploy consistency is also discussed in [QUARTERLY_RESULTS_AND_SIGNAL_TIMING.md](QUARTERLY_RESULTS_AND_SIGNAL_TIMING.md) (Section 5: self-consistency).

---

## 4. Rationale: Why Split Entry Across Three Days?

### Quarterly signal and target horizon

The strategy is built on a **quarterly signal**: the ML model consumes quarterly fundamental features and some price momentum; the target is built over the full quarter (from entry to high5/close5). The intended edge is **structural** and **low-frequency**, not a bet on the exact tick on day one.

So the **entry price** should represent a level we can realistically trade at without being **over-influenced by short-term trends or single-day sentiment**. Splitting entry across three days (operationally: using the mean of the first three closes) supports that in several ways.

### Reducing short-term noise

- **Single-day entry**: One close can be a result-day spike, a sentiment dip, or a liquidity blip. The outcome would be sensitive to which single day we chose.
- **Three-day average**: Smooths single-day outliers and makes the “entry” less sensitive to the exact day within the first few days after the deadline. The realized entry is closer to “average post-deadline level” than to “day-1 sentiment.”

### Alignment with the signal type

The model is not predicting next-day moves; it is predicting relative performance over the quarter. Using a 3-day average makes the **realized entry** conceptually consistent with that: we care about the level at which we are effectively invested for the quarter, not the closing print on one specific day.

### Variance and robustness

Averaging over 3 days **reduces the variance** of the entry price compared to using only the first day’s close. That makes backtest and live results more stable with respect to noisy single-day moves. Sensitivity to window length (e.g. 3 vs 5 vs 10 days) is suggested as future work in [QUARTERLY_RESULTS_AND_SIGNAL_TIMING.md](QUARTERLY_RESULTS_AND_SIGNAL_TIMING.md) (entry-window sensitivity).

### Trade-off

A **longer** window would smooth more but would:

- Start TP/SL monitoring later (e.g. day 6 for a 5-day window).
- Slightly delay “full” deployment in the equity curve.

The default of 3 days matches the target definition and is a reasonable balance. The document does not prescribe a different value; changing the window would require retraining the model with a matching `entry` definition if train–deploy consistency is to be preserved.

---

## 5. Summary and Cross-References

- **Entry price window**: Number of trading days at quarter start over which entry price = mean(close). Default 3.
- **Implementation**: One average price and one logical entry date (middle of window); TP/SL from the day after the window; entry-phase valuation in the equity curve as in [`simulation.py`](../src/backtest/simulation.py).
- **Target link**: Same 3-day “entry” as in [MODEL_TRAINING_TARGETS.md](MODEL_TRAINING_TARGETS.md); backtest/live must use the same convention.
- **Rationale**: Quarterly signal and quarter-horizon target; 3-day average reduces short-term noise, aligns with the target, and fits the low-frequency, post-result design.

**See also:**

- [MODEL_TRAINING_TARGETS.md](MODEL_TRAINING_TARGETS.md) — Target creation and the definition of `entry`.
- [QUARTERLY_RESULTS_AND_SIGNAL_TIMING.md](QUARTERLY_RESULTS_AND_SIGNAL_TIMING.md) — SEBI timing, post-result entry, and entry-window sensitivity as future work.
- [`src/backtest/tpsl.py`](../src/backtest/tpsl.py) — `_resolve_entry()`, `process_trade()`, and monitoring start.
- [`src/backtest/simulation.py`](../src/backtest/simulation.py) — Entry-phase valuation in the equity curve.
- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) — `entry_price_window` in the YAML reference.

*Last updated: March 2026*
