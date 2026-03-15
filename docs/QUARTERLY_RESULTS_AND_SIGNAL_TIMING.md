# Quarterly Results & Signal Timing

> A discussion of the relationship between SEBI result-posting deadlines, the system's quarter-entry timing, and whether ML signals derived from quarterly fundamentals are already priced in by the time the portfolio enters.

---

## 1. Context: SEBI Result-Posting Deadlines

All listed companies on Indian exchanges must file their quarterly financial results within a regulatory window:

| Requirement | Deadline |
|-------------|----------|
| Q1, Q2, Q3 results | **45 days** from quarter-end |
| Q4 (annual) results | **60 days** from quarter-end |

Mapping this to calendar dates:

| Financial Quarter | Period | Quarter-End | SEBI Deadline | System Entry Start | System Exit |
|-------------------|--------|-------------|---------------|--------------------|-------------|
| Q1 | Apr – Jun | Jun 30 | **Aug 14** | **Aug 15** | Nov 14 |
| Q2 | Jul – Sep | Sep 30 | **Nov 14** | **Nov 15** | Feb 14 |
| Q3 | Oct – Dec | Dec 31 | **Feb 14** | **Feb 15** | May 30 |
| Q4 | Jan – Mar | Mar 31 | **May 30** | **May 31** | Aug 14 |

### Deliberate alignment with the system

The alphaBT quarter boundaries (defined in [`utils/quarter.py`](../src/utils/quarter.py)) are **not arbitrary**. Each entry-start date falls exactly one day after the corresponding SEBI filing deadline:

- `YYYY02` → Feb 15 (day after Q3 deadline Feb 14)
- `YYYY05` → May 31 (day after Q4 deadline May 30)
- `YYYY08` → Aug 15 (day after Q1 deadline Aug 14)
- `YYYY11` → Nov 15 (day after Q2 deadline Nov 14)

This guarantees that **all quarterly results are publicly available** when the ML model runs inference and the portfolio enters positions. Correspondingly, each mandatory exit date coincides with the *next* SEBI deadline — meaning the holding period spans exactly one result-filing cycle.

---

## 2. The Concern: Is the Signal Already Priced In?

The current workflow is:

1. Wait until the SEBI deadline (all results available).
2. Run ML inference on the complete dataset.
3. Enter positions starting the next trading day (over `entry_price_window` days, default 3).

The concern is straightforward: **in practice, results are rolling, not simultaneous.** Most companies file well before the 45/60-day deadline. By the time every company has reported and we run inference, the earliest reporters' results are weeks old. The information flow looks like this:

```
Company files results → Market reacts (often within hours) → Price adjusts →
... weeks pass ...
→ SEBI deadline → We run inference → We enter positions
```

For early reporters, the market has had ample time to digest the quarterly numbers. Under a semi-strong form efficient market hypothesis (EMH), all publicly available information — including quarterly results — should be fully reflected in prices by the time we trade. If this holds, the ML model's features (derived from quarterly results) carry no residual predictive power at entry.

This is the strongest version of the objection. The rest of this document examines whether it holds in practice and how the system's design interacts with it.

---

## 3. Target Creation Perspective

The ML target methodology is documented in [MODEL_TRAINING_TARGETS.md](MODEL_TRAINING_TARGETS.md). The key definitions:

- **`entry`** = average close of the first 3 trading days of the quarter (matching `entry_price_window`).
- **Target label** = binary (0/1) based on whether a stock's `(high5 − entry) / entry` or `(close5 − entry) / entry` falls above a percentile threshold (top 20% or top 40%) **within that quarter**.

### The target is inherently post-information

Since `entry` is computed from prices *after* all quarterly results are public, any immediate market reaction to results is already embedded in the entry price. The model is **not** being asked:

> "Will this company's quarterly results be good?"

It is being asked:

> "Given this post-result price (which already reflects the market's reaction to results), will this stock still outperform its peers over the next ~3 months?"

This is a critical distinction. The model learns a cross-sectional ranking: **which stocks, after their results are known and partially digested, will continue to outperform?** This is a legitimate prediction task — it is analogous to asking whether quarterly fundamentals carry *momentum* or *residual information* that the market has not fully arbitraged within the filing window.

### What does this imply for the binary label?

- A label of `1` (top performer) means the stock outperformed peers **from the post-result entry price onward**. The result-day price jump (if any) is not part of the target.
- The model's features encode information derived from quarterly results, but the prediction horizon starts *after* those results are public.
- Predictive power therefore depends on whether quarterly fundamentals carry **residual forward-looking signal** beyond the immediate market reaction.

### Counter-risk

If the market is highly efficient at pricing quarterly results for a given stock, then the entry price will already reflect the information in the features. The model would be left trying to predict noise. This risk is real — but is an empirical question, not a theoretical certainty (see Section 5).

---

## 4. Real-Life Strategy Implementation

### Current workflow

The production flow (via [`get_portfolio.py`](../src/get_portfolio.py)) is:

1. All quarterly results are filed by the SEBI deadline.
2. Feature engineering pipeline runs on the complete dataset.
3. ML model produces probability scores for each stock.
4. `get_portfolio.py --quarter YYYYMM` selects and weights the portfolio.
5. Positions are entered starting the next trading day.

### The timing gap in practice

Inference requires a **complete dataset** — every company's quarterly results must be available for feature computation. In practice:

- ~70–80% of companies file in the first 30 days.
- A long tail files close to (or occasionally past) the deadline.
- If even one company in the universe is missing, the dataset is incomplete.

This creates a practical tension: waiting for completeness means entering later, but entering early means working with partial data.

### The early-entry alternative

Instead of waiting for the deadline, one could run inference as results trickle in and enter positions progressively. This would:

**Potential advantages:**
- Capture more of the immediate post-result price movement for each stock.
- Reduce the "stale signal" concern — entering shortly after a company's results are announced.
- Potentially improve returns by acting on fresher information.

**Practical challenges:**
- The ML model was trained on a target which assumes a shared window over which to measure relative underperformance/outperformance of stocks. The objective of the model would require complete overhaul. 
- The portfolio would need continuous rebalancing as stocks keep posting results.
- Backtesting this strategy requires fundamentally different architecture — the current `simulate_trades()` framework assumes a single entry window per quarter.
- Overfitting risk: optimizing entry timing adds another degree of freedom to an already complex system.

### Where this strategy sits in the alpha spectrum

This is a **low-frequency, quarterly rebalancing strategy**. It is not competing with:

- **Event-driven traders** who react within hours of result announcements.
- **HFT/algorithmic strategies** that trade on order flow or microstructure.
- **News-driven quant funds** that parse filings in real time.

The alpha hypothesis here is **structural**, not informational:

> *Quarterly fundamentals, when considered jointly across the full universe of stocks, predict which stocks will outperform over the next quarter — even after the market has had time to react to individual results.*

This is a bet on the market being **imperfectly efficient at the cross-sectional level** — that while individual result reactions may be rapid, the relative ranking of future performance across hundreds of stocks is not fully determined by the market's immediate reaction.

---

## 5. How the Existing Setup Addresses This

### Self-consistency

The backtesting framework, target creation, and live inference all share the same timing convention:

| Component | Entry timing |
|-----------|-------------|
| **Target creation** | `entry` = avg close of first 3 trading days after deadline |
| **Backtest simulation** | `entry_price_window` = first 3 trading days after quarter start |
| **Live inference** | `get_portfolio.py` → enter next trading day after deadline |

There is **no look-ahead bias** and **no train-deploy mismatch**. The model is trained on the same timing it will encounter in production. Whatever the price at the entry date reflects (including digested results), the model has seen this exact pattern during training.

### Relative ranking, not absolute prediction

The model predicts which stocks will be in the **top percentile of performers** — a cross-sectional ranking task. Even if every individual stock's result is fully priced in, the model may identify multi-variate patterns across companies that the market hasn't fully arbitraged. Example:

- A combination of strong fundamentals + low volatility + being in an under-followed category may predict outperformance — even if each factor alone is priced in.

### Coverage gaps in Indian markets

Indian equity markets exhibit significant **analyst coverage disparity**:

- Large-cap stocks (Nifty 50/100) are heavily covered — result reactions are rapid and potentially efficient.
- Mid-cap and small-cap stocks have thinner coverage. Quarterly results for these companies may be under-analyzed, leaving residual signal even weeks after filing.

---

## 6. Open Questions & Future Research

The following are concrete, testable questions that could resolve the theoretical concern empirically:

- **Rolling entry strategy**: Would entering each stock shortly after its individual result announcement (rather than waiting for the full-universe deadline) improve returns? This requires a different backtest architecture but could be prototyped for a subset of quarters.

- **Large-cap vs. small-cap alpha**: Is there a measurable difference in the model's predictive power for heavily-covered large-caps vs. under-followed small/mid-caps? If alpha is concentrated in smaller stocks, it supports the "coverage gap" hypothesis.

- **Result date as a feature**: Would adding the result announcement date (or days-until-deadline) as a model feature improve predictions? This directly tests whether timing of filing carries information.

- **Information decay quantification**: Compare stock price on result announcement date vs. the eventual entry date. If the two are highly correlated (prices don't move much between announcement and entry), then "priced in" is less of a concern — the market didn't react strongly to begin with, and the signal is in the fundamentals, not the reaction.

- **Entry-window sensitivity**: Does the strategy's alpha change materially if `entry_price_window` is extended from 3 to, say, 5 or 10 trading days? A rapid decay would suggest time-sensitive signal; stability would suggest the signal is structural.

---

*This document discusses a design consideration in the alphaBT framework. The question of whether quarterly result signals are "priced in" is ultimately empirical — the discussion above frames the arguments on both sides and identifies concrete experiments to test them.*

*Last updated: March 2026*
