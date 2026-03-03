# Cash Appreciation: Practitioner's Guide

> How to realise the 6.5% annual return on idle cash assumed by the backtester — instruments, operations, and gap analysis.

---

## 1. What the Backtester Assumes

The framework models idle cash (exit proceeds + uninvested capital) as appreciating at a
configurable annual rate, compounded daily over trading days.

| Parameter | Default | Source |
|-----------|---------|--------|
| Annual rate | 6.5% | `RISK_FREE_RATE` in `config/defaults.py` |
| Compounding base | 252 trading days / year | `TRADING_DAYS_PER_YEAR` in `config/defaults.py` |

The daily rate is computed as:

$$r_{daily} = (1 + r_{annual})^{\,1/252} - 1$$

At the default 6.5%:

$$r_{daily} = 1.065^{\,1/252} - 1 \approx 0.02503\%$$

This is applied in the daily equity-curve loop in `backtest/simulation.py`:

```python
for idx in range(len(date_range)):
    if idx > 0:
        running_cash *= r_multiplier   # appreciate yesterday's balance
    running_cash += cash_inflows[idx]   # add today's inflows
    cash_series[idx] = running_cash
```

**Key implication:** the model assumes that any cash received today is deployed into
a risk-free vehicle *that same day* and begins earning from the next trading day onward.

---

## 2. When Does Cash Appear?

There are two sources of idle cash within a quarter:

### 2.1 Uninvested Capital (Day 1)

If a selected stock has no valid entry price (e.g., the stock is suspended during the
entry window), its allocated capital stays as cash from the first day of the quarter.
This is rare but handled automatically.

### 2.2 Exit Proceeds (Stochastic, Spread Across the Quarter)

When a trade hits its take-profit, stop-loss, regime-exit, or time-based exit,
the proceeds (`shares × exit_price`) flow into the cash pool on the `exit_date`.

Exits are **not clustered on a single day**. In a typical quarter with 30 stocks:

- Some will hit TP/SL within the first few weeks.
- Others will hold until the mandatory end-of-quarter cutoff.
- In volatile markets, exits tend to cluster earlier; in calm markets, more stocks
  survive to the cutoff.

This means cash inflows arrive **irregularly throughout the quarter** — a critical
constraint when choosing a real-world parking vehicle.

---

## 3. Model Mechanics in Detail

The daily cash loop works as follows:

1. **Start of quarter:** `running_cash` = uninvested capital (usually zero).
2. **Each trading day *t* (for *t* > 0):** multiply `running_cash` by $(1 + r_{daily})$.
3. **Then:** add any exit proceeds received on day *t*.
4. **End of quarter:** final `running_cash` rolls into the next quarter's `initial_capital`
   (along with the mark-to-market value of any remaining positions).

**The model implicitly assumes:**

- Same-day deployment of exit proceeds into a risk-free vehicle.
- No settlement lag (equity settlement in India is T+1 as of 2024).
- No transaction costs or expense ratios on the cash vehicle.
- Returns accrue only on trading days (252/year), not calendar days (365/year).

---

## 4. Investment Vehicles for Idle Cash

The table below compares instruments available to for parking short-duration idle cash. 
Yields are indicative for the 2024–2026 rate environment (RBI repo rate ~6.25–6.50%).

| Vehicle | Typical Yield | Liquidity | Min Ticket | Settlement | Best For |
|---------|--------------|-----------|------------|------------|----------|
| **Overnight Mutual Funds** | 6.0–6.8% | T+1 redemption | ₹500 | T+1 | Retail & small AUM; daily accrual, no lock-in |
| **Liquid Mutual Funds** | 6.2–7.0% | T+1 (instant for ≤₹2L) | ₹500 | T+1 | Slightly higher yield, 7-day exit load on some schemes |
| **TREPS (Tri-Party Repo)** | ~6.5% (tracks repo) | Overnight | ₹5 Cr+ | T+0 | **Institutional portfolios — best match for model** |
| **91-day Treasury Bills** | 6.3–6.8% | Secondary market | ₹25,000 face | T+1 (auction) | Predictable lump sums; locks capital ~3 months |
| **182/364-day T-Bills** | 6.5–7.0% | Less liquid secondary | ₹25,000 face | T+1 | Only for known long-idle balances; tenure > 1 quarter |
| **Bank Fixed Deposits** | 6.0–7.5% | Penalty on early withdrawal | Varies | Immediate | Unsuitable — penalty defeats the purpose for uncertain timing |
| **Money Market / CBLO** | 6.0–6.5% | T+0 / T+1 | Varies | T+0/T+1 | Reasonable alternative; slightly lower yield |

### Which vehicle matches the model best?

1. **TREPS** (for institutional portfolios ≥ ₹5 Cr tickets): overnight tenor exactly
   mirrors the daily compounding loop; proceeds are available the next morning.
2. **Overnight / Liquid Mutual Funds** (for smaller portfolios): daily NAV accrual
   provides near-identical behaviour with T+1 liquidity.

T-bills, while offering competitive yields, introduce a **tenor mismatch**: you
cannot know in advance *when* each stock will exit, so locking proceeds into a
91-day bill risks either early liquidation (at an uncertain secondary-market price)
or holding the bill past the quarter boundary.

---

## 5. Managing Irregular Inflows

The core operational challenge is that exit proceeds arrive on unpredictable dates
throughout the quarter. Here are practical approaches:

### 5.1 Sweep Account

Set up an automatic sweep arrangement with an overnight or liquid mutual fund:

- All equity sale proceeds, once settled (T+1), are automatically swept into the fund.
- Cash is redeemed at the start of the next quarter for reinvestment.

Most AMCs and banks offer this as a standard treasury product for institutional clients.

### 5.2 Laddered T-Bill Strategy

For the *predictable* portion of idle cash:

- Analyse historical exit patterns: e.g., "on average, 30% of stocks exit within the
  first 3 weeks" → buy 91-day T-bills at the next weekly auction with those proceeds.
- Keep the residual in an overnight fund for flexibility.
- This can squeeze out an extra 10–20 bps vs. a pure overnight strategy, at the cost
  of operational complexity and some liquidity risk.

### 5.3 Hybrid Approach

- **First 2 weeks of the quarter:** all proceeds go to an overnight fund (exits are
  sparse, amounts small).
- **Weeks 3–8:** once a meaningful cash balance builds, deploy the accumulated amount
  into a 91-day T-bill auction while continuing to sweep new inflows into the overnight
  fund.
- **Last 2 weeks:** stop new T-bill purchases; let existing positions mature or sell in
  secondary market. Fresh proceeds stay in overnight fund.

---

## 6. Gap Analysis: Model vs. Reality

### 6.1 Sources of Overstatement (Model > Reality)

| Factor | Impact | Magnitude |
|--------|--------|-----------|
| **Expense ratios** | Overnight/liquid funds charge 0.10–0.25% TER | Net yield reduced to ~6.25–6.40% |
| **Settlement lag** | Equity proceeds settle T+1; model credits on exit_date | ~1 day of lost interest per exit ≈ negligible individually, ~2–3 bps over a quarter |
| **Tax drag** | Debt fund gains (< 3 years holding) taxed at slab rate | Depends on investor profile; can reduce effective yield by 50–150 bps post-tax |
| **Idle-day gap** | Proceeds may sit in settlement account over exchange holidays | 1–2 lost days around long weekends; ~1–2 bps per occurrence |

### 6.2 Sources of Understatement (Model < Reality)

| Factor | Impact | Magnitude |
|--------|--------|-----------|
| **Calendar-day accrual** | Overnight/liquid funds accrue on all 365 days; model uses 252 trading days | Model understates by ~2–3 bps annually |
| **Higher-yielding instruments** | A disciplined T-bill ladder can exceed 6.5% in certain rate environments | +10–30 bps |
| **Compounding on holidays** | Weekends and holidays earn returns in real funds but not in the model | Small positive gap |

---

## 7. Recommendations

1. **Default to an overnight or liquid fund sweep** for simplicity and close alignment
   with the daily-compounding model.

2. **TREPS is optimal for institutional-size portfolios** (₹5 Cr+ tickets) — overnight
   tenor exactly matches the model's assumption.

3. **Do not use fixed deposits or long-duration instruments** for cash that may need
   to be redeployed at the start of the next quarter.

4. **Monitor the rate environment.** The 6.5% default is benchmarked to the RBI repo
   rate era of 2023–2026. If the repo rate shifts materially, update the default in
   `config/defaults.py` or override per-run in the YAML.

---

## 8. Configuration Reference

The `cash_appreciation_rate` parameter lives in `strategy_config.yaml`:

```yaml
# Annual risk-free rate applied to idle cash.
# - null (default): Uses RISK_FREE_RATE from config/defaults.py (currently 6.5%)
# - 0: Disable cash appreciation (cash stays flat)
# - Any positive float: Override rate (e.g., 0.07 for 7%)
cash_appreciation_rate: null
```

The resolution chain:

1. YAML `cash_appreciation_rate` → if `null` →
2. `RISK_FREE_RATE` in `config/defaults.py` (0.065) →
3. Passed to `compute_portfolio_value_over_quarters()` as `risk_free_rate_annual` →
4. Converted to `r_daily` and applied in the daily cash loop.

See also:
- [`backtest/simulation.py`](../src/backtest/simulation.py) — cash compounding implementation
- [`config/defaults.py`](../src/config/defaults.py) — global constants
- [`strategy_config_template.yaml`](../src/strategy_config_template.yaml) — full parameter reference
