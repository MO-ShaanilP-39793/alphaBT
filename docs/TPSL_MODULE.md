# TP/SL Trade Simulation Module

> Documentation for `backtest/tpsl.py` — the core trade execution engine that simulates entry, threshold calculation, intraday monitoring, and exit for each stock position.

---

## Table of Contents

1. [Overview](#overview)
2. [Quarter Date Mappings](#quarter-date-mappings)
3. [Trade Lifecycle](#trade-lifecycle)
4. [TP/SL Modes](#tpsl-modes)
5. [Index-Guided Exits](#index-guided-exits)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Edge Cases](#edge-cases)

---

## Overview

`backtest/tpsl.py` handles the full lifecycle of simulated trades:

1. **Entry** — Average close price over the first N trading days (`entry_price_window`, default 3)
2. **Threshold calculation** — TP and SL prices computed per one of four modes
3. **Daily monitoring** — Check intraday high/low against thresholds + index regime signals
4. **Exit** — TP hit, SL hit, regime exit, or mandatory time-based exit at quarter end

TP and SL modes can be configured **independently**, enabling mixed strategies (e.g., ATR for take profit with pivot for stop loss).

### Supported Modes

| Mode | Description | Requires |
|------|-------------|----------|
| `tiered` | Static percentage per category | `tiered_config` |
| `flat` | Single percentage for all stocks | `flat_config` |
| `atr` | Dynamic thresholds based on Average True Range | `atr_config` (falls back to `flat` on failure) |
| `pivot` | Dynamic thresholds based on pivot point S/R levels | `pivot_config` (falls back to `flat` on failure) |

---

## Quarter Date Mappings

The module uses `YYYYMM` quarter codes where MM ∈ {02, 05, 08, 11}:

| Quarter Code | Entry Search Start | Mandatory Exit Date |
|--------------|-------------------|---------------------|
| `YYYY02` | Feb 15, YYYY | May 30, YYYY |
| `YYYY05` | May 31, YYYY | Aug 14, YYYY |
| `YYYY08` | Aug 15, YYYY | Nov 14, YYYY |
| `YYYY11` | Nov 15, YYYY | Feb 14, YYYY+1 |

Entry price is calculated by averaging close prices over the first N trading days from the entry search start date. Trading (monitoring) begins after those N days.

---

## Trade Lifecycle

### 1. Entry Price Calculation

```
entry_price = mean(close prices for first N trading days from entry_search_start)
```

Where N = `entry_price_window` (default: 3). Only days with actual trading data are counted.

### 2. Threshold Calculation

TP and SL thresholds are computed independently via `calculate_dynamic_thresholds()`:

- **Tiered mode**: `tp_price = entry_price × (1 + tp_pct)`, `sl_price = entry_price × (1 - sl_pct)`
- **Flat mode**: Same formula, but a single percentage for all stocks
- **ATR mode**: `tp_price = entry + (multiplier × ATR)`, `sl_price = entry - (multiplier × ATR)`
- **Pivot mode**: TP = resistance level (R1, R1_R2, R2, R2_R3, R3), SL = support level (S1, S1_S2, S2, S2_S3, S3)

If ATR or pivot calculation fails (insufficient data), the module automatically falls back to **flat** mode.

Optional **volatility adjustment**: If enabled, the threshold distances are scaled by a market-volatility multiplier.

### 3. Daily Monitoring

For each trading day after entry:

1. **Regime exit check** — If index regime filter is enabled and the index is below its MA by more than the exit threshold, exit at close
2. **Stop Loss check** — If `low ≤ sl_price`, exit at `sl_price`
3. **Take Profit check** — If `high ≥ tp_price`, exit at `tp_price`
4. **Continue** — If none triggered, move to next day

> SL is checked before TP on each day (priority order).

### 4. Exit

| Exit Type | Price Used | Condition |
|-----------|-----------|-----------|
| Stop Loss | `sl_price` | Day's low ≤ SL threshold |
| Take Profit | `tp_price` | Day's high ≥ TP threshold |
| Regime Exit | Day's close | Index regime filter triggers |
| Time Exit | Last available close | Mandatory exit date reached |

---

## TP/SL Modes

### Tiered Mode

Static percentages looked up by category. The category column (`cat`) maps to per-category configs under `tiered_config`.

```yaml
tp_mode: 'tiered'
sl_mode: 'tiered'
category_scheme: 'volatility'   # or 'mcap'
tiered_config:
  tp_pct:
    low_volatility: 0.08
    medium_volatility: 0.10
    high_volatility: 0.12
  sl_pct:
    low_volatility: 0.05
    medium_volatility: 0.07
    high_volatility: 0.10
  default_tp_pct: 0.10
  default_sl_pct: 0.05
```

### Flat Mode

A single TP/SL percentage for all stocks, ignoring categories:

```yaml
tp_mode: 'flat'
sl_mode: 'flat'
flat_config:
  tp_pct: 0.08
  sl_pct: 0.05
```

### ATR Mode

Dynamic thresholds based on each stock's recent price volatility. See [DYNAMIC_LEVELS_REFERENCE.md](DYNAMIC_LEVELS_REFERENCE.md) for the full ATR explanation.

```yaml
tp_mode: 'atr'
sl_mode: 'atr'
atr_config:
  period: 14
  tp_multiplier: 2.0
  sl_multiplier: 1.5
```

### Pivot Mode

Thresholds based on support/resistance levels. See [DYNAMIC_LEVELS_REFERENCE.md](DYNAMIC_LEVELS_REFERENCE.md) for the full pivot point explanation.

```yaml
tp_mode: 'pivot'
sl_mode: 'pivot'
pivot_config:
  lookback_days: 60
  tp_level: 'R1'   # R1, R1_R2, R2, R2_R3, or R3
  sl_level: 'S1'   # S1, S1_S2, S2, S2_S3, or S3
```

### Mixed Modes

TP and SL can use different modes:

```yaml
tp_mode: 'atr'
sl_mode: 'pivot'
atr_config:
  period: 14
  tp_multiplier: 2.0
  sl_multiplier: 1.5
pivot_config:
  lookback_days: 60
  sl_level: 'S1'
```

---

## Index-Guided Exits

Optional exit triggers based on market conditions, configured under `index_exit`:

### Regime Filter

Exit positions when the index falls below its moving average:

```yaml
index_exit:
  regime_filter:
    enabled: true
    ma_period: 20
    exit_threshold: -0.02   # Exit if index is 2% below MA
```

### Volatility Adjustment

Scale TP/SL distances based on current market volatility:

```yaml
index_exit:
  vol_adjustment:
    enabled: true
    lookback: 20
    high_vol_threshold: 0.25
    low_vol_threshold: 0.15
    high_vol_multiplier: 1.5
    low_vol_multiplier: 0.8
```

When market volatility is high, thresholds are widened (multiplier > 1.0) to avoid premature exits. When volatility is low, thresholds are tightened.

---

## API Reference

### `get_date_params(quarter_str)`

Parses a `YYYYMM` quarter string to determine the entry search start and mandatory exit dates.

**Returns:** `(entry_search_start, mandatory_exit_date)` — both `pd.Timestamp`.

---

### `calculate_thresholds_tiered(entry_price, category_scheme, cat, tiered_config)`

Calculates tiered percentage-based TP/SL thresholds.

**Returns:** `(tp_price, sl_price, tp_pct, sl_pct)`

---

### `calculate_dynamic_thresholds(price_df, co_name, entry_price, entry_date, category_scheme, cat, tp_mode, sl_mode, index_df=None, *, atr_config, pivot_config, flat_config, index_exit_config, tiered_config)`

Main threshold calculation function. Dispatches to the appropriate mode for each side (TP/SL) independently, applies volatility adjustment if enabled.

**Returns:** `(tp_price, sl_price, metadata_dict)`

The `metadata_dict` contains:
- `tp_mode`, `sl_mode` — modes used
- `tp_pct`, `sl_pct` — effective percentages
- `tp_atr`, `sl_atr` — ATR values (if ATR mode)
- `tp_pivots`, `sl_pivots` — pivot dicts (if pivot mode)
- `tp_fallback`, `sl_fallback` — set to `'flat'` if a dynamic mode fell back
- `vol_adjustment_multiplier` — if volatility adjustment was applied

---

### `process_trade(row, price_df, category_scheme, index_df=None, tp_mode, sl_mode, tp_enabled, sl_enabled, entry_price_window, *, tiered_config, atr_config, pivot_config, flat_config, index_exit_config)`

Processes a single trade (one row from `selected_stocks`) through the full lifecycle: entry calculation → threshold computation → daily monitoring → exit.

**Returns:** `pd.Series` with: `exit_date`, `entry_price`, `exit_price`, `tp_price`, `sl_price`, `exit_type`, `tp_pct`, `sl_pct`, `tp_mode`, `sl_mode`, and mode-specific metadata fields.

---

### `simulate_trades(selected_stocks, price_data, category_scheme, index_data=None, tp_mode=None, sl_mode=None, tp_enabled=None, sl_enabled=None, tiered_config=None, atr_config=None, pivot_config=None, flat_config=None, index_exit_config=None, entry_price_window=3)`

Main entry point — processes all trades in a portfolio. Applies `process_trade()` row-by-row and appends `stock_return` column.

**Input modes:**
- **Selection mode**: `selected_stocks` has `['quarter', 'co_name', 'cat', 'cat_weight']`
- **Preselected mode**: `selected_stocks` has `['quarter', 'co_name', 'stock_weight']` with optional `'cat'`

**Returns:** DataFrame with original columns plus trade results (`entry_price`, `exit_price`, `exit_date`, `exit_type`, `stock_return`, etc.).

---

## Configuration

All parameters are set in `strategy_config.yaml`. See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for the full parameter reference.

Key config keys consumed by this module:

| Config Key | Default | Description |
|-----------|---------|-------------|
| `tp_mode` | `'flat'` | Take profit calculation mode |
| `sl_mode` | `'flat'` | Stop loss calculation mode |
| `tp_enabled` | `true` | Enable/disable take profit |
| `sl_enabled` | `true` | Enable/disable stop loss |
| `entry_price_window` | `3` | Trading days averaged for entry price |
| `tiered_config` | — | Tiered mode: per-category TP/SL percentages + defaults |
| `atr_config` | — | ATR mode parameters |
| `pivot_config` | — | Pivot mode parameters |
| `flat_config` | — | Flat mode parameters |
| `index_exit` | — | Index-guided exit settings |

---

## Edge Cases

1. **Insufficient entry data** — If fewer than N trading days exist after entry start, returns `None` values
2. **No monitoring data** — If no price data between entry and mandatory exit, returns `entry_price` but `None` exit
3. **Missing trading days** — Uses available trading days only (no assumption of continuous dates)
4. **Dynamic mode failure** — ATR/pivot calculation failures fall back to flat-mode thresholds automatically
5. **No category column** — If `cat` column is missing, a `'_default'` placeholder is used and `default_tp_pct`/`default_sl_pct` in `tiered_config` provides thresholds
6. **TP disabled / SL disabled** — When `tp_enabled: false` or `sl_enabled: false`, that side's threshold is not checked during monitoring

---

## Related Documentation

- [DYNAMIC_LEVELS_REFERENCE.md](DYNAMIC_LEVELS_REFERENCE.md) — ATR and pivot point theory + `dynamic_levels.py` API
- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) — Full YAML parameter reference
- [TUNING_GUIDE.md](TUNING_GUIDE.md) — Optuna optimization of TP/SL parameters
- [STRATEGIC_OPTIONS.md](STRATEGIC_OPTIONS.md) — Strategy selection guidance

*Last updated: March 2026*
