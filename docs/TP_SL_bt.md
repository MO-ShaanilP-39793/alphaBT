# TP/SL Backtesting Script Documentation

## Overview

`TP_SL_bt.py` is a backtesting script that simulates quarterly trading strategies with configurable Take Profit (TP) and Stop Loss (SL) thresholds. The script processes selected stocks, calculates entry prices based on a 3-day average, and monitors positions until they hit TP/SL levels or reach a mandatory quarterly exit date.

## Configuration

The script reads configuration from `TP_SL_config.yaml` which defines:

- **TP_CONFIG**: Take Profit percentages by volatility category
- **SL_CONFIG**: Stop Loss percentages by volatility category

Volatility categories typically include: `Low`, `Medium`, `High`

## Quarter Date Mappings

The script operates on quarterly periods with specific entry and exit windows:

| Quarter Code | Entry Search Start | Mandatory Exit Date |
|--------------|-------------------|---------------------|
| YYYY02       | Feb 15, YYYY      | May 30, YYYY        |
| YYYY05       | May 31, YYYY      | Aug 14, YYYY        |
| YYYY08       | Aug 15, YYYY      | Nov 14, YYYY        |
| YYYY11       | Nov 15, YYYY      | Feb 14, YYYY+1      |

## Core Functions

### `get_date_params(quarter_str)`

Parses a quarter string (format: `YYYYMM`) to determine trading window boundaries.

**Parameters:**
- `quarter_str` (str): Quarter identifier in YYYYMM format (e.g., '202302')

**Returns:**
- `entry_search_start` (pd.Timestamp): Date when entry calculation begins
- `mandatory_exit_date` (pd.Timestamp): Latest possible exit date for the quarter

---

### `process_trade(row, price_df)`

Processes a single trade from entry calculation through exit determination.

**Parameters:**
- `row` (pd.Series): Single row from selected_stocks DataFrame containing:
  - `co_name`: Company name
  - `quarter`: Quarter code (YYYYMM)
  - `vol_cat`: Volatility category (Low/Medium/High)
  - `cat_weight`: Category weight (not used in calculation)
- `price_df` (pd.DataFrame): Price data containing columns: `date`, `co_name`, `open`, `high`, `low`, `close`

**Returns:**
- pd.Series with three values:
  - `exit_date`: Date of trade exit
  - `entry_price`: Average close price over first 3 trading days
  - `exit_price`: Price at exit (TP/SL price or closing price)

**Logic Flow:**

1. **Entry Calculation**
   - Identifies first 3 trading days starting from `entry_search_start`
   - Entry price = mean of close prices for these 3 days
   - Trading begins after day 3

2. **Threshold Calculation**
   - Take Profit Price = `entry_price × (1 + tp_pct)`
   - Stop Loss Price = `entry_price × (1 - sl_pct)`

3. **Exit Determination** (Priority Order)
   - **Stop Loss**: Triggered if `low` ≤ `sl_price` on any day → Exit at `sl_price`
   - **Take Profit**: Triggered if `high` ≥ `tp_price` on any day → Exit at `tp_price`
   - **Time Exit**: If no TP/SL hit by mandatory exit date → Exit at closing price on last available day

---

### `calculate_portfolio_performance(selected_stocks, price_data)`

Main execution wrapper that processes all trades in a portfolio.

**Parameters:**
- `selected_stocks` (pd.DataFrame): Portfolio selections with columns:
  - `quarter`: Quarter code (YYYYMM)
  - `co_name`: Company name
  - `vol_cat`: Volatility category
  - `cat_weight`: Category weight
- `price_data` (pd.DataFrame): Historical price data with columns:
  - `date`: Trading date
  - `co_name`: Company name
  - `open`: Opening price
  - `high`: High price
  - `low`: Low price
  - `close`: Closing price

**Returns:**
- pd.DataFrame: Original selected_stocks DataFrame with three additional columns:
  - `exit_date`: Exit date for each trade
  - `entry_price`: Calculated entry price
  - `exit_price`: Exit price

<!-- **Validation:**
- Checks for required columns in both input DataFrames
- Ensures `date` column is datetime type
- Sorts data by quarter and company name -->

<!-- ## Usage Example

```python
import pandas as pd
from TP_SL_bt import calculate_portfolio_performance

# Load your data
selected_stocks = pd.DataFrame({
    'quarter': ['202302', '202305'],
    'co_name': ['CompanyA', 'CompanyB'],
    'vol_cat': ['Low', 'High'],
    'cat_weight': [0.6, 0.4]
})

price_data = pd.read_csv('price_data.csv')
# Ensure price_data has: date, co_name, open, high, low, close

# Run backtest
results = calculate_portfolio_performance(selected_stocks, price_data)

print(results[['co_name', 'quarter', 'entry_price', 'exit_price', 'exit_date']])
``` -->

## Edge Cases Handled

1. **Insufficient Data**: If fewer than 3 trading days available after entry start → Returns `None` values
2. **No Monitoring Data**: If no price data exists between entry calculation and mandatory exit → Returns entry_price but `None` exit_price
3. **Missing Price Days**: Uses available trading days only (no assumption of continuous dates)


## Notes

- All dates are inclusive in boundary checks
- Stop Loss is checked before Take Profit on each day (priority order)
- Entry price calculation uses closing prices only
- Exit execution uses intraday prices (high/low) for TP/SL detection

---

## Advanced Features (New)

The TP/SL module now supports advanced threshold calculation modes and index-guided exits. For comprehensive documentation, see **[TP/SL Optimization Guide](tpsl_optimization_guide.md)**.

### TP/SL Calculation Modes

Configure via `tpsl_mode` in `strategy_config.yaml`:

| Mode | Description |
|------|-------------|
| `fixed` | Static percentage per category (default, original behavior) |
| `atr` | Dynamic thresholds based on Average True Range |
| `pivot` | Dynamic thresholds based on support/resistance pivot points |

### Index-Guided Exits

New exit triggers based on market conditions:

- **Regime Filter**: Exit positions when index falls below moving average
- **Volatility Adjustment**: Widen/tighten thresholds based on market volatility

### Parameter Optimization

Use `optimize_tpsl.py` to find optimal TP/SL values:

```bash
python optimize_tpsl.py --metric sharpe --tp-range 0.02 0.05 0.08 0.10
```

See the [TP/SL Optimization Guide](tpsl_optimization_guide.md) for detailed usage.