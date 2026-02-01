# Portfolio Backtest Analysis Functions

This module (`addtl_bt_fns.py`) provides comprehensive tools for analyzing portfolio backtest results. It's organized into four main sections:

1. **Daily Portfolio Value Analysis** - Metrics computed from daily portfolio values
2. **Benchmark Comparison Analysis** - Compare portfolio performance vs an index
3. **Plotting Functions** - Visualizations for portfolio analysis
4. **Trade-Level Analysis** - Analyze individual trades and quarterly performance

---

## 📊 Daily Portfolio Value Analysis

These functions work with a DataFrame containing columns: `['date', 'portfolio_value', 'quarter']`

### `compute_portfolio_metrics(daily_pf, risk_free_rate=0.065)`

Computes comprehensive portfolio performance metrics.

**Returns a dictionary with:**
| Metric | Description |
|--------|-------------|
| `total_return_pct` | Total return over the period |
| `cagr_pct` | Compound Annual Growth Rate |
| `volatility_pct` | Annualized volatility (standard deviation) |
| `sharpe_ratio` | Risk-adjusted return (excess return / volatility) |
| `sortino_ratio` | Like Sharpe, but only penalizes downside volatility |
| `max_drawdown_pct` | Largest peak-to-trough decline |
| `calmar_ratio` | CAGR / Max Drawdown |
| `var_95_pct` | Value at Risk (5th percentile of daily returns) |
| `best_day_pct` / `worst_day_pct` | Best and worst single-day returns |
| `positive_days_pct` | Percentage of days with positive returns |

---

### `compute_drawdown_series(daily_pf)`

Calculates the drawdown at each point in time.

**Returns DataFrame with:**
- `date` - Date
- `portfolio_value` - Portfolio value
- `cummax` - Running maximum (peak)
- `drawdown` - Absolute drawdown (value - peak)
- `drawdown_pct` - Percentage drawdown

---

### `compute_rolling_returns(daily_pf, windows=[21, 63, 126, 252])`

Computes rolling returns for various time windows.

**Default windows:**
- 21 days (~1 month)
- 63 days (~3 months)
- 126 days (~6 months)
- 252 days (~1 year)

**Returns DataFrame with:** `date`, `portfolio_value`, `return_1M`, `return_3M`, `return_6M`, `return_1Y`

---

### `compute_monthly_returns(daily_pf)`

Creates a monthly returns table (like you'd see in a fund factsheet).

**Returns:** Pivoted DataFrame with years as rows, months as columns, plus an `Annual` column.

---

## 📈 Benchmark Comparison Analysis

These functions work with a comparison DataFrame containing:
`['date', 'pf_value', 'pf_return', 'index_fund_value', 'index_return', 'alpha']`

### `compute_benchmark_metrics(comparison_df, risk_free_rate=0.065)`

Computes metrics comparing portfolio vs benchmark.

**Returns a dictionary with:**
| Metric | Description |
|--------|-------------|
| `portfolio_return_pct` | Total portfolio return |
| `index_return_pct` | Total benchmark return |
| `alpha_pct` | Outperformance vs benchmark |
| `beta` | Sensitivity to benchmark movements |
| `tracking_error_pct` | Volatility of excess returns |
| `information_ratio` | Risk-adjusted alpha |
| `up_capture_pct` | Performance in up markets (>100% = outperforms) |
| `down_capture_pct` | Performance in down markets (<100% = outperforms) |
| `correlation` | Correlation with benchmark |
| `outperformance_days_pct` | % of days beating the benchmark |

---

### `compute_rolling_alpha(comparison_df, window=63)`

Calculates rolling alpha (outperformance) vs the benchmark.

**Returns DataFrame with:** `date`, `rolling_alpha`

---

## 📉 Plotting Functions

All plotting functions accept an optional `save_path` parameter. If provided, the plot is saved to that path; otherwise, it displays interactively.

| Function | Description |
|----------|-------------|
| `plot_drawdown(daily_pf)` | Two-panel chart: portfolio value with peak overlay + drawdown chart |
| `plot_monthly_returns_heatmap(daily_pf)` | Heatmap of monthly returns (red=negative, green=positive) |
| `plot_return_distribution(daily_pf)` | Histogram of daily returns with VaR lines and statistics |
| `plot_rolling_volatility(daily_pf, window=21)` | Rolling annualized volatility over time |
| `plot_rolling_sharpe(daily_pf, window=252)` | Rolling Sharpe ratio over time |

---

### `generate_full_analysis_report(daily_pf, comparison_df=None, output_dir=None)`

**All-in-one function** that:
1. Computes all portfolio metrics
2. Generates monthly returns table
3. Computes rolling returns and drawdown series
4. If `comparison_df` provided: computes benchmark metrics and rolling alpha
5. If `output_dir` provided: saves all plots to that directory

**Returns:** Dictionary containing all computed metrics and DataFrames

---

## 🔄 Trade-Level Analysis

These functions analyze individual trades from a `trade_results.csv` file with columns:
`['quarter', 'co_name', 'cat', 'cat_weight', 'holding_period', 'SL_triggered', 'TP_triggered', 'stock_return']`

### Stock Counts

| Function | Description |
|----------|-------------|
| `get_stock_counts_by_category(trade_results)` | Count of stocks by category per quarter (long format) |
| `get_stock_counts_pivot(trade_results)` | Same as above, but pivoted (quarters as rows, categories as columns) |

---

### Category Returns

| Function | Description |
|----------|-------------|
| `get_category_returns_by_quarter(trade_results)` | Average return for each category per quarter |
| `get_category_returns_pivot(trade_results)` | Same as above, pivoted format |

---

### Holding Periods

### `get_holding_period_by_quarter(trade_results)`

Returns average holding period per quarter, overall and by category.

---

### `get_comprehensive_quarter_analysis(trade_results)`

**Comprehensive quarterly breakdown** including:
- Total stocks traded (`total_stocks`)
- Stock counts by category (`{cat}_count`)
- Average holding period overall (`avg_holding_period`) and by category (`{cat}_avg_hp`)
- Returns by category (`{cat}_return`)
- Portfolio return weighted by category weights (`portfolio_return`)
- TP/SL statistics (if columns available):

| Column | Description |
|--------|-------------|
| `TP_count` | Take Profit triggers |
| `SL_count` | Stop Loss triggers |
| `time_exit_count` | Exits due to holding period expiry |
| `avg_holding_period_SL` | Avg days for SL trades |
| `avg_holding_period_TP` | Avg days for TP trades |

---

## 🚀 Quick Start Examples

```python
import pandas as pd
from addtl_bt_fns import *

# Load your data
daily_pf = pd.read_csv('daily_portfolio_values.csv')
comparison = pd.read_csv('portfolio_vs_index.csv')
trades = pd.read_csv('trade_results.csv')

# Get all portfolio metrics
metrics = compute_portfolio_metrics(daily_pf)
print(f"CAGR: {metrics['cagr_pct']}%")
print(f"Sharpe: {metrics['sharpe_ratio']}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']}%")

# Generate monthly returns table
monthly = compute_monthly_returns(daily_pf)
print(monthly)

# Compare vs benchmark
benchmark_metrics = compute_benchmark_metrics(comparison)
print(f"Alpha: {benchmark_metrics['alpha_pct']}%")

# Analyze trades by quarter
quarter_stats = get_comprehensive_quarter_analysis(trades)
print(quarter_stats)

# Generate full report with all plots
report = generate_full_analysis_report(
    daily_pf, 
    comparison_df=comparison, 
    output_dir='./analysis_output'
)
```

---

## 📝 Notes

- **Risk-free rate**: Default is 6.5% (appropriate for India). Adjust via `risk_free_rate` parameter.
- **Trading days**: Assumes 252 trading days per year for annualization.
- **Categories (`cat`)**: Can be market cap (largecap/midcap/smallcap) or volatility-based categories.
- **`cat_weight`**: Weight assigned to each category in portfolio construction.
