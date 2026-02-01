# TP/SL Optimization Module Documentation

## Overview

`optimize_tpsl.py` provides grid search functionality to find optimal Take Profit and Stop Loss parameters by systematically testing combinations and evaluating performance metrics.

---

## Quick Start

```bash
cd src

# Basic optimization (uses defaults)
python optimize_tpsl.py

# Custom parameters
python optimize_tpsl.py \
    --metric sharpe \
    --tp-range 0.02 0.04 0.06 0.08 0.10 \
    --sl-range 0.02 0.04 0.06 0.08 \
    --output-dir ../my_optimization_results
```

---

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `strategy_config.yaml` | Path to configuration file |
| `--metric` | `sharpe` | Metric to optimize (`sharpe`, `sortino`, `total_return`, `calmar`, `win_rate`) |
| `--tp-range` | `0.02 0.04 0.06 0.08 0.10 0.12 0.15` | Space-separated TP percentages |
| `--sl-range` | `0.02 0.04 0.06 0.08 0.10` | Space-separated SL percentages |
| `--output-dir` | `../optimization_results` | Output directory |
| `--no-parallel` | (flag) | Disable parallel processing |

---

## Functions

### `run_tpsl_optimization(tp_range, sl_range, config_path, metric, parallel, max_workers)`

Main entry point for grid search optimization.

**Parameters:**
- `tp_range`: List of TP percentages (e.g., `[0.02, 0.05, 0.08]`)
- `sl_range`: List of SL percentages
- `config_path`: Path to strategy config YAML
- `metric`: Metric to optimize by
- `parallel`: Enable parallel execution (default: `True`)
- `max_workers`: Number of parallel workers (default: CPU count - 1)

**Returns:** DataFrame with results for all combinations

**Example:**
```python
from optimize_tpsl import run_tpsl_optimization

results = run_tpsl_optimization(
    tp_range=[0.02, 0.04, 0.06, 0.08, 0.10],
    sl_range=[0.02, 0.04, 0.06, 0.08],
    metric='sharpe'
)

print(results.head())
```

---

### `find_optimal_params(results_df, metric)`

Extracts the best TP/SL combination from optimization results.

**Returns:** Dictionary with optimal parameters and metrics

```python
from optimize_tpsl import find_optimal_params

optimal = find_optimal_params(results, 'sharpe')
print(f"Optimal TP: {optimal['optimal_tp']*100:.1f}%")
print(f"Optimal SL: {optimal['optimal_sl']*100:.1f}%")
print(f"Sharpe: {optimal['sharpe']:.3f}")
```

---

### `generate_optimization_heatmap(results_df, metric, save_path, figsize)`

Creates a heatmap visualization of the parameter surface.

**Parameters:**
- `results_df`: DataFrame from `run_tpsl_optimization`
- `metric`: Metric to visualize
- `save_path`: File path to save (if `None`, displays interactively)
- `figsize`: Tuple for figure dimensions

```python
from optimize_tpsl import generate_optimization_heatmap

generate_optimization_heatmap(
    results, 
    metric='sharpe',
    save_path='sharpe_heatmap.png'
)
```

---

### `generate_optimization_report(results_df, output_dir)`

Generates a comprehensive report with all visualizations and summaries.

**Outputs:**
- `optimization_results.csv` - Raw data
- `heatmap_sharpe.png` - Sharpe ratio surface
- `heatmap_total_return.png` - Return surface
- `heatmap_sortino.png` - Sortino ratio surface
- `heatmap_win_rate.png` - Win rate surface
- `heatmap_calmar.png` - Calmar ratio surface
- `optimal_params_summary.yaml` - Best parameters per metric

---

## Output Format

### Results DataFrame Columns

| Column | Description |
|--------|-------------|
| `tp` | Take profit percentage tested |
| `sl` | Stop loss percentage tested |
| `total_return` | Total return (%) |
| `cagr` | Compound annual growth rate (%) |
| `sharpe` | Sharpe ratio |
| `sortino` | Sortino ratio |
| `max_drawdown` | Maximum drawdown (%) |
| `calmar` | Calmar ratio |
| `volatility` | Annualized volatility (%) |
| `win_rate` | Percentage of profitable trades |
| `avg_holding_period` | Average days per trade |
| `tp_hit_rate` | Percentage of trades hitting TP |
| `sl_hit_rate` | Percentage of trades hitting SL |
| `total_trades` | Number of trades |
| `error` | Error message if calculation failed |

---

## Optimization Metrics

### Sharpe Ratio (Recommended Default)
```
Sharpe = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
```
Higher is better. Measures risk-adjusted return considering total volatility.

### Sortino Ratio
```
Sortino = (Portfolio Return - Risk Free Rate) / Downside Deviation
```
Like Sharpe but only penalizes downside volatility. Better for asymmetric return distributions.

### Calmar Ratio
```
Calmar = CAGR / |Max Drawdown|
```
Focuses on capital preservation. Higher means better return per unit of drawdown risk.

### Win Rate
```
Win Rate = Profitable Trades / Total Trades
```
Simple success rate. Doesn't account for magnitude of wins/losses.

### Total Return
```
Total Return = (Final Value - Initial Value) / Initial Value
```
Absolute profit. Doesn't consider risk or path.

---

## Performance Tips

### Reduce Search Space
Start with coarse ranges, then refine:
```bash
# First pass: coarse
python optimize_tpsl.py --tp-range 0.02 0.06 0.10 0.14 --sl-range 0.02 0.06 0.10

# Second pass: refine around optimal
python optimize_tpsl.py --tp-range 0.04 0.05 0.06 0.07 0.08 --sl-range 0.04 0.05 0.06
```

### Memory Issues
If running out of memory with parallel execution:
```bash
python optimize_tpsl.py --no-parallel
```

### Faster Iteration
Test on fewer quarters first:
```yaml
# strategy_config.yaml
first_quarter: 202402
last_quarter: 202405  # Just 4 quarters
```

---

## Interpreting Results

### Good Parameter Regions
Look for **stable regions** where nearby values also perform well:

```
TP\SL   2%    4%    6%    8%
2%    0.85  0.92  0.88  0.75
4%    0.90  1.05  1.02  0.82   <- Optimal region
6%    0.88  1.01  0.98  0.80
8%    0.72  0.85  0.82  0.65
```

The 4% TP / 4% SL cell is optimal (1.05 Sharpe), but the surrounding region (0.98-1.02) is also strong. This suggests **robust** parameters.

### Overfitting Warning Signs
- Single isolated optimal cell
- Sharp performance cliffs
- Very narrow optimal ranges

---

## Example Workflow

```python
from optimize_tpsl import (
    run_tpsl_optimization,
    find_optimal_params,
    generate_optimization_report
)

# 1. Run optimization
results = run_tpsl_optimization(
    tp_range=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
    sl_range=[0.02, 0.04, 0.06, 0.08, 0.10],
    metric='sharpe'
)

# 2. Find optimal for different metrics
sharpe_optimal = find_optimal_params(results, 'sharpe')
calmar_optimal = find_optimal_params(results, 'calmar')

print(f"Best for Sharpe: TP={sharpe_optimal['optimal_tp']:.0%}, SL={sharpe_optimal['optimal_sl']:.0%}")
print(f"Best for Calmar: TP={calmar_optimal['optimal_tp']:.0%}, SL={calmar_optimal['optimal_sl']:.0%}")

# 3. Generate full report
generate_optimization_report(results, './optimization_output')

# 4. Update config with optimal values
# Edit strategy_config.yaml:
# TP_CONFIG:
#   volatility:
#     high_volatility: 0.06  # From optimization
#     ...
```
