# Benchmark Metrics Documentation

This document provides detailed documentation for the `compute_benchmark_metrics` function in `addtl_bt_fns.py`.

## Overview

The `compute_benchmark_metrics` function computes **benchmark comparison metrics**—statistics that measure how a portfolio performs relative to a market index (benchmark). These are standard metrics used in institutional fund analysis.

---

## Function Signature

```python
def compute_benchmark_metrics(comparison_df: pd.DataFrame, risk_free_rate: float = 0.065) -> dict:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comparison_df` | `pd.DataFrame` | Required | DataFrame with columns: `['date', 'pf_value', 'pf_return', 'index_fund_value', 'index_return', 'alpha']` |
| `risk_free_rate` | `float` | `0.065` | Annual risk-free rate (default 6.5% for India) |

### Returns

A dictionary containing all computed benchmark comparison metrics.

---

## Metrics Explained

### 1. Beta (β)

**What it measures**: Portfolio's sensitivity to market movements.

**Formula**: 

$$\beta = \frac{\text{Cov}(R_p, R_m)}{\text{Var}(R_m)}$$

**Interpretation**:
| Beta Value | Meaning |
|------------|---------|
| β = 1 | Moves exactly with the market |
| β > 1 | More volatile than market (e.g., 1.2 = 20% more volatile) |
| β < 1 | Less volatile than market (defensive) |
| β < 0 | Moves inversely to market (rare) |

**Code**:
```python
covariance = valid['pf_daily_return'].cov(valid['index_daily_return'])
index_variance = valid['index_daily_return'].var()
beta = covariance / index_variance
```

---

### 2. Tracking Error

**What it measures**: How much the portfolio's returns deviate from the benchmark. It's the annualized standard deviation of excess returns (portfolio return minus index return).

**Formula**:

$$\text{Tracking Error} = \sigma(\text{Excess Return}) \times \sqrt{252}$$

**Interpretation**:
| Tracking Error | Meaning |
|----------------|---------|
| ~1-2% | Portfolio closely tracks the index (passive/index fund) |
| ~3-5% | Moderate active management |
| ~5%+ | Significant divergence from benchmark (highly active) |

**Code**:
```python
df['excess_return'] = df['pf_daily_return'] - df['index_daily_return']
tracking_error = df['excess_return'].std() * np.sqrt(252)
```

---

### 3. Information Ratio (IR)

**What it measures**: Risk-adjusted outperformance—how much excess return you're getting per unit of tracking error. It answers: *"Is the active risk being rewarded?"*

**Formula**:

$$\text{IR} = \frac{\text{Annualized Excess Return}}{\text{Tracking Error}}$$

**Interpretation**:
| IR Value | Meaning |
|----------|---------|
| IR < 0 | Underperforming the benchmark |
| IR ~ 0-0.5 | Average active management |
| IR > 0.5 | Good active management |
| IR > 1.0 | Excellent active management (rare to sustain) |

**Code**:
```python
excess_return_mean = df['excess_return'].mean() * 252  # Annualized
information_ratio = excess_return_mean / tracking_error
```

---

### 4. Up Capture Ratio

**What it measures**: How the portfolio participates in market gains. It compares the portfolio's average return on days when the market is up vs. the market's average return on those same days.

**Formula**:

$$\text{Up Capture} = \frac{\text{Avg Portfolio Return (up days)}}{\text{Avg Index Return (up days)}} \times 100$$

**Interpretation**:
| Up Capture | Meaning |
|------------|---------|
| 100% | Captures exactly the market's upside |
| >100% | Outperforms in rising markets |
| <100% | Underperforms in rising markets |

**Code**:
```python
up_days = valid[valid['index_daily_return'] > 0]
up_capture = up_days['pf_daily_return'].mean() / up_days['index_daily_return'].mean() * 100
```

---

### 5. Down Capture Ratio

**What it measures**: How the portfolio participates in market losses. Lower is better—you want to capture less of the downside.

**Formula**:

$$\text{Down Capture} = \frac{\text{Avg Portfolio Return (down days)}}{\text{Avg Index Return (down days)}} \times 100$$

**Interpretation**:
| Down Capture | Meaning |
|--------------|---------|
| 100% | Falls exactly as much as the market |
| <100% | **Defensive** - falls less than market (good!) |
| >100% | Falls more than the market (bad) |

**Ideal Scenario**: Up Capture > 100% AND Down Capture < 100%

**Code**:
```python
down_days = valid[valid['index_daily_return'] < 0]
down_capture = down_days['pf_daily_return'].mean() / down_days['index_daily_return'].mean() * 100
```

---

### 6. Correlation

**What it measures**: How closely portfolio returns move with index returns (linear relationship).

**Interpretation**:
| Correlation | Meaning |
|-------------|---------|
| 1.0 | Perfect positive correlation (moves exactly together) |
| 0.7 - 0.9 | Strong positive correlation |
| 0.3 - 0.7 | Moderate correlation |
| 0.0 | No linear relationship |
| < 0 | Inverse relationship |

**Code**:
```python
correlation = valid['pf_daily_return'].corr(valid['index_daily_return'])
```

---

### 7. Alpha

**What it measures**: The portfolio's total outperformance (or underperformance) vs. the benchmark, expressed as a percentage.

**Note**: This is **not** Jensen's Alpha (risk-adjusted). It's simply the difference between portfolio return and index return.

**Code**:
```python
final_alpha = df['alpha'].iloc[-1]
```

---

### 8. Outperformance Days Percentage

**What it measures**: The percentage of trading days where the portfolio beat the index (i.e., alpha increased from the previous day).

**Code**:
```python
outperform_days = (df['alpha'].diff() > 0).sum()
outperform_pct = outperform_days / total_days * 100
```

---

## Return Dictionary

```python
{
    'portfolio_return_pct': 25.5,      # Total portfolio return (%)
    'index_return_pct': 18.2,          # Total benchmark return (%)
    'alpha_pct': 7.3,                  # Outperformance vs index (%)
    'beta': 0.95,                      # Market sensitivity
    'tracking_error_pct': 8.5,         # Deviation from benchmark (%)
    'information_ratio': 0.86,         # Risk-adjusted outperformance
    'up_capture_pct': 105.2,           # Upside participation (%)
    'down_capture_pct': 88.5,          # Downside participation (%)
    'correlation': 0.87,               # Return correlation with index
    'outperformance_days_pct': 52.3    # % of days beating index
}
```

---

## Example Usage

```python
import pandas as pd
from addtl_bt_fns import compute_benchmark_metrics

# Load comparison data
comparison_df = pd.read_csv('portfolio_vs_index.csv')

# Compute metrics
metrics = compute_benchmark_metrics(comparison_df, risk_free_rate=0.065)

# Print results
print(f"Alpha: {metrics['alpha_pct']}%")
print(f"Beta: {metrics['beta']}")
print(f"Information Ratio: {metrics['information_ratio']}")
print(f"Up Capture: {metrics['up_capture_pct']}%")
print(f"Down Capture: {metrics['down_capture_pct']}%")
```

---

## Interpretation Guide

### What Makes a Good Active Fund?

| Metric | Good Value | Why |
|--------|------------|-----|
| Alpha | > 0 | Beating the benchmark |
| Beta | ~1 (or context-dependent) | Matching market risk |
| Information Ratio | > 0.5 | Efficient use of active risk |
| Up Capture | > 100% | Outperforms in bull markets |
| Down Capture | < 100% | Protects in bear markets |
| Correlation | 0.7-0.9 | Diversified but still equity-like |

### Common Patterns

| Pattern | Up Capture | Down Capture | Interpretation |
|---------|------------|--------------|----------------|
| **Ideal** | 110% | 80% | Captures more upside, less downside |
| **Index-like** | 100% | 100% | Behaves exactly like benchmark |
| **Aggressive** | 130% | 120% | Amplifies both gains and losses |
| **Defensive** | 85% | 70% | Lower returns but better protection |
| **Poor** | 80% | 110% | Misses gains, amplifies losses |

---

## Notes

- **Trading Days**: Uses 252 trading days per year for annualization
- **Risk-Free Rate**: Default 6.5% is appropriate for Indian markets
- **Data Requirements**: Requires aligned dates between portfolio and index data
- **Edge Cases**: Handles division by zero gracefully (returns default values)
