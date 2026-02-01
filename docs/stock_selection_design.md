# Stock Selection Design

This document outlines the design choices in `src/stock_selection.py` for selecting and weighting stocks.

## Two Selection Paradigms

| Function | Categorization | Use Case |
|----------|---------------|----------|
| `select_and_weight_stocks_volatility` | Dynamic terciles (33rd/66th percentile) | When volatility-based diversification is priority |
| `select_and_weight_stocks_mcap` | Pre-labeled categories | When market cap exposure is priority |

## Core Design Decisions

### 1. Per-Quarter Independence
Each quarter is processed in isolation—percentile thresholds and selections are computed fresh per quarter, preventing lookahead bias and adapting to changing market conditions.

### 2. Configurable Selection Counts & Weights
Both functions accept parallel lists for counts and weights:
- **Counts**: `[n_high, n_med, n_low]` or `[n_large, n_mid, n_small]`
- **Weights**: Corresponding category weights for portfolio allocation

This separates *how many* stocks to pick from *how much* to allocate.

### 3. Selection Methods

| Method | Ranking Criterion | Rationale |
|--------|------------------|-----------|
| `probability` | Highest model probability | Maximize expected signal strength |
| `risk_adjusted` | `prob / volatility` | Balance conviction against risk; favors high-probability, low-volatility stocks |

### 4. Minimum Probability Threshold
Optional `min_prob_threshold` filters out low-confidence predictions before selection, ensuring only stocks meeting a confidence floor are considered.

### 5. Graceful Degradation
When fewer stocks exist than requested, the system:
- Emits a warning (not an error)
- Selects all available stocks in that category
- Continues processing

## Volatility Categorization (Dynamic)
Volatility function uses **within-quarter percentiles**:
- **High**: > 66th percentile
- **Medium**: 33rd–66th percentile  
- **Low**: ≤ 33rd percentile

This ensures balanced category sizes regardless of absolute volatility levels across different market regimes.

## Market Cap Categorization (Static)
Market cap function expects a pre-existing `category` column with values `largecap`, `midcap`, `smallcap`. Category boundaries are externally defined, allowing alignment with standard index definitions.

## Output Schema
Both functions return a standardized DataFrame:
```
quarter | co_name | cat | cat_weight [| risk_adj_score]
```
The `risk_adj_score` column is included only when using the risk-adjusted method.
