# Target Creation Process Documentation

This document outlines the process used to generate 6 target variables.

## 1. Base Metrics Calculation

First, we calculate the following quarterly metrics for each company:

*   **`entry`**: The average closing price of the first 3 trading days of the quarter. This represents the "entry" price level for the quarter. Note: in the backtest, the entry window is configurable via `entry_price_window`.
*   **`quarter_high_5`**: The average of the 5 highest prices (`high` column) achieved during the quarter. This represents the potential peak performance.
*   **`quarter_close_5`**: The average of the 5 highest closing prices (`close` column) achieved during the quarter. This represents the potential peak closing performance.
*   **`volatility`**: The standard deviation of the `daily_return` for the company within that specific quarter (forward looking).

Choosing the five highest high/close prices may or may not be optimal. Additional EDA can be carried out.

## 2. Performance Metrics (Features)

Using the base metrics, we derive two performance ratios that measure the potential upside during the quarter relative to the start:

*   **`(high5-entry)/entry`**: Measures the percentage increase from the start of the quarter to the average of the highest peaks.
    $$ \text{(high5-close3)/close3} = \frac{\text{quarter\_high\_5} - \text{avg\_first3\_close}}{\text{avg\_first3\_close}} $$

*   **`(close5-entry)/entry`**: Measures the percentage increase from the start of the quarter to the average of the highest closes.
    $$ \text{(close5-close3)/close3} = \frac{\text{quarter\_close\_5} - \text{avg\_first3\_close}}{\text{avg\_first3\_close}} $$

*   **`Sharpe based`**: These metrics divide the upside potential by the volatility. 
    *   `Sharpe_high` = `(high5-entry)/entry` / `volatility`
    *   `Sharpe_close` = `(close5-entry)/entry` / `volatility`

## 3. Target Generation

For each target, companies are ranked within their specific quarter based on one of the performance metrics. A percentile rank is assigned (0.0 to 1.0), and a binary label (0 or 1) is created based on a specific threshold.

### Target 1
*   **Metric Used**: `(high5-entry)/entry`
*   **Threshold**: Top 40% (Percentile > 0.6)
*   **Description**: Identifies companies that are in the top 40% of performers for that quarter based on their peak high prices.

### Target 2
*   **Metric Used**: `(close5-entry)/entry`
*   **Threshold**: Top 40% (Percentile > 0.6)
*   **Description**: Identifies companies that are in the top 40% of performers for that quarter based on their peak closing prices.

### Target 3
*   **Metric Used**: `(high5-entry)/entry`
*   **Threshold**: Top 20% (Percentile > 0.8)
*   **Description**: A stricter version of Target 1. Identifies the "elite" top 20% of performers based on peak high prices.

### Target 4
*   **Metric Used**: `(close5-entry)/entry`
*   **Threshold**: Top 20% (Percentile > 0.8)
*   **Description**: A stricter version of Target 2. Identifies the "elite" top 20% of performers based on peak closing prices.

### Target 5 (Sharpe - Highs)
*   **Metric Used**: `sharpe_high` = `(high5-entry)/entry` / `volatility`
*   **Threshold**: Top 20% (Percentile > 0.8)
*   **Description**: Identifies companies that provided the best risk-adjusted returns based on intraday highs. It rewards high returns but penalizes high volatility.

### Target 6 (Sharpe - Closes)
*   **Metric Used**: `sharpe_close` = `(close5-entry)/entry` / `volatility`
*   **Threshold**: Top 20% (Percentile > 0.8)
*   **Description**: Identifies companies that provided the best risk-adjusted returns based on closing prices. It rewards steady, sustained trends over volatile spikes.

## 4. Discussion: High5 vs. Close5

Choosing between `high5` (average of the 5 highest intraday highs) and `close5` (average of the 5 highest closing prices) fundamentally changes what the model learns to predict.

### `high5` (The "Optimistic" / Limit Order View)
This metric looks at the absolute highest prices reached during the quarter, regardless of where the stock closed that day.

*   **What it captures:** Maximum potential volatility and candle "wicks". It rewards stocks that spike aggressively.
*   **Best for:** Strategies using **Limit Orders** (Take Profit). If a strategy sets a sell order at +20%, `high5` is appropriate because the price only needs to touch that level momentarily.
*   **Risk:** Can be "noisy". A stock might spike on low liquidity and immediately collapse.

### `close5` (The "Conservative" / Sustained View)
This metric looks at the price at the end of the trading day.

*   **What it captures:** Sustained value. For a price to close high, buyers had to support that level throughout the day. It filters out fleeting intraday spikes.
*   **Best for:** Strategies executing at **Market on Close** or trend-following. For swing traders checking positions at EOD, `close5` is a more realistic representation of realizable profit.
*   **Risk:** Conservative. Might miss out on stocks that spike significantly intraday but close with smaller gains.

### Summary Table

| Feature | `high5` | `close5` |
| :--- | :--- | :--- |
| **Behavior** | Rewards **Volatility** & Spikes | Rewards **Stability** & Trends |
| **Execution** | Assumes **Limit Orders** (Take Profit) | Assumes **EOD / Market Orders** |
| **Noise** | Higher (includes wicks/noise) | Lower (confirmed prices) |
| **Difficulty** | Harder to realize in practice | Easier to realize in practice |

*Last updated: March 2026*