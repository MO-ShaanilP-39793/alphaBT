# Configuration Reference

Complete guide to all configuration parameters in CCQPF. This is your YAML reference manual.

---

## Table of Contents

- [Configuration File Overview](#configuration-file-overview)
- [Data Requirements](#data-requirements)
- [strategy_config.yaml Parameters](#strategy_configyaml-parameters)
- [tuning_config.yaml Parameters](#tuning_configyaml-parameters)
- [Common Configuration Patterns](#common-configuration-patterns)
- [Validation and Troubleshooting](#validation-and-troubleshooting)

---

## Configuration File Overview

CCQPF uses **two YAML configuration files**, each serving a different purpose:

| File | Purpose | When to Edit |
|------|---------|--------------|
| **strategy_config.yaml** | Single backtest configuration | Before each manual backtest run |
| **tuning_config.yaml** | Hyperparameter optimization setup | When defining search spaces for Optuna |

Both files are located in `src/` directory and follow YAML syntax.

### Template Files

The framework provides starter templates:
- `src/strategy_config_template.yaml` – Annotated example for backtests
- `src/tuning_config_template.yaml` – Annotated example for optimization

Copy these to `strategy_config.yaml` / `tuning_config.yaml` and customize.

---

## Data Requirements

### Input Data (ML Inference Results)

Your ML model's predictions. Required columns depend on configuration:

#### Minimum Required (for top-k selection):
```csv
quarter,co_name,prob
202002,Reliance Industries,0.85
202002,HDFC Bank,0.72
202405,Infosys,0.68
```

| Column | Type | Description |
|--------|------|-------------|
| `quarter` | Integer | YYYYMM format where MM ∈ {02, 05, 08, 11} |
| `co_name` | String | Stock name (must exactly match price data) |
| `prob` | Float | ML model probability (0.0 to 1.0) |

#### Additional Columns (conditional):

| Column | When Required | Valid Values |
|--------|---------------|--------------|
| `volatility` | `category_scheme: 'volatility'` OR `selection_method: 'risk_adjusted'` | Float (annualized std dev, e.g., 0.35 for 35%) |
| `category` | `category_scheme: 'mcap'` | Exact strings: `largecap`, `midcap`, `smallcap` |

#### Preselected Portfolio Mode:
If `run_stock_selection: false`, schema changes:
```csv
quarter,co_name,stock_weight,category
202002,Reliance Industries,0.05,largecap
202002,HDFC Bank,0.04,largecap
```

| Column | Required? | Description |
|--------|-----------|-------------|
| `stock_weight` | Yes | Capital allocation (should sum to ≈1.0 per quarter) |
| `category` | Only if using `tp_mode: 'tiered'` and `sl_mode: 'tiered'` | For category-specific TP/SL |

**File Format**: CSV or Parquet (auto-detected by extension)

---

### Price Data (OHLCV)

Historical price data for all stocks:

```csv
date,co_name,open,high,low,close
2020-02-15,Reliance Industries,1450.0,1475.0,1440.0,1460.0
2020-02-16,Reliance Industries,1462.0,1480.0,1455.0,1470.0
```

| Column | Type | Description |
|--------|------|-------------|
| `date` | Date | YYYY-MM-DD or any parseable format |
| `co_name` | String | Stock name (must match inference data) |
| `open` | Float | Opening price |
| `high` | Float | Intraday high |
| `low` | Float | Intraday low |
| `close` | Float | Closing price |

**Optional**: `volume` column (not currently used but can be present)

**File Format**: CSV or Parquet

**Coverage**: Must span all quarters in your backtest range with minimal gaps. The framework validates coverage and filters out stocks with insufficient data.

---

### Index Data (Benchmark)

Benchmark index for comparison (e.g., Nifty 50, Sensex):

```csv
date,value
2020-02-15,12000.5
2020-02-16,12050.3
```

| Column | Type | Description |
|--------|------|-------------|
| `date` | Date | YYYY-MM-DD or parseable format |
| `value` | Float | Index level |

**File Format**: CSV or Parquet

**Note**: Index data is used for:
1. Benchmark comparison in reports
2. Regime filter calculations (if enabled)
3. Volatility adjustment calculations (if enabled)

---

### Quarter Encoding Convention

Quarters are encoded as **integers in YYYYMM format**:

| Quarter Code | Trading Period | Entry Window | Mandatory Exit |
|--------------|----------------|--------------|----------------|
| `202002` | Feb–May 2020 | Feb 15-17 | May 30 |
| `202005` | May–Aug 2020 | May 31–Jun 2 | Aug 14 |
| `202008` | Aug–Nov 2020 | Aug 15-17 | Nov 14 |
| `202011` | Nov 2020–Feb 2021 | Nov 15-17 | Feb 14 (next year) |
| `202102` | Feb–May 2021 | Feb 15-17 | May 30 |

**Month Codes**: Only `02`, `05`, `08`, `11` are valid. Any other month will cause errors.

**Entry Logic**: Positions are entered using the mean closing price over the first N trading days of the period (configurable via `entry_price_window`, default 3).

**Exit Logic**: Positions exit early if TP/SL/regime triggers fire, otherwise at the mandatory cutoff date.

---

### Path Conventions

All file paths in configs are **relative to the `src/` directory**:

```yaml
# Correct (relative to src/)
input_data_path: "../data/inference_data.csv"
price_data_path: "../data/price_data/ohlcv.parquet"

# Incorrect (absolute paths may break portability)
input_data_path: "C:/Users/you/ccqpf/data/inference_data.csv"
```

**Why?** All scripts run from `src/` as the working directory: `cd src && python backtest_strategy.py`

---

## strategy_config.yaml Parameters

### Data Paths

```yaml
input_data_path: "../data/inference_data_formatted/formatted_cat_data.csv"
price_data_path: "../data/price_data/ohlcv2.parquet"
index_data_path: "../data/index_data/nifty500_index_data.csv"
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_data_path` | String | Path to ML inference data (CSV/Parquet) |
| `price_data_path` | String | Path to OHLCV price data (CSV/Parquet) |
| `index_data_path` | String | Path to benchmark index data (CSV/Parquet) |

**All paths are relative to `src/` directory.**

---

### Quarter Range

```yaml
first_quarter: 202002  # Start from Feb 2020
last_quarter: 202411   # End at Nov 2024

# OR use null for full range
first_quarter: null    # Use earliest quarter in data
last_quarter: null     # Use latest quarter in data
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `first_quarter` | Integer or `null` | Starting quarter (YYYYMM format) |
| `last_quarter` | Integer or `null` | Ending quarter (YYYYMM format) |

**Effect**: Filters both inference data and price data to this range.

**Tip**: Narrow the range for faster backtests during development, expand for final analysis.

---

### Stock Selection Mode

```yaml
run_stock_selection: true  # Run selection algorithm
# OR
run_stock_selection: false  # Use preselected portfolio from input data
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_stock_selection` | Boolean | `true` = select stocks based on probabilities<br>`false` = use preselected portfolio (input must have `stock_weight` column) |

**When `false`**: Framework skips selection, expects input data with `quarter`, `co_name`, `stock_weight` columns.

---

### Selection Type (when `run_stock_selection: true`)

```yaml
selection_type: 'category_based'  # Diversify by categories
# OR
selection_type: 'top_k'           # Simple top-N selection
```

| Parameter | Type | Valid Values | Description |
|-----------|------|--------------|-------------|
| `selection_type` | String | `'category_based'`, `'top_k'` | Stock selection paradigm |

---

### Category-Based Selection Parameters

```yaml
selection_type: 'category_based'
category_scheme: 'volatility'             # or 'mcap'
category_counts: [10, 10, 10]             # Stocks per category
category_weights: [0.33, 0.33, 0.34]      # Capital per category
category_based_selection_weighting_scheme: 'use_category_weights'  # or 'equal'
```

#### category_scheme

| Value | Description | Required Input Column |
|-------|-------------|-----------------------|
| `'volatility'` | Dynamic terciles (high/medium/low vol) | `volatility` |
| `'mcap'` | Pre-labeled market cap | `category` (`largecap`, `midcap`, `smallcap`) |

#### category_counts

**Type**: List of 3 integers

**Example**: `[10, 10, 10]` = Select 10 stocks from each category

**Order**:
- For `volatility`: `[high_vol_count, medium_vol_count, low_vol_count]`
- For `mcap`: `[largecap_count, midcap_count, smallcap_count]`

**Total stocks selected** = sum of counts (e.g., 10 + 10 + 10 = 30)

#### category_weights

**Type**: List of 3 floats (must sum to 1.0)

**Example**: `[0.2, 0.3, 0.5]` = 20% capital to category 1, 30% to category 2, 50% to category 3

**Order**: Matches `category_counts` order

**Effect**: Only used if `weighting_scheme: 'use_category_weights'`

#### category_based_selection_weighting_scheme

| Value | Behavior |
|-------|----------|
| `'use_category_weights'` | Divide capital by `category_weights`, then equal within category |
| `'equal'` | Ignore `category_weights`, allocate 1/N to each selected stock |

**Example**:
```yaml
category_counts: [10, 10, 10]
category_weights: [0.5, 0.3, 0.2]
weighting_scheme: 'use_category_weights'

# Result:
# - Category 1 (10 stocks): 50% of capital → 5% per stock
# - Category 2 (10 stocks): 30% of capital → 3% per stock
# - Category 3 (10 stocks): 20% of capital → 2% per stock

# If weighting_scheme: 'equal' instead:
# - All 30 stocks: 1/30 = 3.33% per stock
```

---

### Top-K Selection Parameters

```yaml
selection_type: 'top_k'
top_k_config:
  k: 30                       # Select top 30 stocks
  weighting_scheme: 'equal'   # Currently only option
```

#### top_k_config.k

**Type**: Integer

**Description**: Number of stocks to select per quarter

**Typical range**: 20-50

#### top_k_config.weighting_scheme

**Type**: String

**Valid values**: `'equal'` (only option currently)

**Effect**: Each selected stock gets `1/k` of capital

---

### Selection Method (Ranking Criterion)

```yaml
selection_method: 'probability'     # Rank by ML probability
# OR
selection_method: 'risk_adjusted'   # Rank by prob/volatility
```

| Value | Ranking Formula | Required Input Column |
|-------|----------------|------------------------|
| `'probability'` | Sort by `prob` descending | `prob` |
| `'risk_adjusted'` | Sort by `prob / volatility` descending | `prob`, `volatility` |

**Applies to**: Both category-based (within-category ranking) and top-k (global ranking)

---

### Minimum Probability Threshold

```yaml
min_prob_threshold: 0.5   # Only consider stocks with prob >= 0.5
# OR
min_prob_threshold: null  # No filtering
```

**Type**: Float (0.0 to 1.0) or `null`

**Effect**: Filters inference data before selection. Stocks below threshold are excluded from consideration.

**Use case**: Remove low-confidence predictions.

---

### TP/SL Global Toggles

```yaml
tp_enabled: true   # Enable take profit exits
sl_enabled: true   # Enable stop loss exits
entry_price_window: 3  # Number of trading days averaged for entry price
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tp_enabled` | Boolean | `true` | Allow early exits when profit target hit |
| `sl_enabled` | Boolean | `true` | Allow early exits when stop loss hit |
| `entry_price_window` | Integer | `3` | Number of trading days to average for entry price calculation |

**Both `tp_enabled` and `sl_enabled` `false`** = Hold all positions until quarter-end mandatory exit.

---

### TP/SL Mode

TP and SL modes can be configured **independently**, allowing different threshold strategies for take-profit and stop-loss exits.

```yaml
tp_mode: 'tiered'  # Category-specific TP percentages
sl_mode: 'tiered'  # Category-specific SL percentages
# OR
tp_mode: 'flat'    # Uniform TP percentage
sl_mode: 'flat'    # Uniform SL percentage
# OR
tp_mode: 'atr'     # ATR-based dynamic TP
sl_mode: 'atr'     # ATR-based dynamic SL
# OR
tp_mode: 'pivot'   # Resistance levels for TP
sl_mode: 'pivot'   # Support levels for SL
```

| Parameter | Type | Valid Values | Description |
|-----------|------|--------------|-------------|
| `tp_mode` | String | `'tiered'`, `'flat'`, `'atr'`, `'pivot'` | Take-profit threshold strategy |
| `sl_mode` | String | `'tiered'`, `'flat'`, `'atr'`, `'pivot'` | Stop-loss threshold strategy |

| Value | Description | Required Config Section |
|-------|-------------|-------------------------|
| `'tiered'` | Category-specific TP/SL | `tiered_config` |
| `'flat'` | Same % for all stocks | `flat_config` |
| `'atr'` | ATR-based dynamic levels | `atr_config` |
| `'pivot'` | Pivot point levels | `pivot_config` |

**Note**: While both modes are typically set to the same value, you can mix them (e.g., `tp_mode: 'atr'` with `sl_mode: 'flat'`) for asymmetric exit strategies.

---

### Tiered Mode Configuration

Used when `tp_mode: 'tiered'` and/or `sl_mode: 'tiered'`:

```yaml
tiered_config:
  tp_pct:
    high_volatility: 0.10        # 10% TP
    medium_volatility: 0.05      # 5% TP
    low_volatility: 0.02         # 2% TP
  sl_pct:
    high_volatility: 0.10        # 10% SL
    medium_volatility: 0.10      # 10% SL
    low_volatility: 0.10         # 10% SL
  default_tp_pct: 0.05           # 5% default TP (fallback)
  default_sl_pct: 0.05           # 5% default SL (fallback)
```

**Structure**: Single `tiered_config` block with `tp_pct` and `sl_pct` dictionaries keyed by category, plus `default_tp_pct` / `default_sl_pct` fallbacks.

**Category Keys**:
- For `volatility` scheme: `high_volatility`, `medium_volatility`, `low_volatility`
- For `mcap` scheme: `largecap`, `midcap`, `smallcap`

**Values**: Float (0.01 = 1%, 0.10 = 10%)

**Example**: Stock in `high_volatility` category with entry at ₹100:
- TP = ₹100 × (1 + 0.10) = ₹110
- SL = ₹100 × (1 - 0.10) = ₹90

**Default fallback**: When a stock has no category label (e.g., preselected mode with no `category` column), `default_tp_pct` and `default_sl_pct` are used.

---

### Flat Mode Configuration

Used when `tp_mode: 'flat'` and/or `sl_mode: 'flat'`:

```yaml
flat_config:
  tp_pct: 0.05   # 5% TP for all stocks
  sl_pct: 0.05   # 5% SL for all stocks
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tp_pct` | Float | Take profit percentage (e.g., 0.05 = 5%) |
| `sl_pct` | Float | Stop loss percentage (e.g., 0.05 = 5%) |

**Simplest TP/SL mode** – only 2 parameters.

---

### ATR Mode Configuration

Used when `tp_mode: 'atr'` and/or `sl_mode: 'atr'`:

```yaml
atr_config:
  period: 14            # ATR calculation period (days)
  tp_multiplier: 2.0    # TP = entry + (2.0 × ATR)
  sl_multiplier: 1.5    # SL = entry - (1.5 × ATR)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `period` | Integer | Number of days for ATR calculation (typical: 7-21) |
| `tp_multiplier` | Float | Multiplier for TP threshold (typical: 1.5-3.5) |
| `sl_multiplier` | Float | Multiplier for SL threshold (typical: 1.0-2.5) |

**How it works**: 
- ATR = Average True Range over `period` days
- TP = `entry_price + (tp_multiplier × ATR)`
- SL = `entry_price - (sl_multiplier × ATR)`

**Tuning tips**:
- Shorter period (7-10) = more reactive to recent volatility
- Longer period (14-21) = smoother, less noisy
- Higher multipliers = wider bands, fewer exits

See [dynamic_levels_reference.md](dynamic_levels_reference.md) for details.

---

### Pivot Mode Configuration

Used when `tp_mode: 'pivot'` and/or `sl_mode: 'pivot'`:

```yaml
pivot_config:
  lookback_days: 60   # Historical data for pivot calculation
  tp_level: 'R1'      # Take profit at Resistance 1
  sl_level: 'S1'      # Stop loss at Support 1
```

| Parameter | Type | Valid Values | Description |
|-----------|------|--------------|-------------|
| `lookback_days` | Integer | 20-120 | Days of price history for pivot calc |
| `tp_level` | String | `'R1'`, `'R2'`, `'R3'` | Resistance level for TP |
| `sl_level` | String | `'S1'`, `'S2'`, `'S3'` | Support level for SL |

**Pivot formulas** (classic):
```
Pivot Point (P) = (High + Low + Close) / 3
R1 = (2 × P) - Low
R2 = P + (High - Low)
R3 = High + 2 × (P - Low)
S1 = (2 × P) - High
S2 = P - (High - Low)
S3 = Low - 2 × (High - P)
```

**Level selection**:
- R1/S1 = closest resistance/support (conservative)
- R2/S2 = mid-range (moderate)
- R3/S3 = extended (aggressive)

See [dynamic_levels_reference.md](dynamic_levels_reference.md) for theory.

---

### Index Exit: Regime Filter

```yaml
index_exit:
  regime_filter:
    enabled: false           # Set to true to activate
    ma_period: 20            # Moving average period (days)
    exit_threshold: -0.02    # Exit if index < MA by 2%
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | Boolean | Activate regime-based exits |
| `ma_period` | Integer | Days for index MA calculation (typical: 10-50) |
| `exit_threshold` | Float | Negative value, e.g., -0.02 = 2% below MA |

**Trigger condition**: `(index_value - MA) / MA < exit_threshold`

**Effect**: If triggered, **all positions** exit at close price that day.

**Use case**: Exit to cash during bear markets / downtrends.

---

### Index Exit: Volatility Adjustment

```yaml
index_exit:
  vol_adjustment:
    enabled: false              # Set to true to activate
    lookback: 20                # Vol calculation period (days)
    high_vol_threshold: 0.25    # Annualized vol > 25% = high
    low_vol_threshold: 0.15     # Annualized vol < 15% = low
    high_vol_multiplier: 1.5    # Scale TP/SL by 1.5× in high vol
    low_vol_multiplier: 0.8     # Scale TP/SL by 0.8× in low vol
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | Boolean | Activate volatility-based TP/SL scaling |
| `lookback` | Integer | Days for index volatility calculation |
| `high_vol_threshold` | Float | Annualized vol threshold for "high" (e.g., 0.25 = 25%) |
| `low_vol_threshold` | Float | Annualized vol threshold for "low" (e.g., 0.15 = 15%) |
| `high_vol_multiplier` | Float | TP/SL multiplier during high vol (>1.0 = widen) |
| `low_vol_multiplier` | Float | TP/SL multiplier during low vol (<1.0 = tighten) |

**How it works**:
1. Daily: Calculate index volatility over `lookback` days (annualized)
2. Classify regime:
   - Vol > `high_vol_threshold` → Apply `high_vol_multiplier`
   - Vol < `low_vol_threshold` → Apply `low_vol_multiplier`
   - Otherwise → No adjustment (1.0×)
3. Scale TP/SL thresholds accordingly

**Example**:
- Base TP = 5%, Base SL = 5%
- High vol regime → TP = 7.5%, SL = 7.5%
- Low vol regime → TP = 4%, SL = 4%

---

### Reporting Options

```yaml
generate_report: true              # Create backtest_report.xlsx

report_sub_periods:                # Custom year ranges for sub-period analysis
  - [2020, 2022]
  - [2023, 2025]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `generate_report` | Boolean | Generate `backtest_report.xlsx` (consolidated workbook with all metrics, charts, trade data) |
| `report_sub_periods` | List of [start_year, end_year] | Custom sub-period breakdowns in report |

**Recommendation**: Keep `true` for comprehensive analysis. Set to `false` during tuning for speed.

---

## tuning_config.yaml Parameters

For hyperparameter optimization with Optuna.

### Fixed Parameters Section

Parameters that **do not change** across trials:

```yaml
fixed:
  input_data_path: "../data/inference_data.csv"
  price_data_path: "../data/price_data/ohlcv.parquet"
  index_data_path: "../data/index_data/nifty500.csv"
  first_quarter: 202002
  last_quarter: 202411
  run_stock_selection: true
  generate_report: false             # Keep false for speed
  validate_input_data: false        # Set true for first trial, false after
```

**All `strategy_config.yaml` parameters are valid here.** Common fixed params:
- Data paths (don't change during tuning)
- Quarter range (consistent evaluation period)
- Reporting flags (off for speed, enable for final run with best config)

---

### Search Space Section

Parameters that **Optuna samples** across trials. Each parameter has a `type` and bounds/choices:

#### Categorical Parameters

```yaml
search_space:
  selection_type:
    type: categorical
    choices: ['category_based', 'top_k']
  
  tpsl_mode:
    type: categorical
    choices: ['tiered', 'flat', 'atr', 'pivot']
  
  independent_tpsl_modes:        # Optional: sample tp_mode and sl_mode independently
    type: categorical
    choices: [true, false]       # Default: false (one mode for both TP and SL)
  
  tp_enabled:
    type: categorical
    choices: [true, false]
```

**Note**: The `tpsl_mode` search space key selects the mode choices. When `independent_tpsl_modes` is `false` (default), the sampled mode is assigned to both `tp_mode` and `sl_mode`. When `true`, `tp_mode` and `sl_mode` are sampled independently from the same choices list. The runtime config always uses `tp_mode` and `sl_mode` keys.

**Type**: `categorical`

**Required key**: `choices` (list of valid values)

**Use for**: Discrete options (strings, booleans, specific integers)

#### Integer Parameters

```yaml
search_space:
  atr_period:
    type: int
    low: 7        # Minimum value
    high: 21      # Maximum value
```

**Type**: `int`

**Required keys**: `low` (inclusive), `high` (inclusive)

**Sampling**: Uniform integers in [low, high]

#### Float Parameters

```yaml
search_space:
  flat_tp:
    type: float
    low: 0.02      # Minimum value
    high: 0.15     # Maximum value
    step: 0.01     # Optional: discretization step
```

**Type**: `float`

**Required keys**: `low`, `high`

**Optional key**: `step` (discretize to multiples of step)

**Sampling**: 
- Without `step`: Continuous uniform in [low, high]
- With `step`: Discrete values at low, low+step, low+2×step, ..., high

---

### Mode-Specific Search Spaces

**Conditional parameters** – only sampled when certain modes are selected:

#### top_k Section (when `selection_type: 'top_k'`)

```yaml
top_k:
  k:
    type: int
    low: 10
    high: 50
```

Sampled only when `selection_type` trial value is `'top_k'`.

#### Category-Based Sections (when `selection_type: 'category_based'`)

```yaml
category_counts:
  type: categorical
  choices:
    - [10, 10, 10]
    - [5, 10, 15]
    - [15, 15, 15]

category_weights:
  type: categorical
  choices:
    - [0.33, 0.33, 0.34]
    - [0.2, 0.3, 0.5]
```

Sampled only when `selection_type` trial value is `'category_based'`.

#### tiered_tpsl Section (when `tpsl_mode: 'tiered'`)

```yaml
tiered_tpsl:
  tiered_tp_thresholds:
    type: float
    low: 0.05
    high: 0.20
    step: 0.01
  tiered_sl_thresholds:
    type: float
    low: 0.02
    high: 0.10
    step: 0.01
  # ... etc for all categories
```

Sampled only when `tpsl_mode` trial value is `'tiered'`.

#### flat_tpsl Section (when `tpsl_mode: 'flat'`)

```yaml
flat_tpsl:
  tp:
    type: float
    low: 0.02
    high: 0.15
    step: 0.01
  sl:
    type: float
    low: 0.02
    high: 0.15
    step: 0.01
```

Sampled only when `tpsl_mode` trial value is `'flat'`.

#### atr_tpsl Section (when `tpsl_mode: 'atr'`)

```yaml
atr_tpsl:
  period:
    type: int
    low: 7
    high: 21
  tp_multiplier:
    type: float
    low: 1.5
    high: 3.5
    step: 0.1
  sl_multiplier:
    type: float
    low: 1.0
    high: 2.5
    step: 0.1
```

Sampled only when `tpsl_mode` trial value is `'atr'`.

#### pivot_tpsl Section (when `tpsl_mode: 'pivot'`)

```yaml
pivot_tpsl:
  lookback_days:
    type: int
    low: 20
    high: 120
  tp_level:
    type: categorical
    choices: ['R1', 'R2', 'R3']
  sl_level:
    type: categorical
    choices: ['S1', 'S2', 'S3']
```

Sampled only when `tpsl_mode` trial value is `'pivot'`.

#### index_exit Section (when `regime_filter_enabled: true` or `vol_adjustment_enabled: true`)

```yaml
index_exit:
  regime_ma_period:
    type: int
    low: 10
    high: 50
  regime_exit_threshold:
    type: float
    low: -0.05
    high: -0.01
    step: 0.01
  
  vol_lookback:
    type: int
    low: 10
    high: 30
  high_vol_multiplier:
    type: float
    low: 1.2
    high: 2.0
    step: 0.1
  low_vol_multiplier:
    type: float
    low: 0.5
    high: 0.9
    step: 0.1
```

Conditional sampling based on `regime_filter_enabled` and `vol_adjustment_enabled` in search space.

---

### Optuna Settings

```yaml
optuna:
  study_name: "backtest_optimization_v1"
  objective: "calmar"              # 'calmar', 'cagr', or 'mdd'
  n_trials: 100
  sampler: "TPE"                   # 'TPE', 'Random', 'CmaEs'
  direction: "maximize"            # 'maximize' or 'minimize'
  load_if_exists: true
  calmar_cap: 10.0
  failure_penalty: -999.0
```

| Parameter | Type | Valid Values | Description |
|-----------|------|--------------|-------------|
| `study_name` | String | Any | Unique identifier for this optimization run |
| `objective` | String | `'calmar'`, `'cagr'`, `'mdd'` | What metric to optimize |
| `n_trials` | Integer | 1-1000+ | Number of trials to run (default in CLI: 100) |
| `sampler` | String | `'TPE'`, `'Random'`, `'CmaEs'` | Optuna sampling algorithm |
| `direction` | String | `'maximize'`, `'minimize'` | Optimization direction |
| `load_if_exists` | Boolean | `true`, `false` | Resume existing study or fail if exists |
| `calmar_cap` | Float | 1.0-20.0 | Cap Calmar ratio to avoid inf/inflated values |
| `failure_penalty` | Float | Any | Value returned if trial crashes (use negative for maximize, positive for minimize) |

**Objective details**:
- `'calmar'`: Maximize Calmar Ratio (CAGR / |Max Drawdown|) – best for risk-adjusted returns
- `'cagr'`: Maximize Compound Annual Growth Rate – pure return focus
- `'mdd'`: **Minimize** Maximum Drawdown – risk minimization (set `direction: 'minimize'`)

**Sampler details**:
- `'TPE'` (Tree-structured Parzen Estimator): Default, good for most use cases
- `'Random'`: Baseline, useful for small search spaces
- `'CmaEs'`: Evolution Strategy, good for continuous parameter spaces

**Tip**: Override `n_trials` via CLI: `python run_tuning.py --n-trials 200`

---

## Common Configuration Patterns

### Pattern 1: Simple Equal-Weight Top-30

```yaml
# strategy_config.yaml
run_stock_selection: true
selection_type: 'top_k'
top_k_config:
  k: 30
  weighting_scheme: 'equal'
selection_method: 'probability'
min_prob_threshold: 0.5

tp_enabled: true
sl_enabled: true
tp_mode: 'flat'
sl_mode: 'flat'
flat_config:
  tp_pct: 0.10
  sl_pct: 0.05
```

**Use case**: Simplest backtest, trust top probabilities, uniform exits.

---

### Pattern 2: Volatility-Diversified with ATR Exits

```yaml
# strategy_config.yaml
run_stock_selection: true
selection_type: 'category_based'
category_scheme: 'volatility'
category_counts: [10, 10, 10]
category_weights: [0.33, 0.33, 0.34]
category_based_selection_weighting_scheme: 'use_category_weights'
selection_method: 'probability'

tp_enabled: true
sl_enabled: true
tp_mode: 'atr'
sl_mode: 'atr'
atr_config:
  period: 14
  tp_multiplier: 2.0
  sl_multiplier: 1.5
```

**Use case**: Diversify by volatility, use volatility-aware exits (ATR).

---

### Pattern 3: Market-Cap Tilt with Regime Protection

```yaml
# strategy_config.yaml
run_stock_selection: true
selection_type: 'category_based'
category_scheme: 'mcap'
category_counts: [5, 10, 15]       # Favor small-caps
category_weights: [0.2, 0.3, 0.5]  # More capital to small
category_based_selection_weighting_scheme: 'use_category_weights'
selection_method: 'probability'

tp_enabled: true
sl_enabled: true
tp_mode: 'tiered'
sl_mode: 'tiered'
tiered_config:
  tp_pct:
    largecap: 0.05
    midcap: 0.08
    smallcap: 0.15
  sl_pct:
    largecap: 0.05
    midcap: 0.08
    smallcap: 0.12
  default_tp_pct: 0.05
  default_sl_pct: 0.05

index_exit:
  regime_filter:
    enabled: true
    ma_period: 20
    exit_threshold: -0.02
```

**Use case**: Overweight small-caps, wider TP/SL for riskier segments, exit during downturns.

---

### Pattern 4: Tuning Top-K with Flat TP/SL

```yaml
# tuning_config.yaml
fixed:
  input_data_path: "../data/inference_data.csv"
  price_data_path: "../data/ohlcv.parquet"
  index_data_path: "../data/nifty500.csv"
  first_quarter: 202002
  last_quarter: 202411
  run_stock_selection: true
  selection_type: 'top_k'  # Fixed to top-k
  tp_mode: 'flat'   # Fixed to flat
  sl_mode: 'flat'   # Fixed to flat
  tp_enabled: true
  sl_enabled: true

search_space:
  selection_method:
    type: categorical
    choices: ['probability', 'risk_adjusted']
  
  min_prob_threshold:
    type: float
    low: 0.4
    high: 0.7
    step: 0.05

top_k:
  k:
    type: int
    low: 20
    high: 50

flat_tpsl:
  tp:
    type: float
    low: 0.05
    high: 0.20
    step: 0.01
  sl:
    type: float
    low: 0.03
    high: 0.10
    step: 0.01

optuna:
  study_name: "top_k_flat_optimization"
  objective: "calmar"
  n_trials: 100
```

**Search dimensions**: k (20-50), TP (5%-20%), SL (3%-10%), selection method, min threshold.

---

### Pattern 5: Comprehensive ATR Tuning

```yaml
# tuning_config.yaml
search_space:
  selection_type:
    type: categorical
    choices: ['category_based', 'top_k']
  
  tpsl_mode:
    type: categorical
    choices: ['flat', 'atr']

atr_tpsl:
  period:
    type: int
    low: 7
    high: 21
  tp_multiplier:
    type: float
    low: 1.5
    high: 3.5
    step: 0.1
  sl_multiplier:
    type: float
    low: 1.0
    high: 2.5
    step: 0.1

top_k:
  k:
    type: int
    low: 20
    high: 50

optuna:
  study_name: "atr_comprehensive"
  objective: "calmar"
  n_trials: 200
```

**Search dimensions**: Selection type, TP/SL mode, ATR parameters, k (if top-k selected).

---

## Validation and Troubleshooting

### Pre-Flight Checklist

Before running backtest:

✅ **Data files exist** at paths specified in config
✅ **Quarter range** is valid (YYYYMM format, only 02/05/08/11)
✅ **Required columns** present in input data:
   - Always: `quarter`, `co_name`, `prob`
   - If `category_scheme: 'volatility'`: `volatility`
   - If `category_scheme: 'mcap'`: `category` with `largecap`/`midcap`/`smallcap`
   - If `selection_method: 'risk_adjusted'`: `volatility`
   - If `run_stock_selection: false`: `stock_weight`
✅ **Price data coverage** spans quarter range with minimal gaps
✅ **Stock names** match exactly between inference and price data
✅ **TP/SL config** matches mode:
   - `tp_mode`/`sl_mode: 'tiered'` → `tiered_config`
   - `tp_mode`/`sl_mode: 'flat'` → `flat_config`
   - `tp_mode`/`sl_mode: 'atr'` → `atr_config`
   - `tp_mode`/`sl_mode: 'pivot'` → `pivot_config`

---

### Common Validation Errors

#### "KeyError: 'volatility'"

**Cause**: Config requires `volatility` column but input data doesn't have it.

**Needed when**:
- `category_scheme: 'volatility'`
- `selection_method: 'risk_adjusted'`

**Fix**: Add `volatility` column to input data OR change to `category_scheme: 'mcap'` / `selection_method: 'probability'`.

---

#### "KeyError: 'category'"

**Cause**: Config requires `category` column but input data doesn't have it.

**Needed when**:
- `category_scheme: 'mcap'`
- `run_stock_selection: false` with `tp_mode: 'tiered'` / `sl_mode: 'tiered'` (without `default_tp_pct`/`default_sl_pct` in `tiered_config`)

**Fix**: Add `category` column OR use `category_scheme: 'volatility'` OR set `default_tp_pct`/`default_sl_pct` in `tiered_config`.

---

#### "Many stocks filtered due to insufficient price data"

**Cause**: Price data has gaps or doesn't cover all quarters.

**Check**:
1. Price data date range covers `first_quarter` to `last_quarter`
2. All stocks in inference data have corresponding price rows
3. No multi-day gaps in price data

**Fix**: Extend price data coverage OR narrow `first_quarter`/`last_quarter` range.

---

#### "stock_weight values do not sum to 1.0 per quarter"

**Cause**: In preselected mode (`run_stock_selection: false`), weights should sum to ~1.0 per quarter.

**Tolerance**: Typically 0.98-1.02 is acceptable.

**Fix**: Normalize weights in input data: `weight_i = weight_i / sum(weights)` per quarter.

---

#### "Invalid quarter format"

**Cause**: Quarter value not in YYYYMM format with MM ∈ {02, 05, 08, 11}.

**Examples**:
- ❌ `202001` (January not valid)
- ❌ `20202` (wrong format)
- ✅ `202002` (Feb 2020 quarter)

**Fix**: Correct quarter encoding in input data.

---

#### "FileNotFoundError: [path]"

**Cause**: Data file path doesn't exist relative to `src/` directory.

**Common mistake**: Using absolute paths or forgetting `../` prefix.

**Fix**:
```yaml
# Correct (relative to src/)
input_data_path: "../data/inference_data.csv"

# Incorrect
input_data_path: "data/inference_data.csv"  # Missing ../
```

---

### Performance Considerations

**Slow backtests?**

**Speed up by**:
- Narrow quarter range during development (`first_quarter: 202302`, `last_quarter: 202405`)
- Reduce stock universe (use `min_prob_threshold` to filter)
- Disable reports during tuning: `generate_report: false`
- Use Parquet format for large datasets (faster than CSV)

**Tuning too slow?**

**Optimize by**:
- Set `validate_input_data: false` in tuning config (after validating once)
- Reduce `n_trials` during search space exploration
- Narrow search space (fewer categorical choices, tighter float ranges)
- Use `sampler: 'Random'` for initial exploration (faster than TPE)

---

### Config Traceability

Every backtest saves a copy of the configuration used:

**Location**: `backtesting_results/run_<timestamp>/config_used.yaml`

**Purpose**: Reproducibility – you can always see exactly what config produced which results.

**Tip**: If you get great results, save that `config_used.yaml` file with a descriptive name (e.g., `best_calmar_config.yaml`).

---

## Next Steps

- **Understand strategic implications**: See [STRATEGIC_OPTIONS.md](STRATEGIC_OPTIONS.md)
- **Run optimization**: See [TUNING_GUIDE.md](TUNING_GUIDE.md)
- **Interpret results**: See [OUTPUTS_AND_METRICS.md](OUTPUTS_AND_METRICS.md)
- **Get started**: Back to [USER_GUIDE.md](USER_GUIDE.md)

---

*Last updated: February 2026*
