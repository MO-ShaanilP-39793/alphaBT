# Architecture

> System overview, module inventory, data flows, and pipeline structure for the CCQPF backtesting framework.

---

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Configuration  │────▶│  Stock Selection │────▶│  TP/SL Trades  │────▶│  Simulation  │────▶│  Reporting   │
│  (YAML + defaults)    │  (selection/)    │     │  (backtest/    │     │  (backtest/  │     │  (reporting/)│
│                 │     │                  │     │   tpsl.py)     │     │  simulation) │     │              │
└─────────────────┘     └──────────────────┘     └────────────────┘     └──────────────┘     └──────────────┘
        ▲                       ▲                        ▲                      ▲                    │
        │                       │                        │                      │                    ▼
   strategy_config.yaml    input_data.csv          price_data.parquet     price_data.parquet   backtest_report.xlsx
                           (ML scores)             index_data.csv
```

### Entry Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `backtest_strategy.py` | Run a full backtest | `python backtest_strategy.py` |
| `run_tuning.py` | Optuna hyperparameter optimization | `python run_tuning.py --n-trials 200` |
| `export_config.py` | Export trial configs from Optuna DB | `python export_config.py --study-folder ... --best` |

All scripts must be run from the `src/` directory — paths in configs are relative to it.

---

## Module Inventory

### Top-Level Scripts (`src/`)

| File | Description |
|------|-------------|
| `backtest_strategy.py` | Orchestrator. `run_backtest()` = CLI with logging; `backtest_core()` = headless (used by tuning) |
| `run_tuning.py` | Optuna orchestrator with `DataCache` singleton, single/multi-objective support |
| `export_config.py` | CLI tool to extract trial configs (`--best`, `--trial N`, `--pareto`, `--top N`) |

### `backtest/` — Trade Simulation Engine

| File | Description |
|------|-------------|
| `tpsl.py` | Core TP/SL trade simulation. 4 modes (fixed/flat/atr/pivot), independent TP & SL mode selection, index-guided exits |
| `simulation.py` | Position sizing (`calculate_position_sizes`), daily equity curves (`compute_pf_value_over_quarters`), portfolio vs index comparison |
| `dynamic_levels.py` | ATR calculation, pivot point calculation, index volatility & regime detection functions |
| `regime_dates.py` | Hardcoded crisis regimes (GFC, Covid, etc.) and market regimes (bull/bear/recovery, 2008–2025) |
| `__init__.py` | Re-exports: `simulate_trades`, `compute_pf_value_over_quarters`, `compute_pf_vs_index`, `CRISIS_REGIMES`, `MARKET_REGIMES` |

### `selection/` — Stock Selection & Weighting

| File | Description |
|------|-------------|
| `stock_selection.py` | `select_top_k_stocks()`, `select_and_weight_stocks_volatility()`, `select_and_weight_stocks_mcap()`, plus tradeable-stock filtering & price data validation |
| `__init__.py` | Re-exports all selection functions |

### `reporting/` — Report Generation

| File | Description |
|------|-------------|
| `backtest_report.py` | `generate_backtest_report()` — produces the consolidated multi-sheet Excel workbook with metrics, charts, and raw data |
| `__init__.py` | Re-exports `generate_backtest_report`, `compute_trailing_returns` |

### `config/` — Configuration & Defaults

| File | Description |
|------|-------------|
| `defaults.py` | Single source of truth for all `DEFAULT_*` constants: capital (₹100 Cr), risk-free rate (6.5%), TP/SL mode defaults, selection defaults, analytics constants, tuning defaults |
| `__init__.py` | Re-exports everything from `defaults.py` |

### `tuning/` — Optuna Utilities

| File | Description |
|------|-------------|
| `utils.py` | `load_tuning_config()`, `sample_parameters()` (conditional sampling), `build_config()` (sampled → backtest config), objective computation (`compute_cagr`, `compute_max_drawdown`, `compute_calmar_ratio`), `export_best_config()` |
| `__init__.py` | Re-exports all tuning functions + `SUPPORTED_OBJECTIVES` |

### `target/` — ML Target Analysis

| File | Description |
|------|-------------|
| `analyse_target.py` | Analyzes ML training labels: category distributions, per-stock analysis, temporal trends |
| `target.ipynb` | Notebook companion for target analysis |

### `data_processing/` — Data Preparation

| File | Description |
|------|-------------|
| `downside_deviation.ipynb` | Downside deviation analysis notebook |
| `test.ipynb` | Exploration notebook |

---

## Pipeline Detail

### `backtest_core()` — The Headless Pipeline

This is the core function called by both `run_backtest()` (CLI) and Optuna trials:

```
backtest_core(config, input_data, price_data, index_data)
  │
  ├── [1] Filter input data to configured quarter range
  │
  ├── [2] Stock Selection (one of):
  │     ├── IF run_stock_selection = True:
  │     │     ├── filter_tradeable_stocks()  — validate price coverage, remove untradeable
  │     │     └── run_stock_selection() →
  │     │           ├─ select_top_k_stocks()                    [top_k mode]
  │     │           ├─ select_and_weight_stocks_volatility()    [category_based, volatility]
  │     │           └─ select_and_weight_stocks_mcap()          [category_based, mcap]
  │     └── IF run_stock_selection = False:
  │           ├── validate_price_data_coverage()
  │           └── validate_preselected_input()  — rename category→cat, validate weights
  │
  ├── [3] simulate_trades(selected_stocks, price_data, ...)
  │     └── For each stock-quarter row: process_trade()
  │           ├── get_date_params()              — quarter → entry/exit dates
  │           ├── calculate_dynamic_thresholds() — dispatches to mode-specific calc
  │           └── Daily monitoring: SL → TP → regime exit → time exit
  │
  └── [4] compute_pf_value_over_quarters(trade_results, price_data, ...)
        └── For each quarter:
              ├── calculate_position_sizes()       — capital allocation per stock
              └── generate_quarter_equity_curve()   — daily mark-to-market
        Quarter N ending capital → Quarter N+1 starting capital
  
  Returns: {daily_pf_values, trade_results, first_quarter, last_quarter, data_issues}
```

### `run_backtest()` — CLI Wrapper

Wraps `backtest_core()` with:
1. Config loading from YAML
2. Data loading from file paths
3. `TeeOutput` — captures stdout to buffer for `backtest_log.txt`
4. Output directory creation (`backtesting_results/run_YYYYMMDD_HHMMSS/`)
5. `compute_pf_vs_index()` — benchmark comparison DataFrame
6. `generate_backtest_report()` — Excel workbook
7. Saves `config_used.yaml` and `backtest_log.txt`

---

## Data Flows

### A. Input → Stock Selection

**Input DataFrame:**

| Column | Type | Description |
|--------|------|-------------|
| `quarter` | int (YYYYMM) | Quarter code (02/05/08/11) |
| `co_name` | str | Company name |
| `prob` | float | ML model probability score |
| `volatility` | float | Stock volatility (for risk-adjusted ranking / volatility categories) |
| `category` | str | Optional mcap label (largecap/midcap/smallcap) |

**Selection output depends on mode:**

| Mode | Output Schema |
|------|---------------|
| `category_based` + `use_category_weights` | `[quarter, co_name, cat, cat_weight]` |
| `category_based` + `equal` | `[quarter, co_name, cat, stock_weight]` |
| `top_k` | `[quarter, co_name, stock_weight]` + optional `cat` |

### B. Selection → Trade Simulation

`simulate_trades()` receives:
- Selected stocks DataFrame (from above)
- Price data: `[date, co_name, open, high, low, close]`
- Optional index data: `[date, value]`
- TP/SL config parameters (mode, thresholds, ATR/pivot configs, index exit settings)

**Output:** Original DataFrame augmented with `[entry_date, entry_price, exit_date, exit_price, exit_type, tp_price, sl_price, tp_pct, sl_pct, stock_return, ...]`.

### C. Trade Results → Simulation

`compute_pf_value_over_quarters()` receives:
- Trade results DataFrame
- Price data
- Quarter range, initial capital (₹100 Cr)

For each quarter: allocates capital via weights → tracks daily mark-to-market → chains quarters.

**Output:** Equity curve DataFrame: `[date, Total_Portfolio_Value, quarter]`.

### D. Simulation → Reporting

`generate_backtest_report()` receives:
- `daily_pf`: equity curve
- `trade_results`: full trade data
- `comparison_df`: portfolio vs index (from `compute_pf_vs_index`)
- `data_issues`: optional validation findings

**Output:** `backtest_report.xlsx` with up to 17 sheets (see [REPORT_SHEETS.md](REPORT_SHEETS.md)).

---

## Tuning Architecture

### DataCache Singleton

`DataCache` loads data files **once** and reuses them across all Optuna trials:

```python
class DataCache:
    _instance = None
    
    def load(self, fixed_config):
        if self.loaded:
            return
        self.input_data = load(...)
        self.price_data = load(...)
        self.index_data = load(...)
        self.loaded = True
```

Optionally, `filter_tradeable_stocks()` is run on `data_cache.input_data` once (pre-validates instead of per-trial).

### Trial Flow

```
run_optimization()
  ├── DataCache.load()              — one-time data loading
  ├── filter_tradeable_stocks()     — one-time stock filtering
  └── study.optimize(objective)     — Optuna loop
        └── Per trial:
              ├── sample_parameters()    — conditional sampling from search_space
              ├── build_config()         — merge fixed + sampled → full config
              ├── backtest_core()        — run headless backtest
              └── compute_objective()    — extract metric (CAGR, Calmar, etc.)
```

### Conditional Sampling

`sample_parameters()` in `tuning/utils.py` implements conditional search spaces:
- TP/SL mode-specific parameters are only sampled when that mode is active
- ATR parameters sampled only when `tp_mode='atr'` or `sl_mode='atr'`
- Pivot parameters sampled only for pivot mode
- Index exit parameters sampled only when enabled

See [TUNING_GUIDE.md](TUNING_GUIDE.md) for configuration details.

---

## Configuration System

### Loading

1. `strategy_config.yaml` → `yaml.safe_load()` → dict
2. Each module extracts values with `config.get('key', DEFAULT_VALUE)`
3. All defaults live in `config/defaults.py`

### Key Defaults

| Constant | Value | Description |
|----------|-------|-------------|
| `INITIAL_CAPITAL` | 1,000,000,000 | ₹100 Crores |
| `RISK_FREE_RATE` | 0.065 | 6.5% annual |
| `TRADING_DAYS_PER_YEAR` | 252 | Standard for India |
| `DEFAULT_TP_MODE` | `'fixed'` | Default TP calculation mode |
| `DEFAULT_SL_MODE` | `'fixed'` | Default SL calculation mode |
| `DEFAULT_ENTRY_PRICE_WINDOW` | 3 | Trading days averaged for entry |
| `DEFAULT_SELECTION_TYPE` | `'category_based'` | Default stock selection method |
| `DEFAULT_CATEGORY_SCHEME` | `'volatility'` | Default category grouping |
| `DEFAULT_GENERATE_REPORT` | `True` | Generate Excel report by default |

See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for the full parameter reference.

---

## Output Structure

Each backtest run produces a timestamped directory:

```
backtesting_results/run_YYYYMMDD_HHMMSS/
├── config_used.yaml         # Frozen copy of the config
├── backtest_log.txt         # Console output captured by TeeOutput
└── backtest_report.xlsx     # Consolidated report (up to 17 sheets)
```

See [OUTPUTS_AND_METRICS.md](OUTPUTS_AND_METRICS.md) for details on each output file and metric definitions.

---

## Import Conventions

- All top-level imports use **package imports**: `from backtest import simulate_trades`
- No relative imports at top level
- Each package's `__init__.py` re-exports public functions
- Column rename: `category` → `cat` happens during pipeline (in selection or pre-validation)
