# Documentation Index

> alphaBT — Quarterly Portfolio Backtesting Framework for Indian Equities

---

## Getting Started

| Document | Description |
|----------|-------------|
| [USER_GUIDE.md](USER_GUIDE.md) | First-time setup, running a backtest, reading results |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System overview, module inventory, pipeline & data flows |

## Configuration

| Document | Description |
|----------|-------------|
| [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) | Complete YAML parameter reference for `strategy_config.yaml` |
| [STRATEGIC_OPTIONS.md](STRATEGIC_OPTIONS.md) | Strategy selection guidance — choosing modes, categories, and TP/SL approaches |

## Portfolio Selection

| Document | Description |
|----------|-------------|
| `get_portfolio.py` | Forward-looking stock selection — outputs a portfolio CSV without backtesting |

## Tuning & Optimization

| Document | Description |
|----------|-------------|
| [TUNING_GUIDE.md](TUNING_GUIDE.md) | Setting up and running Optuna hyperparameter optimization |
| [export_config.md](export_config.md) | CLI reference for extracting optimized configs from Optuna studies |

## Outputs & Reporting

| Document | Description |
|----------|-------------|
| [OUTPUTS_AND_METRICS.md](OUTPUTS_AND_METRICS.md) | Output file overview, metric definitions, formula reference |
| [REPORT_SHEETS.md](REPORT_SHEETS.md) | Detailed guide to every sheet in `backtest_report.xlsx` |

## Interactive Dashboard

| Document | Description |
|----------|-------------|
| `visualization/dashboard.py` | Plotly + Streamlit interactive drill-down dashboard — portfolio vs index, TP/SL outcomes, per-stock candlestick charts, P&L waterfall, and more. Launch with `streamlit run src/visualization/dashboard.py -- --run-dir <run_dir>` or set `launch_dashboard: true` in config. |

## Module References

| Document | Description |
|----------|-------------|
| [tpsl_module.md](tpsl_module.md) | `backtest/tpsl.py` — trade simulation, TP/SL modes, API reference |
| [dynamic_levels_reference.md](dynamic_levels_reference.md) | `backtest/dynamic_levels.py` — ATR, pivot points, index functions |
| [model_training_targets.md](model_training_targets.md) | ML target creation methodology |

## Discussion Papers

| Document | Description |
|----------|-------------|
| [ENTRY_PRICE_WINDOW.md](ENTRY_PRICE_WINDOW.md) | How the entry price window works, link to target construction, and rationale for splitting entry across three days |
| [QUARTERLY_RESULTS_AND_SIGNAL_TIMING.md](QUARTERLY_RESULTS_AND_SIGNAL_TIMING.md) | Signal timing, SEBI deadlines, and whether quarterly results are priced in by entry |

## Deep Dives

| Document | Description |
|----------|-------------|
| [CASH_APPRECIATION.md](CASH_APPRECIATION.md) | Practitioner's guide to realising the cash appreciation assumption — instruments, operations, and gap analysis |

## Proposals

| Document | Description |
|----------|-------------|
| [ATR_potential_improvements.md](ATR_potential_improvements.md) | Trailing stop loss proposal (**not implemented**) |
