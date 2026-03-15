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
| [EXIT_STRATEGY_TUNING_METHODOLOGY.md](EXIT_STRATEGY_TUNING_METHODOLOGY.md) | ATR exit tuning — optimization windows, stability analysis, overfitting, and deployment cadence |
| [EXPORT_CONFIG.md](EXPORT_CONFIG.md) | CLI reference for extracting optimized configs from Optuna studies |

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
| [TPSL_MODULE.md](TPSL_MODULE.md) | `backtest/tpsl.py` — trade simulation, TP/SL modes, API reference |
| [DYNAMIC_LEVELS_REFERENCE.md](DYNAMIC_LEVELS_REFERENCE.md) | `backtest/dynamic_levels.py` — ATR, pivot points, index functions |
| [MODEL_TRAINING_TARGETS.md](MODEL_TRAINING_TARGETS.md) | ML target creation methodology |

## Discussion Papers

| Document | Description |
|----------|-------------|
| [ENTRY_PRICE_WINDOW.md](ENTRY_PRICE_WINDOW.md) | How the entry price window works, link to target construction, and rationale for splitting entry across three days |
| [QUARTERLY_RESULTS_AND_SIGNAL_TIMING.md](QUARTERLY_RESULTS_AND_SIGNAL_TIMING.md) | Signal timing, SEBI deadlines, and whether quarterly results are priced in by entry |

## Deep Dives

| Document | Description |
|----------|-------------|
| [CASH_APPRECIATION.md](CASH_APPRECIATION.md) | Practitioner's guide to realising the cash appreciation assumption — instruments, operations, and gap analysis |
| [ATR.md](ATR.md) | Average True Range explained — definition, calculation, and how it drives TP/SL thresholds |
| [PIVOT_POINTS.md](PIVOT_POINTS.md) | Pivot points explained — support/resistance formulas, examples, and TP/SL usage |

## Proposals

| Document | Description |
|----------|-------------|
| [ATR_POTENTIAL_IMPROVEMENTS.md](ATR_POTENTIAL_IMPROVEMENTS.md) | Trailing stop loss proposal (**not implemented**) |

*Last updated: March 2026*
