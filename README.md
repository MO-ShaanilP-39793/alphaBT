# alphaBT

Quarterly Portfolio Backtesting Framework for Indian Equities.

Backtest ML-based stock selection strategies — select stocks each quarter using probability scores, simulate trades with configurable TP/SL exits, benchmark against market indices, optimize parameters with Optuna, and generate professional reports.

## Quick Start

```bash
pip install -r requirements.txt
```

1. Place your ML inference data, price data, and index data in `data/`.
2. Copy the config template and edit it:
   ```bash
   cp src/strategy_config_template.yaml src/strategy_config.yaml
   ```
3. Run a backtest:
   ```bash
   cd src
   python backtest_strategy.py
   ```

Results are written to `backtesting_results/`.

## Documentation

See [docs/README.md](docs/README.md) for the full documentation index — guides, configuration reference, tuning, output metrics, module references, and more.

*Last updated: March 2026*
