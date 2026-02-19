# Export Config Tool

> CLI reference for `export_config.py` — extract optimized backtest configurations from completed Optuna studies.

---

## Overview

After running `run_tuning.py`, use `export_config.py` to extract trial configurations as ready-to-use `strategy_config.yaml` files. The tool reads from the Optuna study database and reconstructs complete backtest configs from trial parameters.

---

## Usage

All commands must be run from the `src/` directory.

### Single-Objective: Export the Best Trial

```bash
python export_config.py --study-folder tuning_logs/my_study --best
```

Exports the trial with the best objective value to `tuning_logs/my_study/best_config.yaml`.

### Export a Specific Trial

```bash
python export_config.py --study-folder tuning_logs/my_study --trial 42
```

Exports trial 42 to `tuning_logs/my_study/trial_42_config.yaml`.

### Multi-Objective: Export Pareto-Optimal Configs

```bash
python export_config.py --study-folder tuning_logs/my_study --pareto
```

Exports all non-dominated solutions to `tuning_logs/my_study/pareto_configs/trial_N.yaml`.

### Multi-Objective: Export Top N Pareto Configs

```bash
python export_config.py --study-folder tuning_logs/my_study --pareto --top 5
```

Exports the 5 best Pareto-optimal trials (sorted by first objective).

---

## CLI Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--study-folder` | `-s` | Yes | Path to the study folder (e.g., `tuning_logs/my_study`) |
| `--best` | `-b` | One of `--best`, `--trial`, `--pareto` | Export the best trial (single-objective only) |
| `--trial N` | `-t N` | One of the above | Export a specific trial by number |
| `--pareto` | `-p` | One of the above | Export Pareto-optimal trials (multi-objective only) |
| `--top N` | — | No | Used with `--pareto` — limit to top N configs |

`--best`, `--trial`, and `--pareto` are mutually exclusive.

---

## Study Folder Structure

The tool expects a study folder containing:

```
tuning_logs/my_study/
├── optuna_study.db          # Optuna SQLite database (created by run_tuning.py)
└── tuning_config_used.yaml  # Frozen copy of tuning config (created by run_tuning.py)
```

After export, the folder will contain:

```
tuning_logs/my_study/
├── optuna_study.db
├── tuning_config_used.yaml
├── best_config.yaml              # From --best
├── trial_42_config.yaml          # From --trial 42
└── pareto_configs/               # From --pareto
    ├── trial_5.yaml
    ├── trial_12.yaml
    └── trial_87.yaml
```

---

## Output Format

Each exported YAML file includes:
1. **Header comments** with trial metadata (trial number, objective values, CAGR, MDD, total return, win rate, number of trades)
2. **Complete backtest config** compatible with `backtest_strategy.py`
3. `generate_report: true` is automatically set

Example header:
```yaml
# Exported from Optuna trial 42
# calmar: 1.234567
# CAGR: 18.50%
# MDD: -15.00%
# Total Return: 245.30%
# Win Rate: 62.50%
# Trades: 480
#

<full config follows>
```

---

## Workflow

```
1. Run tuning          →  python run_tuning.py --n-trials 200
2. (Optional) Review   →  optuna-dashboard sqlite:///tuning_logs/my_study/optuna_study.db
3. Export best config   →  python export_config.py -s tuning_logs/my_study --best
4. Copy to active       →  copy tuning_logs/my_study/best_config.yaml strategy_config.yaml
5. Run full backtest    →  python backtest_strategy.py
```

---

## Related Documentation

- [TUNING_GUIDE.md](TUNING_GUIDE.md) — Setting up and running Optuna optimization
- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) — Full YAML parameter reference
