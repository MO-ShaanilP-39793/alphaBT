# Tuning Guide

Comprehensive guide to strategy optimization using Optuna.

---

## Table of Contents

- [Why Optimize?](#why-optimize)
- [Optimization Workflow](#optimization-workflow)
- [Setting Up tuning_config.yaml](#setting-up-tuning_configyaml)
- [Running Optimization](#running-optimization)
- [Analyzing Results](#analyzing-results)
- [Multi-Objective Optimization](#multi-objective-optimization)
- [Exporting Configs](#exporting-configs)
- [Advanced Tuning Strategies](#advanced-tuning-strategies)
- [From Tuning to Production](#from-tuning-to-production)
- [Troubleshooting](#troubleshooting)

---

## Why Optimize?

### The Parameter Space Problem

A typical backtest has **dozens of parameters**:

- **Stock selection**: k (20-50), category counts (many combinations), selection method (2 options), min threshold (0.4-0.8)
- **TP/SL thresholds**: Tiered (3 categories × 2 values = 6 params), Flat (2 params), ATR (3 params), Pivot (3 params)
- **Index exits**: MA period (10-50), thresholds (-0.01 to -0.05), multipliers (0.5-2.0)

**Manual exploration**: Testing even 5 values per parameter across 10 parameters = **9,765,625 combinations**. Infeasible.

**Optuna solution**: Intelligent sampling that explores high-potential regions, learns from past trials, and finds optimal combinations in 100-500 trials (not millions).

---

### When to Use Optimization

**✅ Use tuning when**:
- You don't know optimal TP/SL thresholds for your data
- You want to compare multiple strategies systematically (e.g., ATR vs pivots)
- You're building a production strategy and need confidence in parameters
- You have computational time (1-8 hours depending on setup)

**❌ Manual testing is fine when**:
- You have strong priors (e.g., "all large-caps should use 5% TP/SL")
- You're doing quick exploratory analysis
- Your parameter space is very small (< 20 combinations)
- You just want to understand what your ML model does

**Best practice**: Start with 1-2 manual backtests to understand data, then optimize once you have a baseline.

---

## Optimization Workflow

### High-Level Process

```
1. Define search space in tuning_config.yaml
   ↓
2. Choose objective (Calmar, CAGR, or MDD)
   ↓
3. Run: python run_tuning.py --n-trials 100
   ↓
4. Optuna samples parameters and evaluates
   ├─ Trial 1: Test combination A → Calmar = 1.5
   ├─ Trial 2: Test combination B → Calmar = 2.1 ✓ (better)
   ├─ Trial 3: Test combination C → Calmar = 1.8
   └─ ... (learns which regions work)
   ↓
5. View results in optuna-dashboard
   ↓
6. Export desired config with export_config.py
   ↓
7. Copy exported config → strategy_config.yaml
   ↓
8. Run final backtest with detailed reports
```

### Time Estimation

| Configuration | Trials | Est. Time | Use Case |
|---------------|--------|-----------|----------|
| Narrow search (2-3 params) | 50 | 15-30 min | Quick refinement |
| Medium search (5-8 params) | 100 | 30-90 min | Standard optimization |
| Wide search (10+ params) | 200-500 | 2-8 hours | Comprehensive tuning |

**Factors affecting speed**:
- Data size (more quarters = slower)
- Number of stocks per quarter
- TP/SL complexity (ATR/pivot slower than flat)
- Report generation (disable for tuning: `generate_report: false`)

---

## Setting Up tuning_config.yaml

### Three-Part Structure

```yaml
1. fixed:           # Parameters that DON'T change
2. search_space:    # Parameters Optuna DOES sample
3. optuna:          # Optimization settings
```

### Part 1: Fixed Parameters

**Purpose**: Define constants that apply to all trials.

```yaml
fixed:
  # Data (almost always fixed)
  input_data_path: "../data/inference_data.csv"
  price_data_path: "../data/price_data/ohlcv.parquet"
  index_data_path: "../data/index_data/nifty500.csv"
  
  # Backtest period (fixed for consistent comparison)
  first_quarter: 202002
  last_quarter: 202411
  
  # Core behavior
  run_stock_selection: true
  
  # Speed optimizations
  generate_report: false             # Disable reports during tuning
  validate_input_data: false        # Set true for first run, then false
```

**What to fix**:
- ✅ Data paths (don't change during optimization)
- ✅ Date ranges (need consistent evaluation period)
- ✅ Report flags (disable for speed)
- ✅ Any parameter you have strong conviction about

**What NOT to fix**:
- Parameters you're unsure about (let Optuna explore)
- Parameters you want to compare (e.g., testing top-k vs category-based)

---

### Part 2: Search Space

**Purpose**: Define what Optuna samples and how.

#### Search Space Architecture

**Two types of parameters**:

1. **Global search space**: Always sampled
2. **Conditional search space**: Only sampled if certain conditions met

```yaml
search_space:
  # Global - always sampled
  selection_type:
    type: categorical
    choices: ['category_based', 'top_k']
  
  tp_enabled:
    type: categorical
    choices: [true, false]
  
  tpsl_mode:
    type: categorical
    choices: ['flat', 'atr']
  
  independent_tpsl_modes:        # Optional: when true, tp_mode and sl_mode are sampled independently
    type: categorical
    choices: [true, false]

# Conditional - only if selection_type == 'top_k'
top_k:
  k:
    type: int
    low: 20
    high: 50

# Conditional - only if tpsl_mode == 'flat'
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
```

**How it works**:
- Trial samples `selection_type: 'top_k'` → `k` is sampled from `top_k` section
- Trial samples `selection_type: 'category_based'` → `k` is NOT sampled (not relevant)
- This prevents invalid configurations (e.g., sampling ATR period when using tiered TP/SL)

**Note**: The `tpsl_mode` search space key controls which mode-specific params are sampled. At runtime, the sampled mode is assigned to `tp_mode` and `sl_mode`. If `independent_tpsl_modes: true`, they are sampled separately.

#### Parameter Types

**Categorical** (discrete options):
```yaml
tpsl_mode:
  type: categorical
  choices: ['tiered', 'flat', 'atr', 'pivot']
```
- Use for: Strings, booleans, specific discrete values
- Optuna samples uniformly from choices
- **Note**: `tpsl_mode` in the search space maps to `tp_mode`/`sl_mode` in the runtime config

**Integer** (whole numbers):
```yaml
atr_period:
  type: int
  low: 7        # Inclusive minimum
  high: 21      # Inclusive maximum
```
- Use for: Counts, periods, discrete thresholds
- Optuna samples intelligently (TPE learns which ranges work)

**Float** (continuous or discretized):
```yaml
flat_tp:
  type: float
  low: 0.02
  high: 0.15
  step: 0.01   # Optional: round to 0.02, 0.03, 0.04, ... 0.15
```
- Use for: Percentages, multipliers, thresholds
- With `step`: Discretizes (good for percentages like TP/SL)
- Without `step`: Continuous sampling

---

### Part 3: Optuna Settings

```yaml
optuna:
  study_name: "my_optimization_v1"
  objective: "calmar"          # 'calmar', 'cagr', or 'mdd'
  n_trials: 100
  sampler: "TPE"               # 'TPE', 'Random', or 'CmaEs'
  direction: "maximize"        # 'maximize' or 'minimize'
  load_if_exists: true
  calmar_cap: 10.0
  failure_penalty: -999.0
```

#### Objective Selection

**Calmar Ratio** (`objective: 'calmar'`):
```
Calmar = CAGR / |Max Drawdown|
```
- **Use when**: You want risk-adjusted returns
- **Optimizes for**: Growth per unit of drawdown risk
- **Best for**: Institutional strategies, conservative investors
- **Direction**: `maximize`
- **Typical values**: 1.0-5.0 (>2.0 is excellent)

**CAGR** (`objective: 'cagr'`):
```
CAGR = (final_value / initial_value)^(1/years) - 1
```
- **Use when**: Pure return maximization (ignore risk)
- **Optimizes for**: Highest growth rate
- **Best for**: Aggressive strategies, high-risk tolerance
- **Direction**: `maximize`
- **Typical values**: 0.10-0.50 (10%-50% annual growth)

**Max Drawdown** (`objective: 'mdd'`):
```
MDD = max((peak - trough) / peak)
```
- **Use when**: Risk minimization (capital preservation)
- **Optimizes for**: Smallest peak-to-trough decline
- **Best for**: Conservative strategies, downside protection
- **Direction**: `minimize` ⚠️
- **Typical values**: 0.10-0.40 (10%-40% worst decline)

**Which to choose?**

| Your Priority | Objective | Why |
|---------------|-----------|-----|
| **Balanced** | `calmar` | Best all-around: growth + risk management |
| **Maximum growth** | `cagr` | Ignore drawdowns, chase returns |
| **Capital preservation** | `mdd` | Avoid large losses, steady growth |
| **Sharpe-like** | `calmar` | Calmar ≈ Sharpe for drawdown risk |

**Default recommendation**: Start with `calmar` (most balanced).

#### Sampler Algorithms

**TPE (Tree-structured Parzen Estimator)** (`sampler: 'TPE'`):
- **Default choice** – works well for most use cases
- **How it works**: Learns which parameter regions produce good results, samples more from those regions
- **Best for**: Medium-to-large search spaces (5-20 parameters)
- **Speed**: Moderate (learning overhead)

**Random** (`sampler: 'Random'`):
- Pure random sampling (no learning)
- **Best for**: Small search spaces, baseline comparison, debugging
- **Speed**: Fastest (no overhead)
- **Use when**: <50 trials or want unbiased exploration

**CmaEs** (`sampler: 'CmaEs'`):
- Evolution Strategy algorithm
- **Best for**: Continuous parameter spaces (mostly floats, few categoricals)
- **Speed**: Moderate
- **Use when**: Tuning numerical parameters like ATR multipliers, TP/SL percentages

**Recommendation**: Use `TPE` unless you have a specific reason not to.

#### Other Settings

**study_name**: Unique identifier for this optimization run
- Results stored in `tuning_logs/<study_name>/optuna_study.db`
- Use descriptive names: `"atr_optimization_2024"`, `"top_k_vs_category_feb2026"`
- If `load_if_exists: true`, resumes existing study with same name

**n_trials**: Number of parameter combinations to test
- Can override via CLI: `python run_tuning.py --n-trials 200`
- **Start with**: 50-100 for exploration
- **Production**: 200-500 for final optimization

**load_if_exists**: 
- `true`: Resume existing study, add more trials
- `false`: Crash if study exists (prevents accidents)
- **Tip**: Use `--fresh` flag to delete + restart: `python run_tuning.py --fresh`

**calmar_cap**: Maximum Calmar ratio value (prevents inf when MDD ≈ 0)
- **Default**: 10.0
- **Effect**: `Calmar = min(CAGR / |MDD|, calmar_cap)`
- **Reason**: Some configs have tiny drawdowns → inflated Calmar → misleading

**failure_penalty**: Value returned if trial crashes
- **For maximize objectives**: Use large negative (e.g., -999.0)
- **For minimize objectives**: Use large positive (e.g., 999.0)
- **Prevents**: Crash propagation; Optuna learns to avoid failing configs

---

## Running Optimization

### Basic Usage

```bash
cd src
python run_tuning.py
```

**Uses**: `tuning_config.yaml` in `src/` directory

**Output**:
```
Starting tuning with config: tuning_config.yaml
Study name: backtest_optimization
Objective: calmar (maximize)
Number of trials: 100

[I 2026-02-11 10:00:00] Trial 0 finished with value: 1.85 ...
[I 2026-02-11 10:02:15] Trial 1 finished with value: 2.13 ...
[I 2026-02-11 10:04:30] Trial 2 finished with value: 1.92 ...
...
[I 2026-02-11 12:30:00] Trial 99 finished with value: 3.41 ...

Best trial: 99
Best value (Calmar): 3.41

Use export_config.py to export trial configurations.
```

### Command-Line Options

#### Use Custom Config

```bash
python run_tuning.py --config my_custom_tuning.yaml
```

Useful for running multiple optimization experiments with different configs.

#### Override Trial Count

```bash
python run_tuning.py --n-trials 200
```

Overrides `n_trials` in YAML config. Useful for:
- Quick tests: `--n-trials 10`
- Deep optimization: `--n-trials 500`

#### Fresh Start (Delete Existing Study)

```bash
python run_tuning.py --fresh
```

**Effect**: Deletes existing study with same name, starts from scratch.

**Use when**: You changed search space significantly and don't want old trials.

**Alternative**: Change `study_name` in config.

#### Combine Flags

```bash
python run_tuning.py --config aggressive.yaml --n-trials 300 --fresh
```

---

### Monitoring Progress

Optimization prints live updates:

```
[I 2026-02-11 10:04:30] Trial 2 finished with value: 1.92 and parameters:
  {'selection_type': 'top_k', 'k': 35, 'tp_enabled': True, 
   'tp_mode': 'flat', 'sl_mode': 'flat', 'flat_tp': 0.12, 'flat_sl': 0.06, ...}
Duration: 2.3 minutes
```

**Interpreting**:
- **value**: Objective function result (Calmar, CAGR, or MDD)
- **parameters**: Sampled configuration for this trial
- **Duration**: Time taken (helps estimate total runtime)

**Best value so far** is tracked automatically (updated when a better trial finishes).

---

### Output Files

After tuning completes:

| File | Location | Description |
|------|----------|-------------|
| `optuna_study.db` | `tuning_logs/<study_name>/` | SQLite database with all trials for this study |
| `tuning_config_used.yaml` | `tuning_logs/<study_name>/` | Copy of the tuning config used |
| `data_quality_issues.csv` | `tuning_logs/<study_name>/` | (If validation found issues) Data coverage report |

**Note**: Unlike previous versions, `best_config.yaml` is **not** auto-exported. Use the `export_config.py` script to export trial configurations (see [Exporting Configs](#exporting-configs) section).

**optuna_study.db**:
- Contains all trials for this specific study
- Per-study database (isolated from other studies)
- Persistent (survives restarts)
- Query via Optuna Dashboard: `optuna-dashboard sqlite:///tuning_logs/<study_name>/optuna_study.db`

**tuning_config_used.yaml**:
- Exact copy of the config used for this optimization run
- Useful for reproducing results or understanding what was optimized

---

## Analyzing Results

### Optuna Dashboard (Recommended)

**Start dashboard**:
```bash
optuna-dashboard sqlite:///tuning_logs/<study_name>/optuna_study.db
```

**Access**: Open browser to `http://localhost:8080`

**Tip**: Replace `<study_name>` with your actual study name (e.g., `cagr_vs_mdd_optimization`).

**Features**:

#### 1. Study List
- View all optimization studies in database
- See trial counts, best values, date ranges

#### 2. Optimization History
- Line plot: objective value vs trial number
- **What to look for**:
  - Early trials: random exploration
  - Later trials: convergence (values plateau)
  - **If still improving at end**: Consider running more trials

#### 3. Parallel Coordinate Plot
- **Most useful visualization**
- Shows relationship between parameters and objective
- **How to read**:
  - Each vertical axis = one parameter
  - Each line = one trial
  - Color = objective value (blue = low, red = high)
  - **Patterns**: Lines converging at top = good parameter region

**Example insights**:
- "All best trials have `flat_tp` between 0.08-0.12" → Narrow that range
- "`selection_type: 'top_k'` always outperforms `'category_based'`" → Fix to top-k
- "`atr_period` doesn't matter much" → Remove from search space

#### 4. Parameter Importances
- Bar chart: which parameters most affect objective
- **Based on**: Variance explained by each parameter
- **Use to**: Identify which params to refine vs which are noise

**Example**:
```
flat_tp:        ████████████ 45%  (very important)
flat_sl:        ████████ 30%      (important)
k:              ███ 15%           (moderate)
sort_by: █ 5%            (doesn't matter much)
```

→ Focus tuning efforts on `flat_tp` and `flat_sl`.

#### 5. Contour Plot
- 2D heatmap of objective value vs two parameters
- **Use to**: Find optimal ranges for parameter pairs
- **Example**: "Contour plot of flat_tp vs flat_sl shows ridge at tp=0.10, sl=0.05"

---

### Interpreting Best Config

**Example best_config.yaml**:
```yaml
# Best trial: 99
# Objective value: 3.41 (calmar)

input_data_path: "../data/inference_data.csv"
price_data_path: "../data/price_data/ohlcv.parquet"
# ... (all fixed params)

selection_type: 'top_k'
top_k_config:
  k: 35
  weighting_scheme: 'equal'

sort_by: 'probability'
min_prob_threshold: 0.55

tp_enabled: true
sl_enabled: true
tp_mode: 'flat'
sl_mode: 'flat'
flat_config:
  tp_pct: 0.12
  sl_pct: 0.06

# ... (rest of config)
```

**Questions to ask**:

1. **Are values at search space boundaries?**
   - If `k: 50` (max of 20-50 range) → Maybe should search 50-80?
   - If `flat_tp: 0.02` (min of 0.02-0.15 range) → Maybe lower bound is optimal?
   - **Action**: Expand search space, re-run optimization

2. **Do results make intuitive sense?**
   - TP > SL (e.g., 12% vs 6%) = asymmetric exits (common in momentum strategies)
   - High `min_prob_threshold` (0.55) = filtering low-confidence predictions
   - **If counterintuitive**: Double-check data quality or search space

3. **What's the Calmar/CAGR/MDD value?**
   - Calmar 3.41 = CAGR is 3.41× the max drawdown → excellent risk-adjusted
   - Compare to baseline (e.g., unoptimized run) to quantify improvement

---

### Validation: Avoid Overfitting

**The problem**: Optimizing on full backtest period may find parameters that work historically but fail forward.

**Solutions**:

#### 1. Out-of-Sample Testing
```yaml
# Optimization config
first_quarter: 202002
last_quarter: 202302   # Train on 2020-2023

# Final backtest config (after optimization)
first_quarter: 202305
last_quarter: 202411   # Test on 2023-2024
```

**Process**:
1. Optimize on early period
2. Apply best params to later period
3. If performance degrades significantly → overfitted

#### 2. Walk-Forward Optimization
- Optimize on rolling windows
- Example: Optimize on 2020-2021, test on 2022, optimize on 2021-2022, test on 2023, ...
- More robust but complex to implement (not built-in)

#### 3. Regularization Through Search Space Design
- Avoid very narrow parameter ranges (e.g., `low: 0.123, high: 0.127`)
- Prefer discrete steps for percentages: `step: 0.01` (not 0.001)
- Limit conditional params (too many = overfitting risk)

---

## Multi-Objective Optimization

### When to Use Multi-Objective

Single-objective optimization collapses your goals into one number (e.g., Calmar ratio = CAGR / MDD). This is convenient but hides information — you don't see the full range of trade-offs available.

**Use multi-objective when:**
- You want to see the full trade-off curve between return and risk
- You don't want to pre-commit to a specific CAGR/MDD weighting
- You want to choose between aggressive (high CAGR, higher drawdown) and conservative (lower CAGR, minimal drawdown) strategies after seeing the options
- Stakeholders have different risk preferences

**Stick with single-objective when:**
- You have a clear, fixed objective (e.g., "maximize Calmar ratio")
- You want a single "best" answer with no ambiguity
- You're doing quick exploration with few trials

### How It Works: Pareto Front

In multi-objective optimization, there is no single "best" trial. Instead, Optuna finds a set of **Pareto-optimal** (non-dominated) solutions.

A trial is **Pareto-optimal** if no other trial is better in ALL objectives simultaneously. The collection of these trials forms the **Pareto front** — a curve showing the best achievable trade-offs.

```
  CAGR ↑
   25% │            ● (aggressive: high return, high drawdown)
       │          ●
   20% │        ●      ← Pareto front (the frontier of best trade-offs)
       │      ●
   15% │    ●
       │  ● (conservative: lower return, minimal drawdown)
   10% │
       └──────────────────────── MDD →
         5%    10%    15%    20%
```

Trials **below** the Pareto front are **dominated** — there exists another trial that is better in at least one objective without being worse in any other.

### Configuration

To enable multi-objective, **replace** `objective`/`direction` with `objectives`/`directions` in the `optuna:` section of your tuning config. These keys are **mutually exclusive** — you cannot have both `objective` and `objectives` in the same config (this will raise an error).

```yaml
optuna:
  study_name: "cagr_vs_mdd_optimization"
  
  # Multi-objective: maximize CAGR while minimizing drawdown
  objectives: ['cagr', 'mdd']
  directions: ['maximize', 'minimize']
  
  # Per-objective failure penalties (must match length of objectives)
  # Use negative for maximize objectives, positive for minimize objectives
  failure_penalties: [-999.0, 999.0]
  
  n_trials: 200
  sampler: "NSGA-II"   # Recommended for multi-objective
  load_if_exists: true
  
  # calmar_cap still applies if 'calmar' is one of your objectives
  # calmar_cap: 10.0
```

**Key differences from single-objective:**

| Setting | Single-Objective | Multi-Objective |
|---------|-----------------|----------------|
| Config key | `objective: "calmar"` | `objectives: ['cagr', 'mdd']` |
| Direction | `direction: "maximize"` | `directions: ['maximize', 'minimize']` |
| Penalty | `failure_penalty: -999.0` | `failure_penalties: [-999.0, 999.0]` |
| Sampler | `TPE` (default) | `NSGA-II` (default if not specified) |
| Result | Single best trial | Pareto front of non-dominated trials |

**Supported objective combinations:**
You can combine any of `'cagr'`, `'mdd'`, `'calmar'`. Common setups:
- `['cagr', 'mdd']` with `['maximize', 'minimize']` — the classic return vs risk trade-off
- `['calmar', 'mdd']` with `['maximize', 'minimize']` — risk-adjusted return vs absolute risk

### Samplers for Multi-Objective

| Sampler | Multi-Obj Support | Notes |
|---------|:-:|-------|
| `NSGA-II` | ✅ | **Recommended.** Evolutionary algorithm designed for multi-objective. Fast, well-tested. |
| `NSGA-III` | ✅ | Extension of NSGA-II for many objectives (3+). Use if you have 3+ objectives. |
| `TPE` | ✅ | Works for multi-objective since Optuna 3.0. Decent but not specialized. |
| `Random` | ✅ | Random search. Useful as a baseline. |
| `CmaEs` | ❌ | **Does NOT support multi-objective.** Will raise an error. |

### Running Multi-Objective Optimization

The command is identical to single-objective — the mode is detected from your config:

```bash
cd src
python run_tuning.py                          # Uses tuning_config.yaml
python run_tuning.py --n-trials 200            # Override trial count
python run_tuning.py --fresh                   # Start a fresh study
```

The output will show the optimization mode and, upon completion, a summary of the Pareto front:

```
======================================================================
OPTUNA HYPERPARAMETER TUNING
======================================================================
  Mode: Multi-objective
  Objective(s): cagr (maximize) + mdd (minimize)
  Sampler: NSGA-II
  ...

======================================================================
OPTIMIZATION COMPLETE
======================================================================

Study Summary:
  Total trials: 200
  Completed: 195
  Failed: 5

Pareto Front: 23 non-dominated solutions
  CAGR (maximize): 0.0812 — 0.2341
  Maximum Drawdown (minimize): 0.0523 — 0.1892

  Top 10 Pareto-optimal trials (sorted by cagr):
  Trial |           CAGR |            MDD |      CAGR% |       MDD% | Win Rate% |    Trades
    142 |         0.2341 |         0.1892 |      23.41 |      18.92 |     58.20 |       450
    087 |         0.2105 |         0.1456 |      21.05 |      14.56 |     56.80 |       420
    ...
```

### Analyzing Multi-Objective Studies in Optuna Dashboard

The Optuna Dashboard has built-in support for multi-objective studies:

```bash
optuna-dashboard sqlite:///tuning_logs/<study_name>/optuna_study.db
```

Open the browser at `http://localhost:8080` and navigate to your study.

#### 1. Pareto Front Plot

The dashboard automatically shows a **scatter plot** of the Pareto front:
- Each axis represents one objective (e.g., X = CAGR, Y = MDD)
- Each dot is a completed trial
- **Pareto-optimal trials** are highlighted on the frontier
- Dominated trials appear behind the front

**How to read it:**
- The top-left region represents "ideal" (high CAGR, low MDD) — but may be unachievable
- The Pareto front shows what IS achievable — the best trade-offs
- Points at the extremes represent the aggressive end (max CAGR) and conservative end (min MDD)
- The **knee point** (sharpest bend in the curve) often represents the best balanced trade-off

#### 2. Parallel Coordinate Plot

Still works in multi-objective mode. Filter by Pareto-optimal trials to see which parameter regions produce the best trade-offs.

**Tip**: Color by one objective (e.g., CAGR) to see how parameters correlate with return, then switch to the other (MDD) to see the risk perspective.

#### 3. Parameter Importances

The dashboard shows parameter importances **per objective**. This is very useful:
- A parameter that is important for CAGR but not for MDD → controls return without affecting risk
- A parameter important for both → a key lever that moves the entire trade-off
- A parameter important for neither → noise, consider removing

#### 4. Trial History

For multi-objective studies, the trial history shows values for each objective over time. You can track convergence of the Pareto front as trials progress.

### Choosing a Solution from the Pareto Front

Unlike single-objective, you must **choose** which Pareto solution to use. Common strategies:

1. **Aggressive**: Pick the trial with highest CAGR (right end of Pareto front)
   - Accepts highest drawdown for maximum return
   - Suitable when capital preservation is secondary

2. **Conservative**: Pick the trial with lowest MDD (left end of Pareto front)
   - Accepts lower returns for minimal drawdown
   - Suitable for risk-averse mandates

3. **Knee Point** (recommended): Pick the trial at the "elbow" of the curve
   - Best marginal trade-off (each additional unit of risk buys the most return)
   - Visual: where the Pareto front bends most sharply
   - Examine the Pareto front plot in the dashboard and choose the point where the curve transitions from steep to flat

4. **Constraint-Based**: Pick the best CAGR trial where MDD < your threshold
   - E.g., "Give me the highest CAGR where max drawdown stays under 15%"
   - Filter Pareto configs by MDD and choose the best CAGR among those

### Comparison: Single-Objective Calmar vs Multi-Objective CAGR+MDD

| Aspect | Calmar (single-obj) | CAGR + MDD (multi-obj) |
|--------|--------------------|-----------------------|
| Output | One best trial | Set of Pareto-optimal trials |
| Trade-off | Pre-defined (CAGR/MDD ratio) | User chooses after seeing options |
| Information | Less (collapsed to ratio) | More (full frontier visible) |
| Simplicity | Simpler to interpret | Requires judgment to pick a solution |
| Risk of distortion | Tiny MDD → inflated ratio | No distortion — each metric is independent |
| When to use | Quick optimization, clear mandate | Exploration, stakeholder presentation |

**Practical recommendation**: Start with single-objective Calmar for baseline, then switch to multi-objective for refinement and stakeholder discussions where seeing the full trade-off curve adds value.

---

## Exporting Configs

After optimization, use `export_config.py` to extract trial configurations as ready-to-run YAML files.

### Single-Objective: Export Best Trial

```bash
cd src
python export_config.py --study-folder tuning_logs/my_study --best
```

This exports `best_config.yaml` into the study folder — the trial with the highest (or lowest) objective value.

### Export a Specific Trial

Works for both single and multi-objective studies:

```bash
python export_config.py --study-folder tuning_logs/my_study --trial 42
```

This exports `trial_42_config.yaml` into the study folder.

### Multi-Objective: Export Pareto Configs

```bash
# Export ALL Pareto-optimal configs
python export_config.py --study-folder tuning_logs/my_study --pareto

# Export only top 5 (sorted by first objective)
python export_config.py --study-folder tuning_logs/my_study --pareto --top 5
```

This creates a `pareto_configs/` folder inside the study folder with one YAML per Pareto trial:
```
tuning_logs/my_study/
  pareto_configs/
    trial_42.yaml    # Aggressive (highest CAGR)
    trial_87.yaml
    trial_123.yaml
    trial_156.yaml
    trial_201.yaml   # Conservative (lowest MDD)
```

Each exported YAML includes a comment header with the trial number, objective values, and key metrics.

### Using an Exported Config

```bash
# Copy to strategy_config.yaml and run
copy tuning_logs\my_study\pareto_configs\trial_87.yaml strategy_config.yaml
python backtest_strategy.py
```

---

## Advanced Tuning Strategies

### Strategy 1: Iterative Refinement

**Process**:
1. **Broad search** (50-100 trials, wide ranges)
   ```yaml
   flat_tp:
     low: 0.02
     high: 0.20
     step: 0.02   # Coarse discretization
   ```

2. **Analyze** results (dashboard)
   - Best trials have `flat_tp: 0.10-0.14`

3. **Narrow search** (100-200 trials, tight ranges)
   ```yaml
   flat_tp:
     low: 0.08
     high: 0.16
     step: 0.01   # Fine discretization
   ```

4. **Final tuning** (refine top-2 params)

**Benefit**: Efficiently explores then exploits good regions.

---

### Strategy 2: Mode Comparison Studies

**Goal**: Compare strategies (e.g., "Is ATR better than flat TP/SL?")

**Setup**:
```yaml
# Study 1: Flat TP/SL optimization
fixed:
  tp_mode: 'flat'
  sl_mode: 'flat'
search_space:
  # ... only flat_tpsl params

# Study 2: ATR optimization
fixed:
  tp_mode: 'atr'
  sl_mode: 'atr'
search_space:
  # ... only atr_tpsl params
```

**Process**:
1. Run separate studies (different `study_name`)
2. Compare best objective values
3. Choose winning mode
4. (Optional) Re-optimize winning mode with refined search space

**Why separate?** Avoids wasting trials on inferior modes.

---

### Strategy 3: Two-Stage Optimization

**Stage 1: Selection parameters**
```yaml
search_space:
  selection_type: [...]
  k: [...]
  category_counts: [...]

fixed:
  tp_mode: 'flat'
  sl_mode: 'flat'
  flat_config:
    tp_pct: 0.10   # Use reasonable defaults
    sl_pct: 0.05
```

**Stage 2: Exit parameters** (using best selection from stage 1)
```yaml
fixed:
  selection_type: 'top_k'  # Best from stage 1
  top_k_config:
    k: 35                  # Best from stage 1

search_space:
  tpsl_mode: [...]
  flat_tpsl: [...]
  atr_tpsl: [...]
```

**Benefit**: Reduces dimensionality (10 params → two 5-param searches).

---

### Strategy 4: Ensemble Analysis

**Idea**: Don't just use single best trial – look at top-10.

**Process**:
1. Run optimization (200+ trials)
2. In dashboard, filter top-10 trials
3. Look for commonalities:
   - "All have `k` between 30-40"
   - "All use `sort_by: 'probability'`"
   - "TP ranges 0.08-0.12, SL ranges 0.04-0.07"

4. Use **median** or **mode** of top-10 as final config (more robust than single best)

**Example**:
```
Top 10 trials:
  Trial 99: k=35, flat_tp=0.12, flat_sl=0.06
  Trial 87: k=33, flat_tp=0.11, flat_sl=0.05
  Trial 76: k=38, flat_tp=0.10, flat_sl=0.06
  ...

Robust config:
  k: 35 (median)
  flat_tp: 0.11 (median)
  flat_sl: 0.06 (mode)
```

---

### Strategy 5: Constraint-Based Optimization

**Goal**: Enforce business rules (e.g., "TP must be > SL")

**Optuna approach**: Use `trial.suggest_*` with validation in objective function:

```python
# In tuning code (advanced users can modify run_tuning.py)
tp = trial.suggest_float('flat_tp', 0.05, 0.20)
sl = trial.suggest_float('flat_sl', 0.02, 0.15)

if tp <= sl:
    return failure_penalty  # Invalid config
```

**Alternatively**: Use search space design:
```yaml
flat_tpsl:
  tp:
    low: 0.05   # Ensure min TP > max SL
    high: 0.20
  sl:
    low: 0.02
    high: 0.05  # Max SL < min TP
```

---

## From Tuning to Production

### Step 1: Export Best Config

Use `export_config.py` to export the best trial (single-objective) or a chosen Pareto solution (multi-objective):

```bash
cd src

# Single-objective: export best trial
python export_config.py --study-folder tuning_logs/my_study --best

# Multi-objective: export Pareto front, then choose one
python export_config.py --study-folder tuning_logs/my_study --pareto
```

This creates `best_config.yaml` (or `pareto_configs/trial_*.yaml`) in the study folder.

### Step 2: Review Exported Config

Open the exported config:

```bash
# Windows
notepad tuning_logs\my_study\best_config.yaml

# Mac/Linux
cat tuning_logs/my_study/best_config.yaml
```

**Sanity checks**:
- ✅ All parameters present
- ✅ Values make sense (no suspicious outliers)
- ✅ Fixed params match your data
- ✅ Objective values in header comment match expectations

### Step 3: Copy to Strategy Config

```bash
# Windows
copy tuning_logs\my_study\best_config.yaml strategy_config.yaml

# Mac/Linux
cp tuning_logs/my_study/best_config.yaml strategy_config.yaml
```

### Step 4: Enable Reports

Edit `strategy_config.yaml`:

```yaml
generate_report: true   # Turn on

# Optional: Add sub-periods for detailed report
report_sub_periods:
  - [2020, 2022]
  - [2023, 2025]
```

### Step 5: Run Final Backtest

```bash
cd src
python backtest_strategy.py
```

**This run**:
- Uses optimized parameters
- Generates full reports (Excel, charts)
- Suitable for stakeholder presentation

### Step 6: Document Your Work

Save a copy of the optimized config with metadata:

```bash
# Create versioned config (from the strategy_config.yaml you're using)
copy strategy_config.yaml configs/production_v1_calmar3.41_feb2026.yaml
```

**Include in filename**:
- Version number
- Objective value achieved
- Date
- Brief description

**Why**: Reproducibility and experiment tracking.

---

## Troubleshooting

### ❌ "All trials failing"

**Symptoms**: Every trial returns `failure_penalty` value.

**Causes**:
1. Data path issues (fixed params point to wrong files)
2. Invalid search space (creates impossible configs)
3. Runtime errors in backtest code

**Debug**:
1. Check `tuning_logs/<study_name>/` for error messages
2. Run single backtest manually with a config from `tuning_logs/`
3. Verify data files exist and are formatted correctly

---

### ❌ "Optimization not improving"

**Symptoms**: Objective value plateaus early, best trial is from first 10.

**Causes**:
1. Search space too narrow (already at optimum)
2. Parameters don't matter (low sensitivity)
3. Random sampler (no learning)

**Solutions**:
1. Check parameter importances in dashboard
   - If all low → parameters don't affect objective much
2. Expand search space
3. Switch to TPE sampler if using Random

---

### ❌ "Best config has params at boundaries"

**Symptoms**: Optimal `k=50` when range is `[20, 50]`, or `flat_tp=0.20` when range is `[0.02, 0.20]`.

**Cause**: True optimum may be outside search space.

**Solution**: Expand bounds and re-run:
```yaml
# Original
k:
  low: 20
  high: 50

# Expanded
k:
  low: 30   # Keep lower bound tighter if best was 50
  high: 80  # Expand upper
```

---

### ❌ "Tuning is too slow"

**Symptoms**: Each trial takes >5 minutes, projected 100 trials = 8+ hours.

**Speed tips**:
1. **Narrow date range** during tuning:
   ```yaml
   first_quarter: 202302  # Just recent 2 years
   last_quarter: 202411
   ```
   
2. **Disable validation**:
   ```yaml
   validate_input_data: false  # After validating once
   ```

3. **Reduce trial count for exploration**:
   ```bash
   python run_tuning.py --n-trials 50  # Quick exploration
   ```

4. **Use simpler TP/SL modes**: Flat is faster than ATR/Pivot.

5. **Disable reports** (should already be off):
   ```yaml
   generate_report: false
   ```

---

### ❌ "Study name conflict"

**Symptoms**: `"A study with name 'X' already exists"`

**Cause**: `load_if_exists: false` but study exists in database.

**Solutions**:
1. **Resume existing study**: Set `load_if_exists: true`
2. **Fresh start**: `python run_tuning.py --fresh`
3. **Use new name**: Change `study_name` in config

---

### ❌ "Results are unrealistic"

**Symptoms**: Calmar > 100, CAGR > 200%, etc.

**Causes**:
1. Data quality issues (wrong prices, missing corporate actions)
2. Lookahead bias (future info leaking into past)
3. Overfitting to small sample size

**Debug**:
1. Run manual backtest with best config
2. Review `trade_results.csv` for anomalies (e.g., 1000% returns on single trade)
3. Check price data for errors
4. Validate quarters have sufficient trades (not just 1-2 lucky picks)

---

## Key Takeaways

### Do's ✅

- **Start broad, then narrow**: Coarse search → Analyze → Refine
- **Use Calmar for balanced optimization**: CAGR/MDD is robust
- **Monitor parameter importances**: Focus on what matters
- **Validate out-of-sample**: Don't just optimize on full period
- **Use dashboard**: Visual analysis >> staring at logs
- **Save versioned configs**: Track what you tried

### Don'ts ❌

- **Don't over-tune**: 500+ trials on 3 parameters = overfitting
- **Don't ignore data quality**: Garbage in, garbage out
- **Don't blindly trust best trial**: Check top-10, look for robustness
- **Don't skip manual baseline**: Optimize improvements, not from scratch
- **Don't forget to disable reports**: During tuning (wastes time)

---

## Next Steps

- **Run your first optimization**: Follow examples in this guide
- **Interpret results**: See [OUTPUTS_AND_METRICS.md](OUTPUTS_AND_METRICS.md)
- **Understand strategic implications**: See [STRATEGIC_OPTIONS.md](STRATEGIC_OPTIONS.md)
- **Master configuration**: See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)

---

*Last updated: March 2026*
