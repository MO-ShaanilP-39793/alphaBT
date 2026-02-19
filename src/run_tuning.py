"""
Optuna Hyperparameter Tuning Script for Backtesting Strategy

Supports both single-objective and multi-objective optimization.

Single-objective:
    Maximizes a single metric (Calmar, CAGR) or minimizes MDD.
    Config uses: objective, direction, failure_penalty

Multi-objective:
    Simultaneously optimizes multiple metrics (e.g., maximize CAGR + minimize MDD).
    Produces a Pareto front of non-dominated solutions.
    Config uses: objectives, directions, failure_penalties

Usage:
    python run_tuning.py                          # Use tuning_config.yaml
    python run_tuning.py --config my_config.yaml  # Use custom config
    python run_tuning.py --n-trials 200           # Override number of trials
    python run_tuning.py --fresh                  # Start fresh study (ignore existing)

After optimization:
    - Study is stored in SQLite for Optuna Dashboard analysis
    - Use export_config.py to export trial configs:
        python export_config.py --study-folder tuning_logs/<study> --best
        python export_config.py --study-folder tuning_logs/<study> --pareto --top 5
    - Run: optuna-dashboard sqlite:///tuning_logs/<study>/optuna_study.db
"""

import argparse
import shutil
import warnings
from datetime import datetime
from pathlib import Path
import pandas as pd
import optuna
from optuna.samplers import (
    TPESampler, 
    RandomSampler, 
    CmaEsSampler, 
    NSGAIISampler, 
    NSGAIIISampler,
)

from tuning import (
    load_tuning_config,
    sample_parameters,
    build_config,
    compute_objective,
    compute_multi_objective,
    compute_cagr,
    compute_max_drawdown,
    get_study_summary,
    SUPPORTED_OBJECTIVES,
)

from config.defaults import (
    DEFAULT_OBJECTIVE,
    DEFAULT_DIRECTION,
    DEFAULT_FAILURE_PENALTY,
    DEFAULT_FAILURE_PENALTY_MINIMIZE,
    DEFAULT_CALMAR_CAP,
    DEFAULT_SAMPLER,
    DEFAULT_MULTI_OBJ_SAMPLER,
    SAMPLER_SEED,
    DEFAULT_RUN_STOCK_SELECTION,
    DEFAULT_TUNING_CONFIG_PATH,
)

from backtest_strategy import (
    load_data,
    backtest_core,
)

from selection import (
    filter_tradeable_stocks,
    validate_price_data_coverage,
)

from reporting.summary_stats import compute_trailing_returns


class DataCache:
    """
    Singleton cache for data to avoid repeated I/O during optimization.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.input_data = None
            cls._instance.price_data = None
            cls._instance.index_data = None
            cls._instance.loaded = False
        return cls._instance
    
    def load(self, config: dict) -> None:
        """Load all data files once."""
        if self.loaded:
            return
        
        print("Loading data files...")
        self.input_data = load_data(config['input_data_path'])
        print(f"  - Input data: {len(self.input_data)} rows")
        
        self.price_data = load_data(config['price_data_path'])
        print(f"  - Price data: {len(self.price_data)} rows")
        
        self.index_data = load_data(config['index_data_path'])
        print(f"  - Index data: {len(self.index_data)} rows")
        
        self.loaded = True
        print("Data loading complete.\n")


def set_trial_user_attributes(
        trial: optuna.Trial,
        sampled_params: dict,
        results: dict) -> None:
    """
    Set user attributes on the trial for enhanced analysis in Optuna Dashboard.
    
    This function stores:
    - Expanded list-valued parameters (category counts, weights, TP/SL thresholds)
    - Performance metrics (total return, CAGR, MDD, win rate, number of trades)
    
    Parameters:
        trial: Optuna trial object
        sampled_params: Parameters sampled for this trial
        results: Results dictionary from backtest_core
    """
    # -------------------------------------------------------------------------
    # Expand list-valued parameters for per-category analysis
    # -------------------------------------------------------------------------
    category_scheme = sampled_params.get('category_scheme')
    
    # Get category labels based on scheme
    if category_scheme == 'volatility':
        cat_labels = ['high_vol', 'medium_vol', 'low_vol']
    elif category_scheme == 'mcap':
        cat_labels = ['largecap', 'midcap', 'smallcap']
    else:
        cat_labels = None
    
    # Expand category_counts (if category_based selection)
    if cat_labels and sampled_params.get('selection_type') == 'category_based':
        category_counts = sampled_params.get('category_counts')
        if category_counts is not None:
            for i, label in enumerate(cat_labels):
                trial.set_user_attr(f'category_count_{label}', category_counts[i])
        
        # Expand category_weights (only when using category weights, not equal)
        if sampled_params.get('category_based_selection_weighting_scheme') == 'use_category_weights':
            category_weights = sampled_params.get('category_weights')
            if category_weights is not None:
                for i, label in enumerate(cat_labels):
                    trial.set_user_attr(f'category_weight_{label}', round(category_weights[i], 3))
    
    # Expand fixed TP/SL thresholds (if fixed mode is used for that side)
    if cat_labels:
        if sampled_params.get('tp_enabled') and sampled_params.get('tp_mode') == 'fixed':
            fixed_tp = sampled_params.get('fixed_tp_thresholds')
            if fixed_tp is not None:
                for i, label in enumerate(cat_labels):
                    trial.set_user_attr(f'tp_threshold_{label}', round(fixed_tp[i], 3))
        
        if sampled_params.get('sl_enabled') and sampled_params.get('sl_mode') == 'fixed':
            fixed_sl = sampled_params.get('fixed_sl_thresholds')
            if fixed_sl is not None:
                for i, label in enumerate(cat_labels):
                    trial.set_user_attr(f'sl_threshold_{label}', round(fixed_sl[i], 3))
    
    # -------------------------------------------------------------------------
    # Store performance metrics
    # -------------------------------------------------------------------------
    trade_results = results.get('trade_results')
    if trade_results is not None and not trade_results.empty:
        # Extract and prepare daily portfolio values
        daily_pf_values = results.get('daily_pf_values')
        if daily_pf_values is not None and not daily_pf_values.empty:
            daily_pf = daily_pf_values.reset_index()
            daily_pf.columns = ['date', 'portfolio_value', 'quarter']
            
            # Number of trades
            trial.set_user_attr('n_trades', len(trade_results))
            
            # Win rate
            if 'stock_return' in trade_results.columns:
                valid_returns = trade_results['stock_return'].dropna()
                if len(valid_returns) > 0:
                    win_rate = (valid_returns > 0).mean()
                    trial.set_user_attr('win_rate', round(win_rate * 100, 2))
            
            # Total return (as percentage)
            initial_val = daily_pf['portfolio_value'].iloc[0]
            final_val = daily_pf['portfolio_value'].iloc[-1]
            total_return = ((final_val - initial_val) / initial_val) * 100
            trial.set_user_attr('total_return', round(total_return, 2))
            
            # CAGR (as percentage)
            cagr = compute_cagr(daily_pf)
            if cagr is not None:
                trial.set_user_attr('cagr', round(cagr * 100, 2))
            
            # Maximum Drawdown (as percentage)
            mdd = compute_max_drawdown(daily_pf)
            if mdd is not None:
                trial.set_user_attr('mdd', round(mdd * 100, 2))
            
            # Trailing Returns (1m, 3m, 6m, 1y, 3y, 5y, 10y)
            if len(daily_pf) > 1:
                # Prepare returns DataFrame (Date index, return columns)
                returns_df = pd.DataFrame({
                    'Portfolio': daily_pf['portfolio_value'].pct_change()
                })
                returns_df.index = pd.to_datetime(daily_pf['date'])
                returns_df = returns_df.dropna()
                
                if not returns_df.empty:
                    trailing_returns = compute_trailing_returns(returns_df, input_frequency="daily")
                    
                    # trailing_returns has strategies as index, periods as columns
                    if not trailing_returns.empty and 'Portfolio' in trailing_returns.index:
                        # Map period labels to attribute names
                        period_map = {
                            '1-m': 'trailing_1m',
                            '3-m': 'trailing_3m',
                            '6-m': 'trailing_6m',
                            '1-year': 'trailing_1y',
                            '3-years': 'trailing_3y',
                            '5-years': 'trailing_5y',
                            '10-years': 'trailing_10y'
                        }
                        
                        for period, attr_name in period_map.items():
                            if period in trailing_returns.columns:
                                value = trailing_returns.loc['Portfolio', period]
                                if not pd.isna(value):
                                    trial.set_user_attr(attr_name, value)  # Already rounded to 2 decimals


def _parse_objective_config(optuna_config: dict) -> dict:
    """
    Parse objective configuration and determine single vs multi-objective mode.
    
    Detection logic:
        - If 'objectives' (list) is present → multi-objective
        - If 'objective' (string) is present → single-objective
        - If both present → error
    
    Parameters:
        optuna_config: The 'optuna' section of tuning config
        
    Returns:
        Dictionary with keys:
            - is_multi_objective (bool)
            - objective_names (list of str): e.g., ['calmar'] or ['cagr', 'mdd']
            - directions (list of str): e.g., ['maximize'] or ['maximize', 'minimize']
            - failure_penalties (list of float): per-objective penalty values
            - calmar_cap (float): cap for calmar objective
    """
    has_single = 'objective' in optuna_config
    has_multi = 'objectives' in optuna_config
    
    # Enforce mutual exclusivity
    if has_single and has_multi:
        raise ValueError(
            "Cannot specify both 'objective' (single-objective) and 'objectives' "
            "(multi-objective) in the same config. Choose one mode:\n"
            "  - For single-objective: use 'objective' + 'direction'\n"
            "  - For multi-objective: use 'objectives' + 'directions'"
        )
    
    if has_multi:
        # Multi-objective mode
        objective_names = optuna_config['objectives']
        if not isinstance(objective_names, list) or len(objective_names) < 2:
            raise ValueError(
                "'objectives' must be a list with at least 2 entries "
                "(e.g., ['cagr', 'mdd']). For single-objective, use 'objective' instead."
            )
        
        # Validate objective names
        for name in objective_names:
            if name not in SUPPORTED_OBJECTIVES:
                raise ValueError(
                    f"Unknown objective: '{name}'. "
                    f"Supported: {SUPPORTED_OBJECTIVES}"
                )
        
        # Directions (required for multi-objective)
        directions = optuna_config.get('directions')
        if directions is None:
            raise ValueError(
                "'directions' is required for multi-objective optimization. "
                "Provide a list matching 'objectives' length "
                "(e.g., ['maximize', 'minimize'])."
            )
        if len(directions) != len(objective_names):
            raise ValueError(
                f"'directions' length ({len(directions)}) must match "
                f"'objectives' length ({len(objective_names)})."
            )
        
        # Failure penalties
        raw_penalties = optuna_config.get('failure_penalties')
        if raw_penalties is None:
            # Auto-derive from directions
            failure_penalties = [
                DEFAULT_FAILURE_PENALTY if d == 'maximize' else DEFAULT_FAILURE_PENALTY_MINIMIZE for d in directions
            ]
        elif isinstance(raw_penalties, list):
            if len(raw_penalties) != len(objective_names):
                raise ValueError(
                    f"'failure_penalties' length ({len(raw_penalties)}) must match "
                    f"'objectives' length ({len(objective_names)})."
                )
            failure_penalties = raw_penalties
        else:
            # Scalar — broadcast to all objectives
            failure_penalties = [float(raw_penalties)] * len(objective_names)
        
        return {
            'is_multi_objective': True,
            'objective_names': objective_names,
            'directions': directions,
            'failure_penalties': failure_penalties,
            'calmar_cap': optuna_config.get('calmar_cap', DEFAULT_CALMAR_CAP),
        }
    
    elif has_single:
        # Single-objective mode
        objective_name = optuna_config['objective']
        if objective_name not in SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"Unknown objective: '{objective_name}'. "
                f"Supported: {SUPPORTED_OBJECTIVES}"
            )
        
        direction = optuna_config.get('direction', DEFAULT_DIRECTION)
        failure_penalty = optuna_config.get('failure_penalty', DEFAULT_FAILURE_PENALTY)
        
        return {
            'is_multi_objective': False,
            'objective_names': [objective_name],
            'directions': [direction],
            'failure_penalties': [failure_penalty],
            'calmar_cap': optuna_config.get('calmar_cap', DEFAULT_CALMAR_CAP),
        }
    
    else:
        # Neither specified — default to single-objective calmar
        return {
            'is_multi_objective': False,
            'objective_names': [DEFAULT_OBJECTIVE],
            'directions': [DEFAULT_DIRECTION],
            'failure_penalties': [DEFAULT_FAILURE_PENALTY],
            'calmar_cap': optuna_config.get('calmar_cap', DEFAULT_CALMAR_CAP),
        }


def create_objective(tuning_config: dict, data_cache: DataCache, 
                     obj_config: dict, suppress_warnings: bool = True):
    """
    Create the objective function for Optuna optimization.
    
    Supports both single-objective (returns float) and multi-objective 
    (returns tuple of floats) modes.
    
    Parameters:
        tuning_config: Full tuning configuration
        data_cache: Cached data files
        obj_config: Parsed objective config from _parse_objective_config()
        suppress_warnings: If True, suppress warnings during trials (default: True)
        
    Returns:
        Objective function for Optuna (returns float or tuple of floats)
    """
    fixed_config = tuning_config['fixed']
    is_multi_obj = obj_config['is_multi_objective']
    objective_names = obj_config['objective_names']
    failure_penalties = obj_config['failure_penalties']
    calmar_cap = obj_config['calmar_cap']
    
    # Single penalty (float) for single-obj, tuple for multi-obj
    failure_return = tuple(failure_penalties) if is_multi_obj else failure_penalties[0]
    
    def objective(trial: optuna.Trial):
        """
        Objective function that samples parameters, runs backtest,
        and returns the optimization objective(s).
        """
        with warnings.catch_warnings():
            if suppress_warnings:
                warnings.simplefilter("ignore")
            
            try:
                # 1. Sample parameters from search space
                sampled_params = sample_parameters(trial, tuning_config)
                
                # 2. Build full config
                config = build_config(fixed_config, sampled_params)
                
                # 3. Run backtest core (skip_validation=True since we pre-validated)
                results = backtest_core(
                    config=config,
                    input_data=data_cache.input_data,
                    price_data=data_cache.price_data,
                    index_data=data_cache.index_data,
                    skip_price_data_validation=True
                )
                
                # 4. Validate results
                if results is None:
                    return failure_return
                
                daily_pf_values = results.get('daily_pf_values')
                if daily_pf_values is None or daily_pf_values.empty:
                    return failure_return
                
                # Prepare daily_pf in expected format
                daily_pf = daily_pf_values.reset_index()
                daily_pf.columns = ['date', 'portfolio_value', 'quarter']
                
                # 5. Compute objective value(s)
                if is_multi_obj:
                    values = compute_multi_objective(
                        objective_names,
                        daily_pf,
                        cap=calmar_cap
                    )
                    # Replace None values with corresponding failure penalties
                    final_values = tuple(
                        fp if v is None else v
                        for v, fp in zip(values, failure_penalties)
                    )
                    if all(v == fp for v, fp in zip(final_values, failure_penalties)):
                        return failure_return
                    
                    # 6. Store user attributes
                    set_trial_user_attributes(trial, sampled_params, results)
                    return final_values
                else:
                    objective_value = compute_objective(
                        objective_names[0],
                        daily_pf,
                        cap=calmar_cap
                    )
                    if objective_value is None:
                        return failure_return
                    
                    # 6. Store user attributes
                    set_trial_user_attributes(trial, sampled_params, results)
                    return objective_value
                
            except Exception as e:
                warnings.warn(f"Trial {trial.number} failed with error: {str(e)}")
                return failure_return
    
    return objective


def get_sampler(sampler_name: str, is_multi_obj: bool = False) -> optuna.samplers.BaseSampler:
    """
    Get Optuna sampler by name.
    
    Parameters:
        sampler_name: Name of sampler ('TPE', 'Random', 'CmaEs', 'NSGA-II', 'NSGA-III')
        is_multi_obj: Whether the study is multi-objective
        
    Returns:
        Optuna sampler instance
        
    Raises:
        ValueError: If sampler is unknown or incompatible with study type
    """
    # CmaEs does not support multi-objective
    if is_multi_obj and sampler_name == 'CmaEs':
        raise ValueError(
            "CmaEsSampler does not support multi-objective optimization. "
            "Use 'NSGA-II', 'NSGA-III', or 'TPE' instead."
        )
    
    samplers = {
        'TPE': TPESampler(seed=SAMPLER_SEED),
        'Random': RandomSampler(seed=SAMPLER_SEED),
        'CmaEs': CmaEsSampler(seed=SAMPLER_SEED),
        'NSGA-II': NSGAIISampler(seed=SAMPLER_SEED),
        'NSGA-III': NSGAIIISampler(seed=SAMPLER_SEED),
    }
    
    if sampler_name not in samplers:
        raise ValueError(f"Unknown sampler: {sampler_name}. Choose from {list(samplers.keys())}")
    
    return samplers[sampler_name]


def setup_study_folder(study_name: str) -> Path:
    """
    Create the study folder structure under tuning_logs/.
    
    Creates:
        tuning_logs/
            <study_name>/
    
    Parameters:
        study_name: Name of the Optuna study
        
    Returns:
        Path to the study folder
    """
    # tuning_logs is at the same level as src/
    src_dir = Path(__file__).parent
    tuning_logs_dir = src_dir.parent / "tuning_logs"
    study_folder = tuning_logs_dir / study_name
    
    # Create directories if they don't exist
    study_folder.mkdir(parents=True, exist_ok=True)
    
    return study_folder


def run_optimization(config_path: str = 'tuning_config.yaml',
                     n_trials_override: int = None,
                     fresh_study: bool = False) -> optuna.Study:
    """
    Run optimization.
    
    Parameters:
        config_path: Path to tuning configuration YAML
        n_trials_override: Override number of trials from config
        fresh_study: If True, delete existing study and start fresh
        
    Returns:
        Completed Optuna study
    """
    print("=" * 70)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config file: {config_path}\n")
    
    # Load tuning configuration
    print("Loading tuning configuration...")
    tuning_config = load_tuning_config(config_path)
    
    fixed_config = tuning_config['fixed']
    optuna_config = tuning_config['optuna']
    
    study_name = optuna_config['study_name']
    n_trials = n_trials_override or optuna_config['n_trials']
    sampler_name = optuna_config.get('sampler', DEFAULT_SAMPLER)
    load_if_exists = optuna_config.get('load_if_exists', True) and not fresh_study
    
    # Parse objective configuration (detects single vs multi-objective)
    obj_config = _parse_objective_config(optuna_config)
    is_multi_obj = obj_config['is_multi_objective']
    
    # Default sampler for multi-objective if not explicitly set
    if is_multi_obj and 'sampler' not in optuna_config:
        sampler_name = DEFAULT_MULTI_OBJ_SAMPLER
    
    # Setup study folder structure
    study_folder = setup_study_folder(study_name)
    storage = f"sqlite:///{study_folder / 'optuna_study.db'}"
    tuning_config_copy_path = study_folder / 'tuning_config_used.yaml'
    
    # Display configuration
    objective_display = (
        ' + '.join(f"{n} ({d})" for n, d in zip(obj_config['objective_names'], obj_config['directions']))
    )
    mode_label = "Multi-objective" if is_multi_obj else "Single-objective"
    
    print(f"  Study name: {study_name}")
    print(f"  Study folder: {study_folder}")
    print(f"  Storage: {storage}")
    print(f"  Mode: {mode_label}")
    print(f"  Objective(s): {objective_display}")
    print(f"  Trials: {n_trials}")
    print(f"  Sampler: {sampler_name}")
    print(f"  Load existing: {load_if_exists}")
    print(f"  Quarter range: {fixed_config['first_quarter']} - {fixed_config['last_quarter']}")
    print()
    
    # Initialize data cache and load data
    data_cache = DataCache()
    data_cache.load(fixed_config)
    
    # -------------------------------------------------------------------------
    # Data Quality Validation
    # -------------------------------------------------------------------------
    validate_input_data_flag = fixed_config.get('validate_input_data', True)  # always default True for tuning
    
    if validate_input_data_flag:
        run_stock_selection_flag = fixed_config.get('run_stock_selection', DEFAULT_RUN_STOCK_SELECTION)
        first_quarter = fixed_config['first_quarter']
        last_quarter = fixed_config['last_quarter']
        
        if run_stock_selection_flag:
            # Filter out stocks with price data issues before selection
            print("Validating price data coverage...")
            filtered_input, data_issues = filter_tradeable_stocks(
                data_cache.input_data,
                data_cache.price_data,
                first_quarter,
                last_quarter
            )
            
            if not data_issues.empty:
                # Save data issues to csv
                issues_path = study_folder / 'data_quality_issues.csv'
                data_issues.to_csv(issues_path, index=False)
                
                original_count = len(data_cache.input_data)
                filtered_count = len(filtered_input)
                removed_count = original_count - filtered_count
                
                print(f"  Data Quality Check Results:")
                print(f"    - Found {len(data_issues)} stock-quarter issues")
                print(f"    - Affecting {data_issues['co_name'].nunique()} unique stocks")
                print(f"    - Filtered {removed_count} rows from input data ({original_count} → {filtered_count})")
                print(f"    - Report saved to: {issues_path}")
            else:
                print("  No price data issues found. All stocks are tradeable.")
            
            # Replace cached input_data with filtered version
            data_cache.input_data = filtered_input
            print()
        else:
            # Save and Report issues but don't filter (user's preselected portfolio)
            print("Validating price data coverage for preselected portfolio...")
            data_issues = validate_price_data_coverage(
                data_cache.input_data,
                data_cache.price_data,
                first_quarter,
                last_quarter
            )
            
            if not data_issues.empty:
                # Save report for user to review
                issues_path = study_folder / 'data_quality_issues.csv'
                data_issues.to_csv(issues_path, index=False)
                
                print(f"  WARNING: Found {len(data_issues)} stock-quarter issues in preselected portfolio")
                print(f"    - Affecting {data_issues['co_name'].nunique()} unique stocks")
                print(f"    - These positions will be held as cash during simulation")
                print(f"    - Report saved: {issues_path}")
            else:
                print("  No price data issues found. All stocks are tradeable.")
            print()
    else:
        print("Skipping input data validation (already validated).\n")
    
    # Copy tuning config to study folder
    shutil.copy2(config_path, tuning_config_copy_path)
    print(f"Tuning config saved to: {tuning_config_copy_path}\n")
    
    # Delete existing study if fresh start requested
    if fresh_study is True:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"Deleted existing study: {study_name}")
        except KeyError:
            pass  # Study doesn't exist
    
    # Create or load study
    sampler = get_sampler(sampler_name, is_multi_obj=is_multi_obj)
    
    if is_multi_obj:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            directions=obj_config['directions'],
            load_if_exists=load_if_exists,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            direction=obj_config['directions'][0],
            load_if_exists=load_if_exists,
        )
    
    existing_trials = len(study.trials)
    if existing_trials > 0:
        print(f"Resuming study with {existing_trials} existing trials")
        print(f"Running {n_trials} additional trials...\n")
    else:
        print(f"Starting new study with {n_trials} trials...\n")
    
    # Create objective function
    objective = create_objective(tuning_config, data_cache, obj_config)
    
    # Run optimization
    print("-" * 70)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,  # memory management
    )
    print("-" * 70)
    
    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    summary = get_study_summary(study)
    print(f"\nStudy Summary:")
    print(f"  Total trials: {summary['n_trials']}")
    print(f"  Completed: {summary['n_completed']}")
    print(f"  Failed: {summary['n_failed']}")
    
    # Map objective names to display labels
    objective_labels = {
        'calmar': 'Calmar ratio',
        'cagr': 'CAGR',
        'mdd': 'Maximum Drawdown'
    }
    
    if is_multi_obj:
        # ----- Multi-objective results -----
        n_pareto = summary.get('n_pareto_optimal', 0)
        if n_pareto > 0:
            print(f"\nPareto Front: {n_pareto} non-dominated solutions")
            
            # Per-objective ranges across Pareto front
            for i, (name, direction) in enumerate(zip(obj_config['objective_names'], obj_config['directions'])):
                label = objective_labels.get(name, name)
                obj_min = summary.get(f'objective_{i}_min')
                obj_max = summary.get(f'objective_{i}_max')
                if obj_min is not None and obj_max is not None:
                    print(f"  {label} ({direction}): {obj_min:.4f} — {obj_max:.4f}")
            
            # Print top Pareto trials (sorted by first objective, respecting direction)
            pareto_trials = study.best_trials
            first_dir = obj_config['directions'][0]
            pareto_sorted = sorted(
                pareto_trials,
                key=lambda t: t.values[0],
                reverse=(first_dir == 'maximize')
            )
            
            n_display = min(10, len(pareto_sorted))
            print(f"\n  Top {n_display} Pareto-optimal trials (sorted by {obj_config['objective_names'][0]}):")
            header_parts = ["  Trial"]
            for name in obj_config['objective_names']:
                header_parts.append(f"{objective_labels.get(name, name):>14s}")
            for extra in ['CAGR%', 'MDD%', 'Win Rate%', 'Trades']:
                header_parts.append(f"{extra:>10s}")
            print("  " + " | ".join(header_parts))
            
            for t in pareto_sorted[:n_display]:
                parts = [f"  {t.number:>5d}"]
                for v in t.values:
                    parts.append(f"{v:>14.4f}")
                # User attrs
                for key, fmt in [('cagr', '{:>10.2f}'), ('mdd', '{:>10.2f}'), 
                                 ('win_rate', '{:>10.2f}'), ('n_trades', '{:>10d}')]:
                    val = t.user_attrs.get(key)
                    if val is not None:
                        parts.append(fmt.format(val))
                    else:
                        parts.append(f"{'N/A':>10s}")
                print(" | ".join(parts))
        else:
            print("\nNo Pareto-optimal trials found.")
    else:
        # ----- Single-objective results -----
        if summary['best_value'] is not None:
            objective_name = obj_config['objective_names'][0]
            objective_label = objective_labels.get(objective_name, objective_name)
            
            print(f"\nBest Trial:")
            print(f"  Trial number: {summary['best_trial_number']}")
            print(f"  {objective_label}: {summary['best_value']:.4f}")
            
            best_trial = study.best_trial
            if 'total_return' in best_trial.user_attrs:
                print(f"  Total return: {best_trial.user_attrs['total_return']:.2f}%")
            if 'cagr' in best_trial.user_attrs:
                print(f"  CAGR: {best_trial.user_attrs['cagr']:.2f}%")
            if 'mdd' in best_trial.user_attrs:
                print(f"  Max Drawdown: {best_trial.user_attrs['mdd']:.2f}%")
            if 'win_rate' in best_trial.user_attrs:
                print(f"  Win rate: {best_trial.user_attrs['win_rate']:.2f}%")
            if 'n_trades' in best_trial.user_attrs:
                print(f"  Number of trades: {best_trial.user_attrs['n_trades']}")
        else:
            print("\nNo successful trials completed.")
    
    # ----- Export guidance -----
    print("\n" + "-" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nStudy artifacts saved to: {study_folder}")
    print(f"  - optuna_study.db (trial history)")
    print(f"  - tuning_config_used.yaml (config used)")
    
    print(f"\nTo export a config from this study:")
    if is_multi_obj:
        print(f"  python export_config.py --study-folder {study_folder} --pareto --top 5")
        print(f"  python export_config.py --study-folder {study_folder} --trial <N>")
    else:
        print(f"  python export_config.py --study-folder {study_folder} --best")
        print(f"  python export_config.py --study-folder {study_folder} --trial <N>")
    
    print(f"\nTo analyze the study visually, run:")
    print(f"  optuna-dashboard {storage}")
    
    return study


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter tuning for backtesting strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tuning.py                          # Default config
  python run_tuning.py --config my_config.yaml  # Custom config
  python run_tuning.py --n-trials 200           # More trials
  python run_tuning.py --fresh                  # Fresh start

After optimization:
  python export_config.py --study-folder tuning_logs/<study> --best     # Export best config (single-obj)
  python export_config.py --study-folder tuning_logs/<study> --pareto   # Export Pareto configs (multi-obj)
  optuna-dashboard sqlite:///tuning_logs/<study>/optuna_study.db        # View dashboard
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default=DEFAULT_TUNING_CONFIG_PATH,
        help='Path to tuning configuration YAML (default: tuning_config.yaml)'
    )
    
    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=None,
        help='Number of trials (overrides config file)'
    )
    
    parser.add_argument(
        '--fresh', '-f',
        action='store_true',
        help='Start fresh study (delete existing)'
    )
    
    args = parser.parse_args()
    
    # Run optimization
    study = run_optimization(
        config_path=args.config,
        n_trials_override=args.n_trials,
        fresh_study=args.fresh,
    )
    
    return study


if __name__ == '__main__':
    main()
