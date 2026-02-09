"""
Optuna Hyperparameter Tuning Script for Backtesting Strategy

This script runs hyperparameter optimization using Optuna to find the best
strategy parameters that maximize the Calmar ratio.

Usage:
    python run_tuning.py                          # Use tuning_config.yaml
    python run_tuning.py --config my_config.yaml  # Use custom config
    python run_tuning.py --n-trials 200           # Override number of trials
    python run_tuning.py --fresh                  # Start fresh study (ignore existing)

After optimization:
    - Best parameters are exported to best_config.yaml
    - Study is stored in SQLite for Optuna Dashboard analysis
    - Run: optuna-dashboard sqlite:///optuna_studies.db
"""

import argparse
import shutil
import warnings
from datetime import datetime
from pathlib import Path
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler

from tuning import (
    load_tuning_config,
    sample_parameters,
    build_config,
    compute_objective,
    export_best_config,
    get_study_summary,
)

from backtest_strategy import (
    load_data,
    backtest_core,
)

from selection import (
    filter_tradeable_stocks,
    validate_price_data_coverage,
)


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


def create_objective(tuning_config: dict, data_cache: DataCache, suppress_warnings: bool = True):
    """
    Create the objective function for Optuna optimization.
    
    This is a factory function that returns the actual objective function
    with the config and data cache bound via closure.
    
    Parameters:
        tuning_config: Full tuning configuration
        data_cache: Cached data files
        suppress_warnings: If True, suppress warnings during trials (default: True)
        
    Returns:
        Objective function for Optuna
    """
    fixed_config = tuning_config['fixed']
    optuna_config = tuning_config['optuna']
    objective_name = optuna_config.get('objective', 'calmar')
    calmar_cap = optuna_config.get('calmar_cap', 10.0)
    failure_penalty = optuna_config.get('failure_penalty', -999.0)
    
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function that samples parameters, runs backtest, 
        and returns the specified optimization objective.
        """
        # Suppress warnings during trial execution to avoid cluttering output
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
                
                # 4. Compute and return objective value
                if results is None:
                    return failure_penalty
                
                daily_pf_values = results.get('daily_pf_values')
                if daily_pf_values is None or daily_pf_values.empty:
                    return failure_penalty
                
                # Prepare daily_pf in expected format
                daily_pf = daily_pf_values.reset_index()
                daily_pf.columns = ['date', 'portfolio_value', 'quarter']
                
                # Compute objective (Calmar or CAGR)
                objective_value = compute_objective(
                    objective_name, 
                    daily_pf, 
                    cap=calmar_cap
                )
                
                if objective_value is None:
                    return failure_penalty
                
                # Store additional metrics as trial user attributes for analysis
                trade_results = results.get('trade_results')
                if trade_results is not None and not trade_results.empty:
                    trial.set_user_attr('n_trades', len(trade_results))
                    
                    # Win rate
                    if 'stock_return' in trade_results.columns:
                        valid_returns = trade_results['stock_return'].dropna()
                        if len(valid_returns) > 0:
                            win_rate = (valid_returns > 0).mean()
                            trial.set_user_attr('win_rate', round(win_rate, 4))
                    
                    # Total return
                    initial_val = daily_pf['portfolio_value'].iloc[0]
                    final_val = daily_pf['portfolio_value'].iloc[-1]
                    total_return = (final_val - initial_val) / initial_val
                    trial.set_user_attr('total_return', round(total_return, 4))
                
                return objective_value
                
            except Exception as e:
                # Log error but don't crash the optimization
                warnings.warn(f"Trial {trial.number} failed with error: {str(e)}")
                return failure_penalty
    
    return objective


def get_sampler(sampler_name: str) -> optuna.samplers.BaseSampler:
    """
    Get Optuna sampler by name.
    
    Parameters:
        sampler_name: Name of sampler ('TPE', 'Random', 'CmaEs')
        
    Returns:
        Optuna sampler instance
    """
    samplers = {
        'TPE': TPESampler(seed=42),
        'Random': RandomSampler(seed=42),
        'CmaEs': CmaEsSampler(seed=42),
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
    Run Optuna hyperparameter optimization.
    
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
    sampler_name = optuna_config.get('sampler', 'TPE')
    direction = optuna_config.get('direction', 'maximize')
    load_if_exists = optuna_config.get('load_if_exists', True) and not fresh_study
    
    # Setup study folder structure
    study_folder = setup_study_folder(study_name)
    storage = f"sqlite:///{study_folder / 'optuna_study.db'}"
    best_config_path = study_folder / 'best_config.yaml'
    tuning_config_copy_path = study_folder / 'tuning_config_used.yaml'
    
    objective_name = optuna_config.get('objective', 'calmar')
    
    print(f"  Study name: {study_name}")
    print(f"  Study folder: {study_folder}")
    print(f"  Storage: {storage}")
    print(f"  Objective: {objective_name}")
    print(f"  Trials: {n_trials}")
    print(f"  Sampler: {sampler_name}")
    print(f"  Direction: {direction}")
    print(f"  Load existing: {load_if_exists}")
    print(f"  Quarter range: {fixed_config['first_quarter']} - {fixed_config['last_quarter']}")
    print()
    
    # Initialize data cache and load data
    data_cache = DataCache()
    data_cache.load(fixed_config)
    
    # -------------------------------------------------------------------------
    # Data Quality Validation
    # -------------------------------------------------------------------------
    validate_input_data_flag = fixed_config.get('validate_input_data', True)
    
    if validate_input_data_flag:
        run_stock_selection_flag = fixed_config.get('run_stock_selection', True)
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
    sampler = get_sampler(sampler_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction=direction,
        load_if_exists=load_if_exists,
    )
    
    existing_trials = len(study.trials)
    if existing_trials > 0:
        print(f"Resuming study with {existing_trials} existing trials")
        print(f"Running {n_trials} additional trials...\n")
    else:
        print(f"Starting new study with {n_trials} trials...\n")
    
    # Create objective function
    objective = create_objective(tuning_config, data_cache)
    
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
    
    if summary['best_value'] is not None:
        objective_name = optuna_config.get('objective', 'calmar')
        
        # Map objective names to display labels
        objective_labels = {
            'calmar': 'Calmar ratio',
            'cagr': 'CAGR',
            'mdd': 'Maximum Drawdown'
        }
        objective_label = objective_labels.get(objective_name, objective_name)
        
        print(f"\nBest Trial:")
        print(f"  Trial number: {summary['best_trial_number']}")
        print(f"  {objective_label}: {summary['best_value']:.4f}")
        
        # Get best trial details
        best_trial = study.best_trial
        if 'total_return' in best_trial.user_attrs:
            print(f"  Total return: {best_trial.user_attrs['total_return']:.2%}")
        if 'win_rate' in best_trial.user_attrs:
            print(f"  Win rate: {best_trial.user_attrs['win_rate']:.2%}")
        if 'n_trades' in best_trial.user_attrs:
            print(f"  Number of trades: {best_trial.user_attrs['n_trades']}")
        
        # Export best config
        print("\n" + "-" * 70)
        export_best_config(best_trial.params, tuning_config, str(best_config_path))
    else:
        print("\nNo successful trials completed.")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nStudy artifacts saved to: {study_folder}")
    print(f"  - optuna_study.db (trial history)")
    print(f"  - tuning_config_used.yaml (config used)")
    if summary['best_value'] is not None:
        print(f"  - best_config.yaml (best parameters)")
    print(f"\nTo analyze the study, run:")
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
  optuna-dashboard sqlite:///optuna_studies.db  # View dashboard
  python backtest_strategy.py                   # Run with best_config.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='tuning_config.yaml',
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
