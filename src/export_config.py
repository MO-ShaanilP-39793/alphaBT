"""
Export Configuration from Optuna Study

Standalone script to export trial configurations from completed Optuna studies.
Supports both single-objective and multi-objective studies.

Usage:
    # Single-objective: export the best trial
    python export_config.py --study-folder tuning_logs/my_study --best

    # Export a specific trial by number
    python export_config.py --study-folder tuning_logs/my_study --trial 42

    # Multi-objective: export all Pareto-optimal configs
    python export_config.py --study-folder tuning_logs/my_study --pareto

    # Multi-objective: export top-5 Pareto configs (sorted by first objective)
    python export_config.py --study-folder tuning_logs/my_study --pareto --top 5
"""

import argparse
import yaml
import optuna
from pathlib import Path

from tuning import (
    load_tuning_config,
    build_config,
    is_multi_objective,
)

from config.defaults import DEFAULT_OBJECTIVE, DEFAULT_DIRECTION


def _load_study(study_folder: Path) -> tuple:
    """
    Load an Optuna study and its tuning config from a study folder.
    
    Parameters:
        study_folder: Path to the study folder (e.g., tuning_logs/my_study)
        
    Returns:
        Tuple of (study, tuning_config)
    """
    db_path = study_folder / 'optuna_study.db'
    config_path = study_folder / 'tuning_config_used.yaml'
    
    if not db_path.exists():
        raise FileNotFoundError(f"Study database not found: {db_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Tuning config not found: {config_path}")
    
    storage = f"sqlite:///{db_path}"
    
    # Load tuning config
    tuning_config = load_tuning_config(str(config_path))
    
    # Get study name from config
    study_name = tuning_config['optuna']['study_name']
    
    # Load study
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )
    
    return study, tuning_config


def _expand_trial_params(trial_params: dict, tuning_config: dict) -> dict:
    """
    Expand index-based parameters from trial.params back to their actual values.
    
    During sampling, list-valued categoricals (e.g., category_counts with choices
    like [[8, 7, 5], [15, 10, 5], ...]) are stored as {name}_idx indices in trial.params.
    This function converts them back to the actual values using the search space.
    
    Parameters:
        trial_params: Raw trial.params dict from Optuna (contains {name}_idx for list categoricals)
        tuning_config: Full tuning configuration (contains search space with choices)
        
    Returns:
        Expanded params dict with indices converted to actual values
    """
    expanded = trial_params.copy()
    
    # Collect all search spaces (base + conditional mode-specific blocks)
    search_spaces = [tuning_config.get('search_space', {})]
    for block_name in ['top_k', 'tiered_tpsl', 'flat_tpsl', 'atr_tpsl', 'pivot_tpsl', 'index_exit']:
        if block_name in tuning_config:
            search_spaces.append(tuning_config[block_name])
    
    # Find list-valued categoricals and expand indices
    for search_space in search_spaces:
        for name, spec in search_space.items():
            if not isinstance(spec, dict):
                continue
            if spec.get('type') != 'categorical':
                continue
            choices = spec.get('choices', [])
            # Check if this is a list-valued categorical
            if choices and isinstance(choices[0], list):
                idx_key = f"{name}_idx"
                if idx_key in trial_params:
                    idx = trial_params[idx_key]
                    if 0 <= idx < len(choices):
                        expanded[name] = choices[idx]
                    # Remove the index key since we've expanded it
                    del expanded[idx_key]
    
    return expanded


def _build_config_from_trial(trial: optuna.trial.FrozenTrial, 
                              tuning_config: dict) -> dict:
    """
    Build a full backtest config from a trial's parameters.
    
    Parameters:
        trial: Optuna trial (FrozenTrial)
        tuning_config: Original tuning configuration
        
    Returns:
        Complete configuration dict compatible with backtest_strategy.py
    """
    # Expand index-based params (e.g., category_counts_idx -> category_counts)
    expanded_params = _expand_trial_params(trial.params, tuning_config)
    config = build_config(tuning_config['fixed'], expanded_params)
    config['generate_report'] = True
    return config


def _write_config(config: dict, output_path: Path, trial: optuna.trial.FrozenTrial,
                  objective_names: list = None) -> None:
    """
    Write a config YAML file with a header comment showing trial metadata.
    
    Parameters:
        config: Configuration dictionary
        output_path: Path to write the YAML file
        trial: The trial this config was built from
        objective_names: Names of objectives (for header comment)
    """
    # Build header comment
    header_lines = [
        f"# Exported from Optuna trial {trial.number}",
    ]
    
    if trial.values is not None:
        if objective_names and len(objective_names) > 1:
            # Multi-objective
            for name, val in zip(objective_names, trial.values):
                header_lines.append(f"# {name}: {val:.6f}")
        elif trial.values:
            val = trial.values[0] if len(trial.values) == 1 else trial.value
            if objective_names:
                header_lines.append(f"# {objective_names[0]}: {val:.6f}")
            else:
                header_lines.append(f"# Objective value: {val:.6f}")
    
    # Add key user attrs
    for key, label in [('cagr', 'CAGR'), ('mdd', 'MDD'), 
                       ('total_return', 'Total Return'), ('win_rate', 'Win Rate'),
                       ('n_trades', 'Trades')]:
        val = trial.user_attrs.get(key)
        if val is not None:
            if key == 'n_trades':
                header_lines.append(f"# {label}: {val}")
            else:
                header_lines.append(f"# {label}: {val:.2f}%")
    
    header_lines.append("#")
    header = "\n".join(header_lines) + "\n\n"
    
    # Write YAML
    yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(header)
        f.write(yaml_content)
    
    print(f"  Exported: {output_path}")


def export_best(study_folder: Path) -> None:
    """
    Export the best trial config from a single-objective study.
    
    Parameters:
        study_folder: Path to the study folder
    """
    study, tuning_config = _load_study(study_folder)
    
    if is_multi_objective(study):
        print("ERROR: --best is only valid for single-objective studies.")
        print("For multi-objective studies, use --pareto or --trial <N>.")
        return
    
    try:
        best_trial = study.best_trial
    except ValueError:
        print("No completed trials found in this study.")
        return
    
    optuna_config = tuning_config.get('optuna', {})
    objective_names = [optuna_config.get('objective', DEFAULT_OBJECTIVE)]
    
    config = _build_config_from_trial(best_trial, tuning_config)
    output_path = study_folder / 'best_config.yaml'
    
    print(f"\nBest trial: {best_trial.number} (value: {best_trial.value:.6f})")
    _write_config(config, output_path, best_trial, objective_names)


def export_trial(study_folder: Path, trial_number: int) -> None:
    """
    Export a specific trial's config.
    
    Parameters:
        study_folder: Path to the study folder
        trial_number: Trial number to export
    """
    study, tuning_config = _load_study(study_folder)
    
    # Find the trial by number
    target_trial = None
    for trial in study.trials:
        if trial.number == trial_number:
            target_trial = trial
            break
    
    if target_trial is None:
        print(f"ERROR: Trial {trial_number} not found in study.")
        print(f"Available trials: 0-{len(study.trials) - 1}")
        return
    
    if target_trial.state != optuna.trial.TrialState.COMPLETE:
        print(f"WARNING: Trial {trial_number} has state '{target_trial.state.name}' (not COMPLETE).")
        print("Exporting anyway, but results may not be meaningful.")
    
    # Determine objective names
    optuna_config = tuning_config.get('optuna', {})
    if 'objectives' in optuna_config:
        objective_names = optuna_config['objectives']
    else:
        objective_names = [optuna_config.get('objective', DEFAULT_OBJECTIVE)]
    
    config = _build_config_from_trial(target_trial, tuning_config)
    output_path = study_folder / f'trial_{trial_number}_config.yaml'
    
    print(f"\nExporting trial {trial_number}:")
    _write_config(config, output_path, target_trial, objective_names)


def export_pareto(study_folder: Path, top_n: int = None) -> None:
    """
    Export Pareto-optimal trial configs from a multi-objective study.
    
    Parameters:
        study_folder: Path to the study folder
        top_n: If specified, export only top N trials (sorted by first objective)
    """
    study, tuning_config = _load_study(study_folder)
    
    if not is_multi_objective(study):
        print("ERROR: --pareto is only valid for multi-objective studies.")
        print("For single-objective studies, use --best or --trial <N>.")
        return
    
    pareto_trials = study.best_trials
    if not pareto_trials:
        print("No Pareto-optimal trials found.")
        return
    
    # Determine objective names and directions from config
    optuna_config = tuning_config.get('optuna', {})
    objective_names = optuna_config.get('objectives', ['objective_0', 'objective_1'])
    directions = optuna_config.get('directions', [DEFAULT_DIRECTION] * len(objective_names))
    
    # Sort by first objective (respecting direction)
    first_dir = directions[0]
    pareto_sorted = sorted(
        pareto_trials,
        key=lambda t: t.values[0],
        reverse=(first_dir == 'maximize')
    )
    
    # Limit to top N if specified
    if top_n is not None:
        pareto_sorted = pareto_sorted[:top_n]
    
    n_export = len(pareto_sorted)
    print(f"\nPareto front: {len(pareto_trials)} non-dominated solutions")
    print(f"Exporting {n_export} configs to: {study_folder / 'pareto_configs'}/\n")
    
    # Print summary table
    header_parts = [f"{'Trial':>6s}"]
    for name in objective_names:
        header_parts.append(f"{name:>14s}")
    print(" | ".join(header_parts))
    print("-" * (8 + 17 * len(objective_names)))
    
    for trial in pareto_sorted:
        parts = [f"{trial.number:>6d}"]
        for v in trial.values:
            parts.append(f"{v:>14.6f}")
        print(" | ".join(parts))
    
    print()
    
    # Export configs
    output_dir = study_folder / 'pareto_configs'
    for trial in pareto_sorted:
        config = _build_config_from_trial(trial, tuning_config)
        output_path = output_dir / f'trial_{trial.number}.yaml'
        _write_config(config, output_path, trial, objective_names)
    
    print(f"\nDone. {n_export} configs exported to {output_dir}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Export configuration from Optuna study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-objective: export the best trial
  python export_config.py --study-folder tuning_logs/my_study --best

  # Export a specific trial by number
  python export_config.py --study-folder tuning_logs/my_study --trial 42

  # Multi-objective: export all Pareto-optimal configs
  python export_config.py --study-folder tuning_logs/my_study --pareto

  # Multi-objective: export top-5 Pareto configs
  python export_config.py --study-folder tuning_logs/my_study --pareto --top 5

After exporting:
  copy <exported_config>.yaml strategy_config.yaml
  python backtest_strategy.py
        """
    )
    
    parser.add_argument(
        '--study-folder', '-s',
        required=True,
        help='Path to the study folder (e.g., tuning_logs/my_study)'
    )
    
    # Mutually exclusive export modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--best', '-b',
        action='store_true',
        help='Export the best trial config (single-objective only)'
    )
    mode_group.add_argument(
        '--trial', '-t',
        type=int,
        metavar='N',
        help='Export a specific trial by number'
    )
    mode_group.add_argument(
        '--pareto', '-p',
        action='store_true',
        help='Export Pareto-optimal trial configs (multi-objective only)'
    )
    
    parser.add_argument(
        '--top',
        type=int,
        metavar='N',
        default=None,
        help='With --pareto: export only top N configs (sorted by first objective)'
    )
    
    args = parser.parse_args()
    study_folder = Path(args.study_folder)
    
    if not study_folder.exists():
        print(f"ERROR: Study folder not found: {study_folder}")
        return
    
    if args.best:
        export_best(study_folder)
    elif args.trial is not None:
        export_trial(study_folder, args.trial)
    elif args.pareto:
        export_pareto(study_folder, top_n=args.top)


if __name__ == '__main__':
    main()
