"""
Tuning Utilities for Optuna Hyperparameter Optimization

This module provides helper functions for:
- Loading and parsing tuning configuration
- Sampling parameters from search space using Optuna trials
- Building full configuration dictionaries from sampled parameters
- Computing optimization metrics (Calmar ratio)
"""

import yaml
import pandas as pd
from typing import Any
import optuna
import warnings

from utils.logging_config import get_logger
from utils.metrics import compute_cagr, compute_max_drawdown, compute_calmar_ratio
from config.defaults import (
    get_dimension_categories,
    DEFAULT_SELECTION_TYPE,
    DEFAULT_CATEGORY_SCHEME,
    DEFAULT_CATEGORY_COUNTS,
    DEFAULT_CATEGORY_WEIGHTS,
    DEFAULT_WEIGHTING_SCHEME,
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_WEIGHTING,
    DEFAULT_SORT_BY,
    DEFAULT_TP_ENABLED,
    DEFAULT_SL_ENABLED,
    DEFAULT_TP_MODE,
    DEFAULT_SL_MODE,
    DEFAULT_ATR_PERIOD,
    DEFAULT_ATR_TP_MULTIPLIER,
    DEFAULT_ATR_SL_MULTIPLIER,
    DEFAULT_PIVOT_LOOKBACK_DAYS,
    DEFAULT_PIVOT_TP_LEVEL,
    DEFAULT_PIVOT_SL_LEVEL,
    DEFAULT_FLAT_TP_PCT,
    DEFAULT_FLAT_SL_PCT,
    DEFAULT_REGIME_FILTER_ENABLED,
    DEFAULT_REGIME_MA_PERIOD,
    DEFAULT_REGIME_EXIT_THRESHOLD,
    DEFAULT_VOL_ADJUSTMENT_ENABLED,
    DEFAULT_VOL_LOOKBACK,
    DEFAULT_HIGH_VOL_THRESHOLD,
    DEFAULT_LOW_VOL_THRESHOLD,
    DEFAULT_HIGH_VOL_MULTIPLIER,
    DEFAULT_LOW_VOL_MULTIPLIER,
    DEFAULT_TIERED_TP_THRESHOLDS,
    DEFAULT_TIERED_SL_THRESHOLDS,
    DEFAULT_CALMAR_CAP,
    DEFAULT_INDEPENDENT_TPSL_MODES,
    DEFAULT_TPSL_CATEGORY_DIMENSION,
    DEFAULT_TPSL_FALLBACK_PCT,
)

logger = get_logger(__name__)


def load_tuning_config(config_path: str = 'tuning_config.yaml') -> dict:
    """
    Load tuning configuration from YAML file.
    
    Parameters:
        config_path: Path to the tuning configuration YAML file
        
    Returns:
        Dictionary containing the full tuning configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def sample_parameter(trial: optuna.Trial, name: str, spec: dict) -> Any:
    """
    Sample a parameter from its search space.
    
    Parameters:
        trial: Optuna trial object
        name: Parameter name (used for Optuna's internal tracking)
        spec: Parameter specification dict with 'type' and bounds/choices
        
    Returns:
        Sampled parameter value
    """
    param_type = spec['type']
    
    if param_type == 'categorical':
        choices = spec['choices']
        # If choices is a list of lists
        if choices and isinstance(choices[0], list):
            # For list-valued categoricals, use index-based selection
            idx = trial.suggest_int(f"{name}_idx", 0, len(choices) - 1)
            return choices[idx]
        else:
            return trial.suggest_categorical(name, choices)
    
    elif param_type == 'int':
        return trial.suggest_int(name, spec['low'], spec['high'])
    
    elif param_type == 'float':
        if 'step' in spec:
            return trial.suggest_float(name, spec['low'], spec['high'], step=spec['step'])
        else:
            return trial.suggest_float(name, spec['low'], spec['high'])
    
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def _sample_selection_params(
    trial: optuna.Trial,
    params: dict,
    search_space: dict,
    tuning_config: dict,
    run_stock_selection: bool,
) -> tuple:
    """Sample stock-selection related parameters.
    
    Returns:
        Tuple of (selection_type, cat_weighting_scheme) for downstream conditional logic
    """
    selection_type = None
    cat_weighting_scheme = None

    if not run_stock_selection:
        return selection_type, cat_weighting_scheme

    params['selection_type'] = sample_parameter(trial, 'selection_type', search_space['selection_type'])
    selection_type = params['selection_type']

    # Sample weighting scheme early (needed for conditional sampling of category_weights)
    if selection_type == 'category_based' and 'category_based_selection_weighting_scheme' in search_space:
        params['category_based_selection_weighting_scheme'] = sample_parameter(
            trial, 'category_based_selection_weighting_scheme',
            search_space['category_based_selection_weighting_scheme'],
        )
        cat_weighting_scheme = params['category_based_selection_weighting_scheme']

    # top_k parameters
    if selection_type == 'top_k' and 'top_k' in tuning_config:
        for name, spec in tuning_config['top_k'].items():
            params[f'top_k_{name}'] = sample_parameter(trial, f'top_k_{name}', spec)

    return selection_type, cat_weighting_scheme


def _sample_tpsl_params(
    trial: optuna.Trial,
    params: dict,
    search_space: dict,
    tuning_config: dict,
) -> tuple:
    """Sample TP/SL mode and mode-specific parameters.
    
    Returns:
        Tuple of (tp_mode, sl_mode, tp_enabled, sl_enabled)
    """
    params['tp_enabled'] = sample_parameter(trial, 'tp_enabled', search_space['tp_enabled'])
    tp_enabled = params['tp_enabled']
    params['sl_enabled'] = sample_parameter(trial, 'sl_enabled', search_space['sl_enabled'])
    sl_enabled = params['sl_enabled']

    independent_tpsl_modes = tuning_config.get('independent_tpsl_modes', DEFAULT_INDEPENDENT_TPSL_MODES)
    tp_mode = None
    sl_mode = None

    if independent_tpsl_modes:
        if tp_enabled and 'tpsl_mode' in search_space:
            params['tp_mode'] = sample_parameter(trial, 'tp_mode', search_space['tpsl_mode'])
            tp_mode = params['tp_mode']
        if sl_enabled and 'tpsl_mode' in search_space:
            params['sl_mode'] = sample_parameter(trial, 'sl_mode', search_space['tpsl_mode'])
            sl_mode = params['sl_mode']
    else:
        if (tp_enabled or sl_enabled) and 'tpsl_mode' in search_space:
            shared_mode = sample_parameter(trial, 'tpsl_mode', search_space['tpsl_mode'])
            tp_mode = shared_mode
            sl_mode = shared_mode
            params['tp_mode'] = shared_mode
            params['sl_mode'] = shared_mode

    # Sample mode-specific parameters
    active_modes = set(filter(None, [tp_mode, sl_mode]))

    if 'tiered' in active_modes and 'tiered_tpsl' in tuning_config:
        tiered_cfg = tuning_config['tiered_tpsl']
        if tp_enabled and tp_mode == 'tiered':
            params['tiered_tp_thresholds'] = sample_parameter(
                trial, 'tiered_tp_thresholds', tiered_cfg['tp_thresholds'],
            )
        if sl_enabled and sl_mode == 'tiered':
            params['tiered_sl_thresholds'] = sample_parameter(
                trial, 'tiered_sl_thresholds', tiered_cfg['sl_thresholds'],
            )

    if 'atr' in active_modes and 'atr_tpsl' in tuning_config:
        atr_cfg = tuning_config['atr_tpsl']
        params['atr_period'] = sample_parameter(trial, 'atr_period', atr_cfg['period'])
        if tp_enabled and tp_mode == 'atr':
            params['atr_tp_multiplier'] = sample_parameter(trial, 'atr_tp_multiplier', atr_cfg['tp_multiplier'])
        if sl_enabled and sl_mode == 'atr':
            params['atr_sl_multiplier'] = sample_parameter(trial, 'atr_sl_multiplier', atr_cfg['sl_multiplier'])

    if 'pivot' in active_modes and 'pivot_tpsl' in tuning_config:
        pivot_cfg = tuning_config['pivot_tpsl']
        params['pivot_lookback_days'] = sample_parameter(trial, 'pivot_lookback_days', pivot_cfg['lookback_days'])
        if tp_enabled and tp_mode == 'pivot':
            params['pivot_tp_level'] = sample_parameter(trial, 'pivot_tp_level', pivot_cfg['tp_level'])
        if sl_enabled and sl_mode == 'pivot':
            params['pivot_sl_level'] = sample_parameter(trial, 'pivot_sl_level', pivot_cfg['sl_level'])

    if 'flat' in active_modes and 'flat_tpsl' in tuning_config:
        flat_cfg = tuning_config['flat_tpsl']
        if tp_enabled and tp_mode == 'flat':
            params['flat_tp'] = sample_parameter(trial, 'flat_tp', flat_cfg['tp'])
        if sl_enabled and sl_mode == 'flat':
            params['flat_sl'] = sample_parameter(trial, 'flat_sl', flat_cfg['sl'])

    return tp_mode, sl_mode, tp_enabled, sl_enabled


def _sample_index_exit_params(
    trial: optuna.Trial,
    params: dict,
    tuning_config: dict,
) -> None:
    """Sample index-exit (regime filter + vol adjustment) parameters in-place."""
    if 'index_exit' not in tuning_config:
        return

    index_config = tuning_config['index_exit']

    if 'regime_filter_enabled' in index_config:
        regime_filter_enabled = sample_parameter(trial, 'regime_filter_enabled', index_config['regime_filter_enabled'])
        params['regime_filter_enabled'] = regime_filter_enabled
        if regime_filter_enabled:
            params['regime_ma_period'] = sample_parameter(trial, 'regime_ma_period', index_config['regime_ma_period'])
            params['regime_exit_threshold'] = sample_parameter(trial, 'regime_exit_threshold', index_config['regime_exit_threshold'])

    if 'vol_adjustment_enabled' in index_config:
        vol_adjustment_enabled = sample_parameter(trial, 'vol_adjustment_enabled', index_config['vol_adjustment_enabled'])
        params['vol_adjustment_enabled'] = vol_adjustment_enabled
        if vol_adjustment_enabled:
            for sub_param in [
                'vol_lookback', 'high_vol_threshold', 'low_vol_threshold',
                'high_vol_multiplier', 'low_vol_multiplier',
            ]:
                params[sub_param] = sample_parameter(trial, sub_param, index_config[sub_param])


def sample_parameters(trial: optuna.Trial, tuning_config: dict) -> dict:
    """
    Sample all parameters from the search space for a single trial.
    
    Parameters:
        trial: Optuna trial object
        tuning_config: Full tuning configuration dictionary
        
    Returns:
        Dictionary of sampled parameter values
    """
    params = {}
    search_space = tuning_config['search_space']
    fixed_config = tuning_config['fixed']
    run_stock_selection = fixed_config['run_stock_selection']

    # ----- Selection params -----
    selection_type, cat_weighting_scheme = _sample_selection_params(
        trial, params, search_space, tuning_config, run_stock_selection,
    )

    # ----- TP/SL params -----
    tp_mode, sl_mode, tp_enabled, sl_enabled = _sample_tpsl_params(
        trial, params, search_space, tuning_config,
    )
    
    # ----- Determine if category_scheme needs sampling -----
    selection_params = {
        'selection_type', 'sort_by', 'min_prob_threshold',
        'category_counts', 'category_weights',
        'category_based_selection_weighting_scheme',
        'selection_dimension', 'weighting_dimension',
    }
    category_based_only_params = {
        'category_counts', 'category_weights',
        'category_based_selection_weighting_scheme',
        'selection_dimension', 'weighting_dimension',
    }
    already_sampled = {
        'selection_type', 'tp_enabled', 'sl_enabled', 'tpsl_mode',
        'tp_mode', 'sl_mode', 'category_based_selection_weighting_scheme',
    }

    has_new_dimensions = 'selection_dimension' in search_space or 'weighting_dimension' in search_space
    needs_category_scheme = (
        (run_stock_selection and selection_type == 'category_based' and not has_new_dimensions)
        or (tp_mode == 'tiered') or (sl_mode == 'tiered')
    )
    
    # ----- Sample remaining base parameters -----
    for name, spec in search_space.items():
        if name in already_sampled:
            continue
        if not run_stock_selection and name in selection_params:
            continue
        if name == 'category_scheme' and not needs_category_scheme:
            continue
        if name in category_based_only_params and selection_type != 'category_based':
            continue
        if name == 'category_weights' and cat_weighting_scheme == 'equal':
            continue
        if name == 'tpsl_category_dimension' and tp_mode != 'tiered' and sl_mode != 'tiered':
            continue
        params[name] = sample_parameter(trial, name, spec)
    
    # ----- Index exit params -----
    _sample_index_exit_params(trial, params, tuning_config)
    
    return params


def _build_tiered_config(category_scheme: str, tp_thresholds: list, sl_thresholds: list) -> dict:
    """Build tiered_config dict from threshold lists.
    
    Returns:
        Dict with structure: {
            'tp_pct': {cat_name: float, ...},
            'sl_pct': {cat_name: float, ...},
            'default_tp_pct': 0.05,
            'default_sl_pct': 0.05,
        }
    """
    categories = get_dimension_categories(category_scheme)
    tiered_config = {
        'tp_pct': dict(zip(categories, tp_thresholds)),
        'sl_pct': dict(zip(categories, sl_thresholds)),
        'default_tp_pct': DEFAULT_TPSL_FALLBACK_PCT,
        'default_sl_pct': DEFAULT_TPSL_FALLBACK_PCT,
    }
    return tiered_config


def build_config(fixed_config: dict, sampled_params: dict) -> dict:
    """
    Build a complete configuration dictionary from fixed and sampled parameters.
    
    Populates the full config structure expected by backtest_core, using sampled
    values where available and defaults otherwise.
    
    Parameters:
        fixed_config: Fixed parameters from tuning config
        sampled_params: Parameters sampled by Optuna
        
    Returns:
        Complete configuration dictionary compatible with backtest_core()
    """
    # Helper to get sampled value or default
    def get(key: str, default=None):
        return sampled_params.get(key, default)
    
    # Start with fixed config as base
    config = fixed_config.copy()
    
    # Fill out stock selection params
    config['selection_type'] = get('selection_type', DEFAULT_SELECTION_TYPE)
    config['category_scheme'] = get('category_scheme', DEFAULT_CATEGORY_SCHEME)
    config['category_counts'] = get('category_counts', DEFAULT_CATEGORY_COUNTS)
    config['category_weights'] = get('category_weights', DEFAULT_CATEGORY_WEIGHTS)
    config['category_based_selection_weighting_scheme'] = get('category_based_selection_weighting_scheme', DEFAULT_WEIGHTING_SCHEME)
    config['top_k_config'] = {
        'k': get('top_k_k', DEFAULT_TOP_K),
        'weighting_scheme': get('top_k_weighting_scheme', DEFAULT_TOP_K_WEIGHTING),
    }
    config['sort_by'] = get('sort_by', DEFAULT_SORT_BY)
    config['min_prob_threshold'] = get('min_prob_threshold')

    # Cross-dimensional selection/weighting: fall back to category_scheme when not specified
    config['selection_dimension'] = get('selection_dimension', config['category_scheme'])
    config['weighting_dimension'] = get('weighting_dimension', config['category_scheme'])
    config['tpsl_category_dimension'] = get('tpsl_category_dimension', DEFAULT_TPSL_CATEGORY_DIMENSION)

    # Fill out TP SL params
    config['tp_enabled'] = get('tp_enabled', DEFAULT_TP_ENABLED)
    config['sl_enabled'] = get('sl_enabled', DEFAULT_SL_ENABLED)
    config['tp_mode'] = get('tp_mode', DEFAULT_TP_MODE)
    config['sl_mode'] = get('sl_mode', DEFAULT_SL_MODE)
    
    config['atr_config'] = {
        'period': get('atr_period', DEFAULT_ATR_PERIOD),
        'tp_multiplier': get('atr_tp_multiplier', DEFAULT_ATR_TP_MULTIPLIER),
        'sl_multiplier': get('atr_sl_multiplier', DEFAULT_ATR_SL_MULTIPLIER),
    }

    config['pivot_config'] = {
        'lookback_days': get('pivot_lookback_days', DEFAULT_PIVOT_LOOKBACK_DAYS),
        'tp_level': get('pivot_tp_level', DEFAULT_PIVOT_TP_LEVEL),
        'sl_level': get('pivot_sl_level', DEFAULT_PIVOT_SL_LEVEL),
    }

    config['flat_config'] = {
        'tp_pct': get('flat_tp', DEFAULT_FLAT_TP_PCT),
        'sl_pct': get('flat_sl', DEFAULT_FLAT_SL_PCT),
    }

    config['index_exit'] = {
        'regime_filter': {
            'enabled': get('regime_filter_enabled', DEFAULT_REGIME_FILTER_ENABLED),
            'ma_period': get('regime_ma_period', DEFAULT_REGIME_MA_PERIOD),
            'exit_threshold': get('regime_exit_threshold', DEFAULT_REGIME_EXIT_THRESHOLD),
        },
        'vol_adjustment': {
            'enabled': get('vol_adjustment_enabled', DEFAULT_VOL_ADJUSTMENT_ENABLED),
            'lookback': get('vol_lookback', DEFAULT_VOL_LOOKBACK),
            'high_vol_threshold': get('high_vol_threshold', DEFAULT_HIGH_VOL_THRESHOLD),
            'low_vol_threshold': get('low_vol_threshold', DEFAULT_LOW_VOL_THRESHOLD),
            'high_vol_multiplier': get('high_vol_multiplier', DEFAULT_HIGH_VOL_MULTIPLIER),
            'low_vol_multiplier': get('low_vol_multiplier', DEFAULT_LOW_VOL_MULTIPLIER),
        }
    }
    
    # Build tiered_config (only when tp_mode or sl_mode == 'tiered')
    if config['tp_mode'] == 'tiered' or config['sl_mode'] == 'tiered':
        tp_thresholds = get('tiered_tp_thresholds', DEFAULT_TIERED_TP_THRESHOLDS)
        sl_thresholds = get('tiered_sl_thresholds', DEFAULT_TIERED_SL_THRESHOLDS)
        
        # Resolve which dimension to use for tiered TP/SL category keys
        tpsl_cat_dim = config['tpsl_category_dimension']
        if tpsl_cat_dim == 'selection':
            tpsl_scheme = config['selection_dimension']
        elif tpsl_cat_dim == 'weighting':
            tpsl_scheme = config['weighting_dimension']
        else:  # Direct dimension name: 'volatility' or 'mcap'
            tpsl_scheme = tpsl_cat_dim
        
        config['tiered_config'] = _build_tiered_config(tpsl_scheme, tp_thresholds, sl_thresholds)
    
    return config


# Supported objective names for validation
SUPPORTED_OBJECTIVES = ('calmar', 'cagr', 'mdd')


def compute_objective(objective_name: str, daily_pf_values: pd.DataFrame, **kwargs) -> float:
    """
    Compute the specified optimization objective from daily portfolio values.
    
    This is a dispatcher function that routes to the appropriate objective computation.
    
    Parameters:
        objective_name: Name of the objective ('calmar', 'cagr', 'mdd')
        daily_pf_values: DataFrame with columns ['date', 'portfolio_value', 'quarter']
        **kwargs: Additional objective-specific parameters:
            - cap (float): For 'calmar', maximum ratio value (default: 10.0)
        
    Returns:
        Objective value as a float, or None if computation fails
        
    Raises:
        ValueError: If objective_name is not supported
    """
    if objective_name == 'calmar':
        cap = kwargs.get('cap', DEFAULT_CALMAR_CAP)
        return compute_calmar_ratio(daily_pf_values, cap=cap)
    
    elif objective_name == 'cagr':
        return compute_cagr(daily_pf_values)
    
    elif objective_name == 'mdd':
        return compute_max_drawdown(daily_pf_values)
    
    else:
        raise ValueError(
            f"Unknown objective: '{objective_name}'. "
            f"Supported objectives: {SUPPORTED_OBJECTIVES}"
        )


def compute_multi_objective(
        objective_names: list, 
        daily_pf_values: pd.DataFrame, 
        **kwargs) -> tuple:
    """
    Compute multiple optimization objectives from daily portfolio values.
    
    Used by multi-objective optimization to return a tuple of values,
    one per objective.
    
    Parameters:
        objective_names: List of objective names (e.g., ['cagr', 'mdd'])
        daily_pf_values: DataFrame with columns ['date', 'portfolio_value', 'quarter']
        **kwargs: Additional objective-specific parameters:
            - cap (float): For 'calmar', maximum ratio value (default: 10.0)
        
    Returns:
        Tuple of objective values (one per objective name), or None for any
        that fail to compute
        
    Raises:
        ValueError: If any objective_name is not supported
    """
    values = []
    for name in objective_names:
        values.append(compute_objective(name, daily_pf_values, **kwargs))
    return tuple(values)


def export_best_config(
        best_params: dict, 
        tuning_config: dict, 
        output_path: str = 'best_config.yaml') -> None:
    """
    Export the best trial parameters to a YAML config file.
    
    The exported config is compatible with the standard backtest_strategy.py workflow.
    
    Parameters:
        best_params: Best parameters from Optuna study
        tuning_config: Original tuning configuration
        output_path: Path for the output YAML file
    """
    # Build full config from best params
    config = build_config(tuning_config['fixed'], best_params)
    
    # Add report generation flag (enable for final run)
    config['generate_report'] = True
    
    # Write to YAML
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info("Best configuration exported to: %s", output_path)


def is_multi_objective(study: optuna.Study) -> bool:
    """Check if a study is multi-objective (has multiple directions)."""
    return len(study.directions) > 1


def get_study_summary(study: optuna.Study) -> dict:
    """
    Get summary statistics from an Optuna study.
    
    Supports both single-objective and multi-objective studies.
    
    Parameters:
        study: Completed Optuna study
        
    Returns:
        Dictionary with study summary statistics.
        For multi-objective studies, includes 'is_multi_objective', 
        'n_pareto_optimal', and per-objective statistics.
    """
    trials_df = study.trials_dataframe()
    
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
    
    multi_obj = is_multi_objective(study)
    
    summary = {
        'study_name': study.study_name,
        'is_multi_objective': multi_obj,
        'directions': [d.name for d in study.directions],
        'n_trials': len(study.trials),
        'n_completed': len(completed_trials),
        'n_pruned': len(trials_df[trials_df['state'] == 'PRUNED']),
        'n_failed': len(trials_df[trials_df['state'] == 'FAIL']),
    }
    
    if multi_obj:
        # Multi-objective: report Pareto front statistics
        try:
            pareto_trials = study.best_trials
            summary['n_pareto_optimal'] = len(pareto_trials)
            
            if pareto_trials:
                # Per-objective statistics across Pareto front
                n_objectives = len(study.directions)
                pareto_values = [t.values for t in pareto_trials]
                
                for i in range(n_objectives):
                    obj_vals = [v[i] for v in pareto_values]
                    summary[f'objective_{i}_min'] = min(obj_vals)
                    summary[f'objective_{i}_max'] = max(obj_vals)
                    summary[f'objective_{i}_mean'] = sum(obj_vals) / len(obj_vals)
        except (RuntimeError, ValueError, IndexError) as e:
            warnings.warn(f"Could not compute Pareto statistics: {e}")
            summary['n_pareto_optimal'] = 0
        
        # No single best value/trial for multi-objective
        summary['best_value'] = None
        summary['best_trial_number'] = None
    else:
        # Single-objective
        summary['direction'] = study.directions[0].name
        try:
            summary['best_value'] = study.best_value
            summary['best_trial_number'] = study.best_trial.number
        except ValueError:
            summary['best_value'] = None
            summary['best_trial_number'] = None
        
        if not completed_trials.empty:
            summary['mean_value'] = completed_trials['value'].mean()
            summary['std_value'] = completed_trials['value'].std()
            summary['min_value'] = completed_trials['value'].min()
            summary['max_value'] = completed_trials['value'].max()
    
    return summary
