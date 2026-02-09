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
    
    # Check if the trials require stock selection
    run_stock_selection = fixed_config['run_stock_selection']
    
    # Create set of params that should not be sampled when run_stock_selection is False
    selection_params = {
        'selection_type',
        'selection_method',
        'min_prob_threshold',
        'category_counts',
        'category_weights'
    }
    
    # Create set of params that need to be sampled only when selection_type is 'category_based'
    category_based_only_params = {'category_counts', 'category_weights'}
    
    # ----- Sample key parameters first to enable conditional sampling -----
    
    # Initialise selection_type to None
    selection_type = None
    # Sample it only if run_stock_selection is True
    if run_stock_selection is True:
        params['selection_type'] = sample_parameter(trial, 'selection_type', search_space['selection_type'])
        selection_type = params['selection_type']
    
    # Sample tp_enabled and sl_enabled
    params['tp_enabled'] = sample_parameter(trial, 'tp_enabled', search_space['tp_enabled'])
    tp_enabled = params['tp_enabled']
    params['sl_enabled'] = sample_parameter(trial, 'sl_enabled', search_space['sl_enabled'])
    sl_enabled = params['sl_enabled']

    # Sample tpsl_mode if at least one of TP or SL is enabled
    tpsl_mode = None
    if tp_enabled or sl_enabled:
        params['tpsl_mode'] = sample_parameter(trial, 'tpsl_mode', search_space['tpsl_mode'])
        tpsl_mode = params['tpsl_mode']
    
    # category_scheme needs to be sampled if
    # tpsl_mode is fixed
    # or if run stock selection is True and selection type is category based
    needs_category_scheme = (run_stock_selection and selection_type == 'category_based') or (tpsl_mode == 'fixed')
    
    # ----- Sample base parameters (excluding conditional ones) -----
    for name, spec in search_space.items():
        # Skip already sampled parameters
        if name in ('selection_type', 'tp_enabled', 'sl_enabled', 'tpsl_mode'):
            continue
        # Skip all selection related params if run_stock_selection is False
        if not run_stock_selection and name in selection_params:
            continue
        # Skip category_scheme if not needed
        if name == 'category_scheme' and not needs_category_scheme:
            continue
        # Skip category-based-only params if not category_based selection
        if name in category_based_only_params and selection_type != 'category_based':
            continue
        params[name] = sample_parameter(trial, name, spec)
    
    # ----- Sample selection type-specific parameters (conditional) -----
    # Only sample top_k params if stock selection is enabled and type is top_k
    if run_stock_selection is True and selection_type == 'top_k':
        top_k_space = tuning_config['top_k']
        for name, spec in top_k_space.items():
            params[f'top_k_{name}'] = sample_parameter(trial, f'top_k_{name}', spec)
    
    # ----- Sample TP/SL mode-specific parameters (conditional) -----
    if tpsl_mode == 'fixed':
        fixed_tpsl_config = tuning_config['fixed_tpsl']
        if tp_enabled:
            params['fixed_tp_thresholds'] = sample_parameter(trial, 'fixed_tp_thresholds', fixed_tpsl_config['tp_thresholds'])
        if sl_enabled:
            params['fixed_sl_thresholds'] = sample_parameter(trial, 'fixed_sl_thresholds', fixed_tpsl_config['sl_thresholds'])
    
    elif tpsl_mode == 'atr':
        atr_config = tuning_config['atr_tpsl']
        params['atr_period'] = sample_parameter(trial, 'atr_period', atr_config['period'])
        if tp_enabled:
            params['atr_tp_multiplier'] = sample_parameter(trial, 'atr_tp_multiplier', atr_config['tp_multiplier'])
        if sl_enabled:
            params['atr_sl_multiplier'] = sample_parameter(trial, 'atr_sl_multiplier', atr_config['sl_multiplier'])
    
    elif tpsl_mode == 'pivot':
        pivot_config = tuning_config['pivot_tpsl']
        params['pivot_lookback_days'] = sample_parameter(trial, 'pivot_lookback_days', pivot_config['lookback_days'])
        if tp_enabled:
            params['pivot_tp_level'] = sample_parameter(trial, 'pivot_tp_level', pivot_config['tp_level'])
        if sl_enabled:
            params['pivot_sl_level'] = sample_parameter(trial, 'pivot_sl_level', pivot_config['sl_level'])
    
    elif tpsl_mode == 'flat':
        flat_config = tuning_config['flat_tpsl']
        if tp_enabled:
            params['flat_tp'] = sample_parameter(trial, 'flat_tp', flat_config['tp'])
        if sl_enabled:
            params['flat_sl'] = sample_parameter(trial, 'flat_sl', flat_config['sl'])
    
    # ----- Sample index exit parameters (conditional) -----
    if 'index_exit' in tuning_config:
        index_config = tuning_config['index_exit']
        
        # If regime filter is enabled, sample its params
        if 'regime_filter_enabled' in index_config:
            regime_filter_enabled = sample_parameter(trial, 'regime_filter_enabled', index_config['regime_filter_enabled'])
            params['regime_filter_enabled'] = regime_filter_enabled
            
            if regime_filter_enabled:
                params['regime_ma_period'] = sample_parameter(trial, 'regime_ma_period', index_config['regime_ma_period'])
                params['regime_exit_threshold'] = sample_parameter(trial, 'regime_exit_threshold', index_config['regime_exit_threshold'])
        
        # If volatility adjustment is enabled, sample its params
        if 'vol_adjustment_enabled' in index_config:
            vol_adjustment_enabled = sample_parameter(trial, 'vol_adjustment_enabled', index_config['vol_adjustment_enabled'])
            params['vol_adjustment_enabled'] = vol_adjustment_enabled
            
            if vol_adjustment_enabled:
                for sub_param in [
                    'vol_lookback', 
                    'high_vol_threshold', 
                    'low_vol_threshold',
                    'high_vol_multiplier', 
                    'low_vol_multiplier'
                    ]:
                    params[sub_param] = sample_parameter(trial, sub_param, index_config[sub_param])
    
    return params


def _get_categories(category_scheme: str) -> list:
    """Return category names for a given scheme."""
    if category_scheme == 'volatility':
        return ['high_volatility', 'medium_volatility', 'low_volatility']
    else:  # mcap
        return ['largecap', 'midcap', 'smallcap']


def _build_tpsl_config(category_scheme: str, tp_thresholds: list, sl_thresholds: list) -> tuple:
    """Build TP_CONFIG and SL_CONFIG dicts from threshold lists."""
    categories = _get_categories(category_scheme)
    tp_config = {category_scheme: dict(zip(categories, tp_thresholds))}
    sl_config = {category_scheme: dict(zip(categories, sl_thresholds))}
    return tp_config, sl_config


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
    config['selection_type'] = get('selection_type', 'category_based')
    config['category_scheme'] = get('category_scheme', 'volatility')
    config['category_counts'] = get('category_counts', [5, 10, 15])
    config['category_weights'] = get('category_weights', [0.3, 0.3, 0.4])
    config['top_k_config'] = {
        'k': get('top_k_k', 30),
        'weighting_scheme': get('top_k_weighting_scheme', 'equal'),
    }
    config['selection_method'] = get('selection_method', 'probability')
    config['min_prob_threshold'] = get('min_prob_threshold')

    # Fill out TP SL params
    config['tp_enabled'] = get('tp_enabled', False)
    config['sl_enabled'] = get('sl_enabled', False)
    config['tpsl_mode'] = get('tpsl_mode')
    
    config['atr_config'] = {
        'period': get('atr_period', 14),
        'tp_multiplier': get('atr_tp_multiplier', 2.0),
        'sl_multiplier': get('atr_sl_multiplier', 1.5),
    }

    config['pivot_config'] = {
        'lookback_days': get('pivot_lookback_days', 60),
        'tp_level': get('pivot_tp_level', 'R1'),
        'sl_level': get('pivot_sl_level', 'S1'),
    }

    config['flat_config'] = {
        'tp_pct': get('flat_tp', 0.05),
        'sl_pct': get('flat_sl', 0.05),
    }

    config['index_exit'] = {
        'regime_filter': {
            'enabled': get('regime_filter_enabled', False),
            'ma_period': get('regime_ma_period', 20),
            'exit_threshold': get('regime_exit_threshold', -0.02),
        },
        'vol_adjustment': {
            'enabled': get('vol_adjustment_enabled', False),
            'lookback': get('vol_lookback', 20),
            'high_vol_threshold': get('high_vol_threshold', 0.25),
            'low_vol_threshold': get('low_vol_threshold', 0.15),
            'high_vol_multiplier': get('high_vol_multiplier', 1.5),
            'low_vol_multiplier': get('low_vol_multiplier', 0.8),
        }
    }
    
    # Build TP_CONFIG/SL_CONFIG (used only when tpsl_mode == 'fixed')
    tp_thresholds = get('fixed_tp_thresholds', [0.05, 0.05, 0.05])
    sl_thresholds = get('fixed_sl_thresholds', [0.05, 0.05, 0.05])
    
    config['TP_CONFIG'], config['SL_CONFIG'] = _build_tpsl_config(config['category_scheme'], tp_thresholds, sl_thresholds)
    
    return config


def compute_cagr(daily_pf_values: pd.DataFrame) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR) from daily portfolio values.
    
    CAGR = (Final Value / Initial Value) ^ (1 / Years) - 1
    
    Parameters:
        daily_pf_values: DataFrame with columns ['date', 'portfolio_value', 'quarter']
        
    Returns:
        Annualized CAGR as a float, or None if computation fails
    """
    if daily_pf_values is None or daily_pf_values.empty:
        return None
    
    df = daily_pf_values.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ignore_index=True)
    
    if len(df) < 2:
        return None
    
    # Store initial value and final_value
    initial_value = df['portfolio_value'].iloc[0]
    final_value = df['portfolio_value'].iloc[-1]
    
    if initial_value <= 0 or final_value <= 0:
        return None
    
    # Compute years
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    years = (end_date - start_date).days / 365.25
    
    if years <= 0:
        return None
    
    # Compute CAGR
    cagr = (final_value / initial_value) ** (1 / years) - 1
    
    return cagr


def compute_max_drawdown(daily_pf_values: pd.DataFrame) -> float:
    """
    Compute Maximum Drawdown (MDD) from daily portfolio values.
    
    MDD = min((Portfolio Value - Cumulative Max) / Cumulative Max)
    
    Returns the magnitude (positive value) of the maximum drawdown.
    
    Parameters:
        daily_pf_values: DataFrame with columns ['date', 'portfolio_value', 'quarter']
        
    Returns:
        Maximum drawdown as a positive float, or None if computation fails
    """
    if daily_pf_values is None or daily_pf_values.empty:
        return None
    
    df = daily_pf_values.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ignore_index=True)
    
    if len(df) < 2:
        return None
    
    # Compute Maximum Drawdown
    df['cummax'] = df['portfolio_value'].cummax()
    df['drawdown'] = (df['portfolio_value'] - df['cummax']) / df['cummax']
    max_drawdown = df['drawdown'].min()
    
    # Return magnitude (positive value)
    return abs(max_drawdown)


def compute_calmar_ratio(daily_pf_values: pd.DataFrame, cap: float = 10.0) -> float:
    """
    Compute Calmar ratio from daily portfolio values.
    
    Calmar Ratio = CAGR / |Max Drawdown|
    
    Parameters:
        daily_pf_values: DataFrame with columns ['date', 'portfolio_value', 'quarter']
        cap: Maximum Calmar ratio value to return (avoids inflated values)
        
    Returns:
        Calmar ratio (capped)
    """
    if daily_pf_values is None or daily_pf_values.empty:
        return None
    
    if len(daily_pf_values) < 2:
        return None
    
    # Compute CAGR using dedicated function
    cagr = compute_cagr(daily_pf_values)
    if cagr is None:
        return None
    
    # Compute Maximum Drawdown using dedicated function
    max_drawdown = compute_max_drawdown(daily_pf_values)
    if max_drawdown is None:
        return None
    
    # Compute Calmar Ratio
    if max_drawdown == 0:
        # cap at maximum
        calmar = cap
    else:
        calmar = cagr / max_drawdown
    
    # Clip
    calmar = min(calmar, cap)
    
    return calmar


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
        cap = kwargs.get('cap', 10.0)
        return compute_calmar_ratio(daily_pf_values, cap=cap)
    
    elif objective_name == 'cagr':
        return compute_cagr(daily_pf_values)
    
    elif objective_name == 'mdd':
        return compute_max_drawdown(daily_pf_values)
    
    else:
        raise ValueError(
            f"Unknown objective: '{objective_name}'. "
            f"Supported objectives: 'calmar', 'cagr', 'mdd'"
        )


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
    
    # Add report generation flags (enable for final run)
    config['generate_analysis_report'] = True
    config['generate_detailed_report'] = True
    
    # Write to YAML
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Best configuration exported to: {output_path}")


def get_study_summary(study: optuna.Study) -> dict:
    """
    Get summary statistics from an Optuna study.
    
    Parameters:
        study: Completed Optuna study
        
    Returns:
        Dictionary with study summary statistics
    """
    trials_df = study.trials_dataframe()
    
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
    
    summary = {
        'study_name': study.study_name,
        'direction': study.direction.name,
        'n_trials': len(study.trials),
        'n_completed': len(completed_trials),
        'n_pruned': len(trials_df[trials_df['state'] == 'PRUNED']),
        'n_failed': len(trials_df[trials_df['state'] == 'FAIL']),
        'best_value': study.best_value if study.best_trial else None,
        'best_trial_number': study.best_trial.number if study.best_trial else None,
    }
    
    if not completed_trials.empty:
        summary['mean_value'] = completed_trials['value'].mean()
        summary['std_value'] = completed_trials['value'].std()
        summary['min_value'] = completed_trials['value'].min()
        summary['max_value'] = completed_trials['value'].max()
    
    return summary
