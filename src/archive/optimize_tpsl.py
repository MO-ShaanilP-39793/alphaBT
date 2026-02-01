"""
TP/SL Optimization Module

This module provides grid search functionality to find optimal TP/SL parameters
by running multiple backtests and evaluating performance metrics.

Supports three optimization modes:
- 'fixed': Optimize static TP/SL percentages per category
- 'atr': Optimize ATR multipliers (volatility-adaptive thresholds)
- 'pivot': Optimize pivot point levels (support/resistance-based)

Usage Examples:
    # Fixed mode (default) - optimize static percentages
    python optimize_tpsl.py --mode fixed --tp-range 0.02 0.05 0.08 0.10 --sl-range 0.02 0.05 0.08
    
    # ATR mode - optimize volatility-adaptive multipliers
    python optimize_tpsl.py --mode atr --tp-mult-range 1.5 2.0 2.5 3.0 --sl-mult-range 1.0 1.5 2.0
    
    # Pivot mode - optimize support/resistance levels
    python optimize_tpsl.py --mode pivot --tp-levels R1 R2 R3 --sl-levels S1 S2 S3
"""

import pandas as pd
import numpy as np
import yaml
import os
import itertools
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules
from stock_selection import select_and_weight_stocks_volatility, select_and_weight_stocks_mcap
from TP_SL_bt import simulate_trades
from simulate_quarter import compute_pf_value_over_quarters
from addtl_bt_fns import compute_portfolio_metrics


def load_config(config_path='strategy_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_data(file_path):
    """Load data from CSV or Parquet file."""
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def filter_data_by_quarters(df, first_quarter, last_quarter):
    """Filter dataframe to include only rows within the specified quarter range."""
    if first_quarter is None and last_quarter is None:
        return df.copy()
    return df[(df['quarter'] >= first_quarter) & (df['quarter'] <= last_quarter)].copy()


def run_stock_selection(input_data, category_scheme, category_counts, category_weights,
                        selection_method='probability', min_prob_threshold=None):
    """Run stock selection based on the specified category scheme."""
    if category_scheme == 'volatility':
        return select_and_weight_stocks_volatility(
            input_data, 
            selection_counts=category_counts, 
            category_weights=category_weights,
            selection_method=selection_method,
            min_prob_threshold=min_prob_threshold
        )
    elif category_scheme == 'mcap':
        return select_and_weight_stocks_mcap(
            input_data, 
            lms_count=category_counts, 
            lms_w=category_weights,
            selection_method=selection_method,
            min_prob_threshold=min_prob_threshold
        )
    else:
        raise ValueError(f"Unknown category_scheme: {category_scheme}")


def build_tp_sl_config(category_scheme, tp_value, sl_value):
    """
    Build TP/SL config dictionaries for a given set of values.
    
    Applies the same TP/SL percentage to all categories for simplicity.
    For more granular optimization, this can be extended.
    
    Parameters:
    - category_scheme: 'volatility' or 'mcap'
    - tp_value: Take profit percentage (e.g., 0.05 for 5%)
    - sl_value: Stop loss percentage (e.g., 0.05 for 5%)
    
    Returns:
    - Tuple of (tp_config, sl_config)
    """
    if category_scheme == 'volatility':
        categories = ['high_volatility', 'medium_volatility', 'low_volatility']
    else:  # mcap
        categories = ['largecap', 'midcap', 'smallcap']
    
    tp_config = {category_scheme: {cat: tp_value for cat in categories}}
    sl_config = {category_scheme: {cat: sl_value for cat in categories}}
    
    return tp_config, sl_config


def build_atr_config(tp_multiplier, sl_multiplier, period=14):
    """
    Build ATR config dictionary for optimization.
    
    Parameters:
    - tp_multiplier: TP = entry_price + (tp_multiplier × ATR)
    - sl_multiplier: SL = entry_price - (sl_multiplier × ATR)
    - period: ATR calculation period in days
    
    Returns:
    - ATR config dict
    """
    return {
        'period': period,
        'tp_multiplier': tp_multiplier,
        'sl_multiplier': sl_multiplier
    }


def build_pivot_config(tp_level, sl_level, lookback_days=60):
    """
    Build pivot config dictionary for optimization.
    
    Parameters:
    - tp_level: Resistance level for TP ('R1', 'R2', or 'R3')
    - sl_level: Support level for SL ('S1', 'S2', or 'S3')
    - lookback_days: Days of price history for pivot calculation
    
    Returns:
    - Pivot config dict
    """
    return {
        'lookback_days': lookback_days,
        'tp_level': tp_level,
        'sl_level': sl_level
    }


def evaluate_single_combination(args):
    """
    Evaluate a single TP/SL parameter combination.
    
    This function is designed to be called in parallel.
    Supports fixed, ATR, and pivot modes.
    
    Parameters:
    - args: Tuple containing mode-specific parameters:
        - For fixed: (mode, tp_value, sl_value, selected_stocks, price_data, 
                      category_scheme, first_quarter, last_quarter, initial_capital)
        - For atr: (mode, tp_mult, sl_mult, period, selected_stocks, price_data,
                    category_scheme, first_quarter, last_quarter, initial_capital)
        - For pivot: (mode, tp_level, sl_level, lookback_days, selected_stocks, price_data,
                      category_scheme, first_quarter, last_quarter, initial_capital)
    
    Returns:
    - Dictionary with results
    """
    mode = args[0]
    
    try:
        if mode == 'fixed':
            (_, tp_value, sl_value, selected_stocks, price_data, category_scheme,
             first_quarter, last_quarter, initial_capital) = args
            
            # Build config for this combination
            tp_config, sl_config = build_tp_sl_config(category_scheme, tp_value, sl_value)
            
            # Run trade simulation
            portfolio_results = simulate_trades(
                selected_stocks.copy(),
                price_data.copy(),
                category_scheme,
                index_data=None,
                tpsl_mode='fixed',
                custom_tp_config=tp_config,
                custom_sl_config=sl_config
            )
            
            result_params = {'tp': tp_value, 'sl': sl_value}
            
        elif mode == 'atr':
            (_, tp_mult, sl_mult, period, selected_stocks, price_data, category_scheme,
             first_quarter, last_quarter, initial_capital) = args
            
            # Build ATR config
            atr_config = build_atr_config(tp_mult, sl_mult, period)
            
            # Run trade simulation
            portfolio_results = simulate_trades(
                selected_stocks.copy(),
                price_data.copy(),
                category_scheme,
                index_data=None,
                tpsl_mode='atr',
                custom_atr_config=atr_config
            )
            
            result_params = {'tp_mult': tp_mult, 'sl_mult': sl_mult, 'atr_period': period}
            
        elif mode == 'pivot':
            (_, tp_level, sl_level, lookback_days, selected_stocks, price_data, category_scheme,
             first_quarter, last_quarter, initial_capital) = args
            
            # Build pivot config
            pivot_config = build_pivot_config(tp_level, sl_level, lookback_days)
            
            # Run trade simulation
            portfolio_results = simulate_trades(
                selected_stocks.copy(),
                price_data.copy(),
                category_scheme,
                index_data=None,
                tpsl_mode='pivot',
                custom_pivot_config=pivot_config
            )
            
            result_params = {'tp_level': tp_level, 'sl_level': sl_level, 'lookback_days': lookback_days}
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Compute equity curve
        equity_curve = compute_pf_value_over_quarters(
            portfolio_results, 
            price_data.copy(), 
            first_quarter, 
            last_quarter, 
            initial_capital
        )
        
        if equity_curve is None or equity_curve.empty:
            return {
                **result_params,
                'total_return': None, 'sharpe': None, 'sortino': None,
                'max_drawdown': None, 'calmar': None, 'win_rate': None,
                'avg_holding_period': None, 'tp_hit_rate': None, 'sl_hit_rate': None,
                'error': 'Empty equity curve'
            }
        
        # Prepare daily_pf for metrics calculation
        daily_pf = equity_curve.reset_index()
        daily_pf.columns = ['date', 'portfolio_value', 'quarter']
        
        # Compute metrics
        metrics = compute_portfolio_metrics(daily_pf)
        
        # Calculate additional trade-level metrics
        valid_trades = portfolio_results[portfolio_results['holding_period'].notna()]
        total_trades = len(valid_trades)
        
        if total_trades > 0:
            winning_trades = (valid_trades['stock_return'] > 0).sum()
            win_rate = winning_trades / total_trades
            avg_holding_period = valid_trades['holding_period'].mean()
            tp_hit_rate = valid_trades['TP_triggered'].sum() / total_trades
            sl_hit_rate = valid_trades['SL_triggered'].sum() / total_trades
        else:
            win_rate = None
            avg_holding_period = None
            tp_hit_rate = None
            sl_hit_rate = None
        
        return {
            **result_params,
            'total_return': metrics.get('total_return_pct'),
            'cagr': metrics.get('cagr_pct'),
            'sharpe': metrics.get('sharpe_ratio'),
            'sortino': metrics.get('sortino_ratio'),
            'max_drawdown': metrics.get('max_drawdown_pct'),
            'calmar': metrics.get('calmar_ratio'),
            'volatility': metrics.get('volatility_pct'),
            'win_rate': round(win_rate * 100, 2) if win_rate else None,
            'avg_holding_period': round(avg_holding_period, 1) if avg_holding_period else None,
            'tp_hit_rate': round(tp_hit_rate * 100, 2) if tp_hit_rate else None,
            'sl_hit_rate': round(sl_hit_rate * 100, 2) if sl_hit_rate else None,
            'total_trades': total_trades,
            'error': None
        }
        
    except Exception as e:
        # Return error result with appropriate params based on mode
        if mode == 'fixed':
            result_params = {'tp': args[1], 'sl': args[2]}
        elif mode == 'atr':
            result_params = {'tp_mult': args[1], 'sl_mult': args[2], 'atr_period': args[3]}
        elif mode == 'pivot':
            result_params = {'tp_level': args[1], 'sl_level': args[2], 'lookback_days': args[3]}
        else:
            result_params = {}
            
        return {
            **result_params,
            'total_return': None, 'sharpe': None, 'sortino': None,
            'max_drawdown': None, 'calmar': None, 'win_rate': None,
            'avg_holding_period': None, 'tp_hit_rate': None, 'sl_hit_rate': None,
            'error': str(e)
        }


def run_tpsl_optimization(mode: str = 'fixed',
                           tp_range: list = None, 
                           sl_range: list = None,
                           tp_mult_range: list = None,
                           sl_mult_range: list = None,
                           atr_period: int = 14,
                           tp_levels: list = None,
                           sl_levels: list = None,
                           lookback_days: int = 60,
                           config_path: str = 'strategy_config.yaml',
                           metric: str = 'sharpe',
                           parallel: bool = True,
                           max_workers: int = None) -> pd.DataFrame:
    """
    Run grid search optimization over TP/SL parameter space.
    
    Supports three modes:
    - 'fixed': Optimize static TP/SL percentages
    - 'atr': Optimize ATR multipliers
    - 'pivot': Optimize pivot point levels
    
    Parameters:
    - mode: 'fixed', 'atr', or 'pivot'
    
    Fixed mode parameters:
    - tp_range: List of TP percentages (e.g., [0.02, 0.05, 0.08, 0.10])
    - sl_range: List of SL percentages (e.g., [0.02, 0.05, 0.08])
    
    ATR mode parameters:
    - tp_mult_range: List of TP multipliers (e.g., [1.5, 2.0, 2.5, 3.0])
    - sl_mult_range: List of SL multipliers (e.g., [1.0, 1.5, 2.0])
    - atr_period: ATR calculation period (default: 14)
    
    Pivot mode parameters:
    - tp_levels: List of TP levels (e.g., ['R1', 'R2', 'R3'])
    - sl_levels: List of SL levels (e.g., ['S1', 'S2', 'S3'])
    - lookback_days: Days for pivot calculation (default: 60)
    
    Common parameters:
    - config_path: Path to strategy configuration file
    - metric: Metric to optimize ('sharpe', 'sortino', 'total_return', 'calmar', 'win_rate')
    - parallel: Whether to run evaluations in parallel
    - max_workers: Max parallel workers (default: CPU count - 1)
    
    Returns:
    - DataFrame with results for all combinations
    """
    print("=" * 60)
    print(f"TP/SL PARAMETER OPTIMIZATION - {mode.upper()} MODE")
    print("=" * 60)
    
    # Load configuration and data
    print("\n[1/4] Loading configuration and data...")
    config = load_config(config_path)
    
    input_data_path = config['input_data_path']
    price_data_path = config['price_data_path']
    first_quarter = config['first_quarter']
    last_quarter = config['last_quarter']
    category_scheme = config['category_scheme']
    category_counts = config['category_counts']
    category_weights = config['category_weights']
    selection_method = config.get('selection_method', 'probability')
    min_prob_threshold = config.get('min_prob_threshold', None)
    
    input_data = load_data(input_data_path)
    price_data = load_data(price_data_path)
    
    # Filter data
    print("\n[2/4] Running stock selection...")
    input_data_filtered = filter_data_by_quarters(input_data, first_quarter, last_quarter)
    
    selected_stocks = run_stock_selection(
        input_data_filtered,
        category_scheme,
        category_counts,
        category_weights,
        selection_method=selection_method,
        min_prob_threshold=min_prob_threshold
    )
    print(f"  - Selected {len(selected_stocks)} positions")
    
    initial_capital = 1_000_000_000
    
    # Generate combinations based on mode
    print("\n[3/4] Running optimization...")
    
    if mode == 'fixed':
        if tp_range is None:
            tp_range = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
        if sl_range is None:
            sl_range = [0.02, 0.04, 0.06, 0.08, 0.10]
            
        combinations = list(itertools.product(tp_range, sl_range))
        print(f"  - Testing {len(combinations)} fixed TP/SL combinations")
        print(f"  - TP range: {tp_range}")
        print(f"  - SL range: {sl_range}")
        
        eval_args = [
            ('fixed', tp, sl, selected_stocks.copy(), price_data.copy(), category_scheme,
             first_quarter, last_quarter, initial_capital)
            for tp, sl in combinations
        ]
        
    elif mode == 'atr':
        if tp_mult_range is None:
            tp_mult_range = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        if sl_mult_range is None:
            sl_mult_range = [1.0, 1.5, 2.0, 2.5, 3.0]
            
        combinations = list(itertools.product(tp_mult_range, sl_mult_range))
        print(f"  - Testing {len(combinations)} ATR multiplier combinations")
        print(f"  - TP multiplier range: {tp_mult_range}")
        print(f"  - SL multiplier range: {sl_mult_range}")
        print(f"  - ATR period: {atr_period}")
        
        eval_args = [
            ('atr', tp_mult, sl_mult, atr_period, selected_stocks.copy(), price_data.copy(),
             category_scheme, first_quarter, last_quarter, initial_capital)
            for tp_mult, sl_mult in combinations
        ]
        
    elif mode == 'pivot':
        if tp_levels is None:
            tp_levels = ['R1', 'R2', 'R3']
        if sl_levels is None:
            sl_levels = ['S1', 'S2', 'S3']
            
        combinations = list(itertools.product(tp_levels, sl_levels))
        print(f"  - Testing {len(combinations)} pivot level combinations")
        print(f"  - TP levels: {tp_levels}")
        print(f"  - SL levels: {sl_levels}")
        print(f"  - Lookback days: {lookback_days}")
        
        eval_args = [
            ('pivot', tp_level, sl_level, lookback_days, selected_stocks.copy(), price_data.copy(),
             category_scheme, first_quarter, last_quarter, initial_capital)
            for tp_level, sl_level in combinations
        ]
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'fixed', 'atr', or 'pivot'")
    
    total_combinations = len(combinations)
    
    # Run evaluations
    results = []
    
    if parallel and total_combinations > 1:
        workers = max_workers or max(1, os.cpu_count() - 1)
        print(f"  - Using {workers} parallel workers")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_single_combination, args): args 
                      for args in eval_args}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                results.append(result)
                
                # Progress update
                if completed % 5 == 0 or completed == total_combinations:
                    print(f"  - Progress: {completed}/{total_combinations} combinations evaluated")
    else:
        for i, args in enumerate(eval_args):
            result = evaluate_single_combination(args)
            results.append(result)
            
            if (i + 1) % 5 == 0 or (i + 1) == total_combinations:
                print(f"  - Progress: {i + 1}/{total_combinations} combinations evaluated")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by chosen metric
    if metric in results_df.columns:
        results_df = results_df.sort_values(metric, ascending=False, na_position='last')
    
    print("\n[4/4] Optimization complete!")
    
    return results_df


def find_optimal_params(results_df: pd.DataFrame, metric: str = 'sharpe', mode: str = 'fixed') -> dict:
    """
    Find the optimal TP/SL parameters from optimization results.
    
    Parameters:
    - results_df: DataFrame from run_tpsl_optimization
    - metric: Metric to optimize by
    - mode: 'fixed', 'atr', or 'pivot' to determine which columns to return
    
    Returns:
    - Dictionary with optimal parameters and their metrics
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results")
    
    # Filter out rows with errors or missing metric
    valid_results = results_df[results_df[metric].notna() & results_df['error'].isna()]
    
    if valid_results.empty:
        return {'error': 'No valid results found'}
    
    # Find best row
    best_idx = valid_results[metric].idxmax()
    best_row = valid_results.loc[best_idx]
    
    # Build result based on mode
    result = {
        'mode': mode,
        'metric_name': metric,
        'metric_value': best_row[metric],
        'total_return': best_row.get('total_return'),
        'sharpe': best_row.get('sharpe'),
        'sortino': best_row.get('sortino'),
        'max_drawdown': best_row.get('max_drawdown'),
        'win_rate': best_row.get('win_rate'),
        'tp_hit_rate': best_row.get('tp_hit_rate'),
        'sl_hit_rate': best_row.get('sl_hit_rate')
    }
    
    if mode == 'fixed':
        result['optimal_tp'] = best_row['tp']
        result['optimal_sl'] = best_row['sl']
    elif mode == 'atr':
        result['optimal_tp_mult'] = best_row['tp_mult']
        result['optimal_sl_mult'] = best_row['sl_mult']
        result['atr_period'] = best_row.get('atr_period')
    elif mode == 'pivot':
        result['optimal_tp_level'] = best_row['tp_level']
        result['optimal_sl_level'] = best_row['sl_level']
        result['lookback_days'] = best_row.get('lookback_days')
    
    return result


def generate_optimization_heatmap(results_df: pd.DataFrame, metric: str = 'sharpe',
                                   mode: str = 'fixed', save_path: str = None, 
                                   figsize: tuple = (12, 10)):
    """
    Generate a heatmap visualization of optimization results.
    
    Parameters:
    - results_df: DataFrame from run_tpsl_optimization
    - metric: Metric to visualize
    - mode: 'fixed', 'atr', or 'pivot'
    - save_path: Path to save the figure (if None, displays plot)
    - figsize: Figure size tuple
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results")
    
    # Determine column names based on mode
    if mode == 'fixed':
        x_col, y_col = 'tp', 'sl'
        x_label = 'Take Profit (%)'
        y_label = 'Stop Loss (%)'
        title_prefix = 'Fixed TP/SL'
    elif mode == 'atr':
        x_col, y_col = 'tp_mult', 'sl_mult'
        x_label = 'TP Multiplier (× ATR)'
        y_label = 'SL Multiplier (× ATR)'
        title_prefix = 'ATR-Based TP/SL'
    elif mode == 'pivot':
        x_col, y_col = 'tp_level', 'sl_level'
        x_label = 'TP Level (Resistance)'
        y_label = 'SL Level (Support)'
        title_prefix = 'Pivot-Based TP/SL'
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Create pivot table for heatmap
    pivot = results_df.pivot(index=y_col, columns=x_col, values=metric)
    
    # Sort axes appropriately
    if mode == 'pivot':
        # For pivot mode, use natural ordering (R1, R2, R3 and S1, S2, S3)
        tp_order = ['R1', 'R2', 'R3']
        sl_order = ['S3', 'S2', 'S1']  # Reversed for visual clarity (S1 at bottom)
        pivot = pivot.reindex(index=[s for s in sl_order if s in pivot.index])
        pivot = pivot.reindex(columns=[r for r in tp_order if r in pivot.columns])
    else:
        # For numeric modes, sort by value
        pivot = pivot.sort_index(ascending=False)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colormap (higher is better for most metrics)
    cmap = 'RdYlGn'
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap, center=None,
                ax=ax, cbar_kws={'label': metric.replace('_', ' ').title()})
    
    # Mark the optimal cell
    valid_results = results_df[results_df[metric].notna()]
    if not valid_results.empty:
        if metric == 'max_drawdown':
            best_idx = valid_results[metric].idxmax()  # Least negative = best
        else:
            best_idx = valid_results[metric].idxmax()
        
        best_row = valid_results.loc[best_idx]
        best_x = best_row[x_col]
        best_y = best_row[y_col]
        
        # Find position in heatmap
        x_list = list(pivot.columns)
        y_list = list(pivot.index)
        
        if best_x in x_list and best_y in y_list:
            x_pos = x_list.index(best_x)
            y_pos = y_list.index(best_y)
            
            # Add star marker for optimal
            ax.plot(x_pos + 0.5, y_pos + 0.5, marker='*', markersize=20, 
                   color='gold', markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{title_prefix} Optimization: {metric.replace("_", " ").title()}\n(Star = Optimal)', fontsize=14)
    
    # Format tick labels based on mode
    if mode == 'fixed':
        ax.set_xticklabels([f'{float(x.get_text())*100:.0f}%' for x in ax.get_xticklabels()])
        ax.set_yticklabels([f'{float(y.get_text())*100:.0f}%' for y in ax.get_yticklabels()])
    elif mode == 'atr':
        ax.set_xticklabels([f'{float(x.get_text()):.1f}x' for x in ax.get_xticklabels()])
        ax.set_yticklabels([f'{float(y.get_text()):.1f}x' for y in ax.get_yticklabels()])
    # For pivot mode, labels are already categorical (R1, R2, etc.)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to: {save_path}")
    else:
        plt.show()


def generate_optimization_report(results_df: pd.DataFrame, output_dir: str, mode: str = 'fixed'):
    """
    Generate a comprehensive optimization report with multiple visualizations.
    
    Parameters:
    - results_df: DataFrame from run_tpsl_optimization
    - output_dir: Directory to save outputs
    - mode: 'fixed', 'atr', or 'pivot'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    results_path = os.path.join(output_dir, 'optimization_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    # Generate heatmaps for key metrics
    metrics = ['sharpe', 'total_return', 'sortino', 'win_rate', 'calmar']
    
    for metric in metrics:
        if metric in results_df.columns and results_df[metric].notna().any():
            heatmap_path = os.path.join(output_dir, f'heatmap_{metric}.png')
            try:
                generate_optimization_heatmap(results_df, metric, mode=mode, save_path=heatmap_path)
            except Exception as e:
                print(f"Warning: Could not generate heatmap for {metric}: {e}")
    
    # Find optimal parameters for each metric
    optimal_params = {}
    for metric in metrics:
        if metric in results_df.columns and results_df[metric].notna().any():
            optimal_params[metric] = find_optimal_params(results_df, metric, mode=mode)
    
    # Save optimal parameters summary
    summary_path = os.path.join(output_dir, 'optimal_params_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(optimal_params, f, default_flow_style=False)
    print(f"Optimal parameters saved to: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"OPTIMIZATION SUMMARY ({mode.upper()} MODE)")
    print("=" * 60)
    
    for metric, params in optimal_params.items():
        if 'error' not in params:
            print(f"\nOptimal for {metric.upper()}:")
            
            # Print mode-specific parameters
            if mode == 'fixed':
                print(f"  TP: {params['optimal_tp']*100:.1f}%  |  SL: {params['optimal_sl']*100:.1f}%")
            elif mode == 'atr':
                print(f"  TP Mult: {params['optimal_tp_mult']:.1f}x  |  SL Mult: {params['optimal_sl_mult']:.1f}x")
            elif mode == 'pivot':
                print(f"  TP Level: {params['optimal_tp_level']}  |  SL Level: {params['optimal_sl_level']}")
            
            print(f"  {metric}: {params['metric_value']:.3f}")
            if params.get('total_return'):
                print(f"  Total Return: {params['total_return']:.2f}%")
            if params.get('max_drawdown'):
                print(f"  Max Drawdown: {params['max_drawdown']:.2f}%")
    
    return optimal_params


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TP/SL Parameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fixed mode (default)
  python optimize_tpsl.py --mode fixed --tp-range 0.02 0.05 0.08 0.10 --sl-range 0.02 0.05 0.08
  
  # ATR mode
  python optimize_tpsl.py --mode atr --tp-mult-range 1.5 2.0 2.5 3.0 --sl-mult-range 1.0 1.5 2.0
  
  # Pivot mode
  python optimize_tpsl.py --mode pivot --tp-levels R1 R2 R3 --sl-levels S1 S2 S3
        """
    )
    
    # Common arguments
    parser.add_argument('--config', default='strategy_config.yaml', help='Path to config file')
    parser.add_argument('--mode', default=None, choices=['fixed', 'atr', 'pivot'],
                       help='TP/SL mode to optimize (default: fixed)')
    parser.add_argument('--metric', default=None, 
                       choices=['sharpe', 'sortino', 'total_return', 'calmar', 'win_rate'],
                       help='Metric to optimize (default: sharpe)')
    parser.add_argument('--output-dir', default='../optimization_results',
                       help='Output directory for results')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel execution')
    
    # Fixed mode arguments
    parser.add_argument('--tp-range', nargs='+', type=float,
                       help='[Fixed mode] TP percentages (e.g., 0.02 0.05 0.08 0.10)')
    parser.add_argument('--sl-range', nargs='+', type=float,
                       help='[Fixed mode] SL percentages (e.g., 0.02 0.05 0.08)')
    
    # ATR mode arguments
    parser.add_argument('--tp-mult-range', nargs='+', type=float,
                       help='[ATR mode] TP multipliers (e.g., 1.5 2.0 2.5 3.0)')
    parser.add_argument('--sl-mult-range', nargs='+', type=float,
                       help='[ATR mode] SL multipliers (e.g., 1.0 1.5 2.0)')
    parser.add_argument('--atr-period', type=int, default=None,
                       help='[ATR mode] ATR calculation period (default: 14)')
    
    # Pivot mode arguments
    parser.add_argument('--tp-levels', nargs='+', type=str,
                       help='[Pivot mode] TP resistance levels (e.g., R1 R2 R3)')
    parser.add_argument('--sl-levels', nargs='+', type=str,
                       help='[Pivot mode] SL support levels (e.g., S1 S2 S3)')
    parser.add_argument('--lookback-days', type=int, default=None,
                       help='[Pivot mode] Lookback days for pivot calculation (default: 60)')
    
    args = parser.parse_args()
    
    # Load config to check for optimization settings
    try:
        config = load_config(args.config)
        opt_config = config.get('optimization_config', {})
    except Exception as e:
        print(f"Warning: Could not load config file to check for optimization settings: {e}")
        opt_config = {}
        
    # Resolve mode
    if args.mode is None:
        args.mode = opt_config.get('mode', 'fixed')
    
    # Resolve metric
    if args.metric is None:
        args.metric = opt_config.get('metric', 'sharpe')
        
    # Resolve parameters based on mode
    mode_config = opt_config.get(args.mode, {})
    
    if args.mode == 'fixed':
        if args.tp_range is None:
            args.tp_range = mode_config.get('tp_range')
        if args.sl_range is None:
            args.sl_range = mode_config.get('sl_range')
            
    elif args.mode == 'atr':
        if args.tp_mult_range is None:
            args.tp_mult_range = mode_config.get('tp_mult_range')
        if args.sl_mult_range is None:
            args.sl_mult_range = mode_config.get('sl_mult_range')
        if args.atr_period is None:
            args.atr_period = mode_config.get('atr_period', 14)
            
    elif args.mode == 'pivot':
        if args.tp_levels is None:
            args.tp_levels = mode_config.get('tp_levels')
        if args.sl_levels is None:
            args.sl_levels = mode_config.get('sl_levels')
        if args.lookback_days is None:
            args.lookback_days = mode_config.get('lookback_days', 60)
            
    # Ensure defaults are set if still None (fallback to hardcoded defaults in run_tpsl_optimization)
    if args.atr_period is None: args.atr_period = 14
    if args.lookback_days is None: args.lookback_days = 60
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'opt_{args.mode}_{timestamp}')
    
    # Run optimization based on mode
    results_df = run_tpsl_optimization(
        mode=args.mode,
        # Fixed mode params
        tp_range=args.tp_range,
        sl_range=args.sl_range,
        # ATR mode params
        tp_mult_range=args.tp_mult_range,
        sl_mult_range=args.sl_mult_range,
        atr_period=args.atr_period,
        # Pivot mode params
        tp_levels=args.tp_levels,
        sl_levels=args.sl_levels,
        lookback_days=args.lookback_days,
        # Common params
        config_path=args.config,
        metric=args.metric,
        parallel=not args.no_parallel
    )
    
    # Generate report
    optimal_params = generate_optimization_report(results_df, output_dir, mode=args.mode)
    
    print(f"\nAll outputs saved to: {output_dir}")
    
    return results_df, optimal_params


if __name__ == '__main__':
    results, optimal = main()
