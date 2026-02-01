"""
Strategy Optimization via Grid Search

This script performs grid search over the strategy parameter space,
running backtests for each combination and computing comprehensive metrics.

Usage:
    python optimize_strategy.py                          # Uses default config
    python optimize_strategy.py --config my_grid.yaml    # Custom config file

Output:
    - CSV with all results and metrics
    - Summary report with top configurations
"""

import pandas as pd
import numpy as np
import yaml
import os
import warnings
import itertools
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Import backtest components
from stock_selection import select_and_weight_stocks_volatility, select_and_weight_stocks_mcap
from TP_SL_bt import simulate_trades
from simulate_quarter import compute_pf_value_over_quarters, compute_pf_vs_index
from addtl_bt_fns import compute_portfolio_metrics, compute_benchmark_metrics

# Suppress warnings during optimization (they can be noisy)
warnings.filterwarnings('ignore')


def load_grid_config(config_path: str = 'strategy_grid_config.yaml') -> dict:
    """Load the grid search configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required data files."""
    def read_file(path):
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    
    input_data = read_file(config['input_data_path'])
    price_data = read_file(config['price_data_path'])
    index_data = read_file(config['index_data_path'])
    
    return input_data, price_data, index_data


def filter_by_quarters(df: pd.DataFrame, first_q: int, last_q: int) -> pd.DataFrame:
    """Filter dataframe by quarter range."""
    if first_q is None and last_q is None:
        return df
    return df[(df['quarter'] >= first_q) & (df['quarter'] <= last_q)].copy()


def generate_parameter_combinations(config: dict) -> List[Dict[str, Any]]:
    """
    Generate all valid parameter combinations from the grid config.
    
    Returns a list of dictionaries, each representing one parameter combination.
    """
    grid = config['parameter_grid']
    constraints = config.get('constraints', {})
    
    # Base parameters (always present)
    base_params = {
        'category_scheme': grid['category_scheme'],
        'category_counts': grid['category_counts'],
        'category_weights': grid['category_weights'],
        'selection_method': grid['selection_method'],
        'min_prob_threshold': grid['min_prob_threshold'],
        'tpsl_mode': grid['tpsl_mode']
    }
    
    # Generate all base combinations
    base_keys = list(base_params.keys())
    base_values = [base_params[k] for k in base_keys]
    base_combos = list(itertools.product(*base_values))
    
    all_combinations = []
    
    for combo in base_combos:
        params = dict(zip(base_keys, combo))
        
        # Apply constraints
        weights = params['category_weights']
        counts = params['category_counts']
        
        # Check weights sum
        weights_sum_tol = constraints.get('weights_sum_tolerance', 0.01)
        if abs(sum(weights) - 1.0) > weights_sum_tol:
            continue
        
        # Check stock counts
        total_stocks = sum(counts)
        min_stocks = constraints.get('min_total_stocks', 0)
        max_stocks = constraints.get('max_total_stocks', 999)
        if total_stocks < min_stocks or total_stocks > max_stocks:
            continue
        
        # Add mode-specific parameters
        tpsl_mode = params['tpsl_mode']
        
        if tpsl_mode == 'fixed':
            fixed_grid = config.get('fixed_mode_grid', {})
            tp_pcts = fixed_grid.get('tp_pct', [[0.08, 0.05, 0.03]])
            sl_pcts = fixed_grid.get('sl_pct', [[0.05, 0.04, 0.03]])
            
            for tp_pct, sl_pct in itertools.product(tp_pcts, sl_pcts):
                full_params = params.copy()
                full_params['tp_pct'] = tp_pct
                full_params['sl_pct'] = sl_pct
                all_combinations.append(full_params)
                
        elif tpsl_mode == 'atr':
            atr_grid = config.get('atr_mode_grid', {})
            periods = atr_grid.get('period', [14])
            tp_mults = atr_grid.get('tp_multiplier', [2.0])
            sl_mults = atr_grid.get('sl_multiplier', [1.5])
            
            for period, tp_mult, sl_mult in itertools.product(periods, tp_mults, sl_mults):
                full_params = params.copy()
                full_params['atr_period'] = period
                full_params['atr_tp_multiplier'] = tp_mult
                full_params['atr_sl_multiplier'] = sl_mult
                all_combinations.append(full_params)
                
        elif tpsl_mode == 'pivot':
            pivot_grid = config.get('pivot_mode_grid', {})
            lookbacks = pivot_grid.get('lookback_days', [60])
            tp_levels = pivot_grid.get('tp_level', ['R1'])
            sl_levels = pivot_grid.get('sl_level', ['S1'])
            
            for lookback, tp_lvl, sl_lvl in itertools.product(lookbacks, tp_levels, sl_levels):
                full_params = params.copy()
                full_params['pivot_lookback'] = lookback
                full_params['pivot_tp_level'] = tp_lvl
                full_params['pivot_sl_level'] = sl_lvl
                all_combinations.append(full_params)
        else:
            # Unknown mode, add without mode-specific params
            all_combinations.append(params.copy())
    
    return all_combinations


def build_fixed_config(params: dict, category_scheme: str) -> Tuple[dict, dict]:
    """
    Build TP_CONFIG and SL_CONFIG dictionaries for fixed mode.
    
    The tp_pct and sl_pct lists are mapped to categories based on scheme:
    - volatility: [high_volatility, medium_volatility, low_volatility]
    - mcap: [largecap, midcap, smallcap]
    """
    tp_pct = params.get('tp_pct', [0.08, 0.05, 0.03])
    sl_pct = params.get('sl_pct', [0.05, 0.04, 0.03])
    
    if category_scheme == 'volatility':
        cats = ['high_volatility', 'medium_volatility', 'low_volatility']
    else:
        cats = ['largecap', 'midcap', 'smallcap']
    
    tp_config = {category_scheme: dict(zip(cats, tp_pct))}
    sl_config = {category_scheme: dict(zip(cats, sl_pct))}
    
    return tp_config, sl_config


def run_single_backtest(
    params: dict,
    input_data: pd.DataFrame,
    price_data: pd.DataFrame,
    index_data: pd.DataFrame,
    first_quarter: int,
    last_quarter: int,
    initial_capital: float,
    risk_free_rate: float
) -> Optional[Dict[str, Any]]:
    """
    Run a single backtest with the given parameters.
    
    Returns a dictionary with all metrics, or None if the backtest fails.
    """
    try:
        # 1. Stock Selection
        category_scheme = params['category_scheme']
        counts = params['category_counts']
        weights = params['category_weights']
        selection_method = params['selection_method']
        min_prob = params['min_prob_threshold']
        
        if category_scheme == 'volatility':
            selected_stocks = select_and_weight_stocks_volatility(
                input_data, counts, weights, selection_method, min_prob
            )
        else:
            selected_stocks = select_and_weight_stocks_mcap(
                input_data, counts, weights, selection_method, min_prob
            )
        
        if selected_stocks.empty:
            return None
        
        # 2. Build mode-specific configs
        tpsl_mode = params['tpsl_mode']
        custom_tp = None
        custom_sl = None
        custom_atr = None
        custom_pivot = None
        
        if tpsl_mode == 'fixed':
            custom_tp, custom_sl = build_fixed_config(params, category_scheme)
        elif tpsl_mode == 'atr':
            custom_atr = {
                'period': params.get('atr_period', 14),
                'tp_multiplier': params.get('atr_tp_multiplier', 2.0),
                'sl_multiplier': params.get('atr_sl_multiplier', 1.5)
            }
        elif tpsl_mode == 'pivot':
            custom_pivot = {
                'lookback_days': params.get('pivot_lookback', 60),
                'tp_level': params.get('pivot_tp_level', 'R1'),
                'sl_level': params.get('pivot_sl_level', 'S1')
            }
        
        # 3. Simulate Trades
        trade_results = simulate_trades(
            selected_stocks,
            price_data,
            category_scheme,
            index_data=index_data,
            tpsl_mode=tpsl_mode,
            custom_tp_config=custom_tp,
            custom_sl_config=custom_sl,
            custom_atr_config=custom_atr,
            custom_pivot_config=custom_pivot
        )
        
        if trade_results.empty:
            return None
        
        # 4. Compute Portfolio Values
        equity_curve = compute_pf_value_over_quarters(
            trade_results, price_data, first_quarter, last_quarter, initial_capital
        )
        
        if equity_curve is None or equity_curve.empty:
            return None
        
        # 5. Format equity curve for metrics computation
        daily_pf = equity_curve.reset_index()
        daily_pf.columns = ['date', 'portfolio_value', 'quarter']
        
        # 6. Compute Portfolio Metrics
        pf_metrics = compute_portfolio_metrics(daily_pf, risk_free_rate)
        
        # 7. Compute Benchmark Comparison
        comparison_df = compute_pf_vs_index(
            trade_results, price_data, index_data,
            first_quarter, last_quarter, initial_capital
        )
        
        benchmark_metrics = {}
        if comparison_df is not None and not comparison_df.empty:
            benchmark_metrics = compute_benchmark_metrics(comparison_df)
        
        # 8. Compute Trade-Level Stats
        valid_trades = trade_results[trade_results['holding_period'].notna()]
        n_trades = len(valid_trades)
        
        if n_trades > 0:
            winning_trades = valid_trades[valid_trades['stock_return'] > 0]
            losing_trades = valid_trades[valid_trades['stock_return'] < 0]
            
            win_rate = len(winning_trades) / n_trades * 100
            avg_win = winning_trades['stock_return'].mean() * 100 if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['stock_return'].mean() * 100 if len(losing_trades) > 0 else 0
            
            # Profit factor
            gross_profit = winning_trades['stock_return'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['stock_return'].sum()) if len(losing_trades) > 0 else 0.0001
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit / 0.0001
            
            # TP/SL stats
            tp_count = valid_trades['TP_triggered'].sum() if 'TP_triggered' in valid_trades.columns else 0
            sl_count = valid_trades['SL_triggered'].sum() if 'SL_triggered' in valid_trades.columns else 0
            time_exits = n_trades - tp_count - sl_count
            
            avg_holding_period = valid_trades['holding_period'].mean()
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            tp_count = sl_count = time_exits = avg_holding_period = 0
        
        trade_metrics = {
            'n_trades': n_trades,
            'win_rate_pct': round(win_rate, 2),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 3),
            'tp_exits': int(tp_count),
            'sl_exits': int(sl_count),
            'time_exits': int(time_exits),
            'avg_holding_days': round(avg_holding_period, 1) if avg_holding_period else 0
        }
        
        # 9. Combine all metrics
        result = {**pf_metrics, **benchmark_metrics, **trade_metrics}
        
        return result
        
    except Exception as e:
        # Log error but don't crash the whole optimization
        print(f"  Error: {str(e)[:100]}")
        return None


def params_to_string(params: dict) -> str:
    """Convert parameter dict to a readable string ID."""
    parts = [
        f"scheme={params['category_scheme']}",
        f"counts={'_'.join(map(str, params['category_counts']))}",
        f"weights={'_'.join(map(str, params['category_weights']))}",
        f"select={params['selection_method']}",
        f"minprob={params['min_prob_threshold']}",
        f"mode={params['tpsl_mode']}"
    ]
    
    if params['tpsl_mode'] == 'fixed':
        parts.append(f"tp={'_'.join(map(str, params.get('tp_pct', [])))}")
        parts.append(f"sl={'_'.join(map(str, params.get('sl_pct', [])))}")
    elif params['tpsl_mode'] == 'atr':
        parts.append(f"atr_p={params.get('atr_period')}")
        parts.append(f"atr_tp={params.get('atr_tp_multiplier')}")
        parts.append(f"atr_sl={params.get('atr_sl_multiplier')}")
    elif params['tpsl_mode'] == 'pivot':
        parts.append(f"pvt_lb={params.get('pivot_lookback')}")
        parts.append(f"pvt_tp={params.get('pivot_tp_level')}")
        parts.append(f"pvt_sl={params.get('pivot_sl_level')}")
    
    return '|'.join(parts)


def run_optimization(config_path: str = 'strategy_grid_config.yaml'):
    """
    Main function to run the full grid search optimization.
    """
    print("=" * 70)
    print("STRATEGY OPTIMIZATION - GRID SEARCH")
    print("=" * 70)
    
    # 1. Load Configuration
    print("\n[1] Loading configuration...")
    config = load_grid_config(config_path)
    
    first_quarter = config['first_quarter']
    last_quarter = config['last_quarter']
    initial_capital = config.get('initial_capital', 1_000_000_000)
    risk_free_rate = config.get('risk_free_rate', 0.065)
    output_config = config.get('output', {})
    output_dir = output_config.get('output_dir', '../optimization_results')
    top_n = output_config.get('top_n', 10)
    
    print(f"  Quarter range: {first_quarter} to {last_quarter}")
    print(f"  Initial capital: ₹{initial_capital/1e7:.0f} Cr")
    print(f"  Risk-free rate: {risk_free_rate*100:.1f}%")
    
    # 2. Load Data
    print("\n[2] Loading data...")
    input_data, price_data, index_data = load_data(config)
    
    # Filter input data by quarters
    input_data_filtered = filter_by_quarters(input_data, first_quarter, last_quarter)
    print(f"  Input data: {len(input_data_filtered)} rows")
    print(f"  Price data: {len(price_data)} rows")
    print(f"  Index data: {len(index_data)} rows")
    
    # 3. Generate Parameter Combinations
    print("\n[3] Generating parameter combinations...")
    combinations = generate_parameter_combinations(config)
    n_combos = len(combinations)
    print(f"  Total combinations to test: {n_combos}")
    
    if n_combos == 0:
        print("  No valid combinations generated. Check your configuration.")
        return
    
    # 4. Run Backtests
    print("\n[4] Running backtests...")
    print("-" * 70)
    
    results = []
    successful = 0
    failed = 0
    
    for i, params in enumerate(combinations):
        progress = (i + 1) / n_combos * 100
        print(f"\r  Progress: {progress:5.1f}% ({i+1}/{n_combos}) | Success: {successful} | Failed: {failed}", end='')
        
        metrics = run_single_backtest(
            params,
            input_data_filtered,
            price_data,
            index_data,
            first_quarter,
            last_quarter,
            initial_capital,
            risk_free_rate
        )
        
        if metrics is not None:
            # Add parameters to result
            result_row = {**params, **metrics}
            result_row['param_id'] = params_to_string(params)
            results.append(result_row)
            successful += 1
        else:
            failed += 1
    
    print(f"\n  Completed: {successful} successful, {failed} failed")
    
    if not results:
        print("\n  No successful backtests. Cannot generate report.")
        return
    
    # 5. Create Results DataFrame
    print("\n[5] Processing results...")
    results_df = pd.DataFrame(results)
    
    # Reorder columns for readability
    param_cols = [
        'param_id', 'category_scheme', 'category_counts', 'category_weights',
        'selection_method', 'min_prob_threshold', 'tpsl_mode'
    ]
    
    # Add mode-specific columns
    if 'tp_pct' in results_df.columns:
        param_cols.extend(['tp_pct', 'sl_pct'])
    if 'atr_period' in results_df.columns:
        param_cols.extend(['atr_period', 'atr_tp_multiplier', 'atr_sl_multiplier'])
    if 'pivot_lookback' in results_df.columns:
        param_cols.extend(['pivot_lookback', 'pivot_tp_level', 'pivot_sl_level'])
    
    metric_cols = [
        # Return metrics
        'total_return_pct', 'cagr_pct', 'alpha_pct',
        # Risk metrics  
        'volatility_pct', 'max_drawdown_pct', 'var_95_pct',
        # Risk-adjusted
        'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio',
        # Benchmark comparison
        'beta', 'correlation', 'up_capture_pct', 'down_capture_pct',
        # Trade metrics
        'n_trades', 'win_rate_pct', 'profit_factor', 'avg_holding_days',
        'tp_exits', 'sl_exits', 'time_exits', 'avg_win_pct', 'avg_loss_pct'
    ]
    
    # Only include columns that exist
    param_cols = [c for c in param_cols if c in results_df.columns]
    metric_cols = [c for c in metric_cols if c in results_df.columns]
    other_cols = [c for c in results_df.columns if c not in param_cols + metric_cols]
    
    results_df = results_df[param_cols + metric_cols + other_cols]
    
    # 6. Save Results
    print("\n[6] Saving results...")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full results
    results_path = os.path.join(output_dir, f'optimization_results_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"  Full results saved to: {results_path}")
    
    # Save config copy
    config_copy_path = os.path.join(output_dir, f'config_used_{timestamp}.yaml')
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Config saved to: {config_copy_path}")
    
    # 7. Generate Summary Report
    print("\n[7] Generating summary report...")
    print("=" * 70)
    
    # Top by different metrics
    ranking_metrics = [
        ('total_return_pct', 'Total Return', False),
        ('sharpe_ratio', 'Sharpe Ratio', False),
        ('sortino_ratio', 'Sortino Ratio', False),
        ('max_drawdown_pct', 'Max Drawdown (least negative)', True),  # Higher (less negative) is better
        ('calmar_ratio', 'Calmar Ratio', False),
        ('alpha_pct', 'Alpha vs Index', False),
        ('win_rate_pct', 'Win Rate', False),
        ('profit_factor', 'Profit Factor', False)
    ]
    
    summary_lines = []
    summary_lines.append(f"\nOPTIMIZATION SUMMARY - {timestamp}")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Total combinations tested: {n_combos}")
    summary_lines.append(f"Successful backtests: {successful}")
    summary_lines.append(f"Failed backtests: {failed}")
    summary_lines.append(f"Quarter range: {first_quarter} to {last_quarter}")
    summary_lines.append("")
    
    for metric, name, ascending in ranking_metrics:
        if metric not in results_df.columns:
            continue
            
        summary_lines.append(f"\n{'─' * 70}")
        summary_lines.append(f"TOP {top_n} BY {name.upper()}")
        summary_lines.append(f"{'─' * 70}")
        
        sorted_df = results_df.sort_values(metric, ascending=ascending).head(top_n)
        
        for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
            summary_lines.append(f"\n#{rank}: {row['param_id'][:80]}")
            summary_lines.append(f"    {name}: {row[metric]:.2f}")
            
            # Show other key metrics
            other_metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct']
            other_metrics = [m for m in other_metrics if m != metric and m in row]
            metric_str = " | ".join([f"{m.replace('_pct','%').replace('_ratio','')}: {row[m]:.2f}" for m in other_metrics])
            summary_lines.append(f"    {metric_str}")
    
    # Overall statistics
    summary_lines.append(f"\n\n{'=' * 70}")
    summary_lines.append("OVERALL STATISTICS")
    summary_lines.append("=" * 70)
    
    for metric in ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct']:
        if metric in results_df.columns:
            summary_lines.append(f"\n{metric}:")
            summary_lines.append(f"  Mean: {results_df[metric].mean():.2f}")
            summary_lines.append(f"  Std:  {results_df[metric].std():.2f}")
            summary_lines.append(f"  Min:  {results_df[metric].min():.2f}")
            summary_lines.append(f"  Max:  {results_df[metric].max():.2f}")
    
    # Print and save summary
    summary_text = '\n'.join(summary_lines)
    print(summary_text)
    
    summary_path = os.path.join(output_dir, f'summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\n  Summary saved to: {summary_path}")
    
    # 8. Final output
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - optimization_results_{timestamp}.csv (full results)")
    print(f"  - summary_{timestamp}.txt (summary report)")
    print(f"  - config_used_{timestamp}.yaml (configuration)")
    
    return results_df


def filter_results(
    results_df: pd.DataFrame,
    min_sharpe: float = None,
    max_drawdown: float = None,
    min_return: float = None,
    min_win_rate: float = None
) -> pd.DataFrame:
    """
    Filter optimization results based on constraints.
    
    Example:
        filtered = filter_results(df, min_sharpe=1.0, max_drawdown=-15)
    """
    filtered = results_df.copy()
    
    if min_sharpe is not None and 'sharpe_ratio' in filtered.columns:
        filtered = filtered[filtered['sharpe_ratio'] >= min_sharpe]
    
    if max_drawdown is not None and 'max_drawdown_pct' in filtered.columns:
        # max_drawdown is negative, so we want values >= max_drawdown (less negative)
        filtered = filtered[filtered['max_drawdown_pct'] >= max_drawdown]
    
    if min_return is not None and 'total_return_pct' in filtered.columns:
        filtered = filtered[filtered['total_return_pct'] >= min_return]
    
    if min_win_rate is not None and 'win_rate_pct' in filtered.columns:
        filtered = filtered[filtered['win_rate_pct'] >= min_win_rate]
    
    return filtered


def get_pareto_frontier(
    results_df: pd.DataFrame,
    maximize_cols: List[str],
    minimize_cols: List[str] = None
) -> pd.DataFrame:
    """
    Find Pareto-optimal configurations (non-dominated solutions).
    
    A solution is Pareto-optimal if no other solution is better in all metrics.
    
    Example:
        pareto = get_pareto_frontier(df, 
            maximize_cols=['total_return_pct', 'sharpe_ratio'],
            minimize_cols=['max_drawdown_pct'])  # Note: drawdown is negative
    """
    if minimize_cols is None:
        minimize_cols = []
    
    # Convert minimize columns to maximize (by negating)
    df = results_df.copy()
    for col in minimize_cols:
        if col in df.columns:
            df[f'{col}_neg'] = -df[col]
            maximize_cols = maximize_cols + [f'{col}_neg']
    
    # Get relevant columns
    cols = [c for c in maximize_cols if c in df.columns]
    if not cols:
        return results_df
    
    # Find Pareto frontier
    pareto_mask = np.ones(len(df), dtype=bool)
    
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            # Check if j dominates i (j is better in all metrics)
            dominates = all(df[cols].iloc[j] >= df[cols].iloc[i])
            strictly_better = any(df[cols].iloc[j] > df[cols].iloc[i])
            
            if dominates and strictly_better:
                pareto_mask[i] = False
                break
    
    # Clean up temporary columns
    for col in minimize_cols:
        if f'{col}_neg' in df.columns:
            df = df.drop(columns=[f'{col}_neg'])
    
    return results_df[pareto_mask]


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Strategy Optimization via Grid Search')
    parser.add_argument('--config', type=str, default='strategy_grid_config.yaml',
                       help='Path to grid configuration file')
    
    args = parser.parse_args()
    
    results = run_optimization(args.config)
