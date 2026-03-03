"""
Get Portfolio — Forward-Looking Stock Selection

Runs the stock selection pipeline from strategy_config.yaml (or a custom config)
WITHOUT backtesting. Outputs a CSV of selected stocks with portfolio weights
for the coming quarter(s).

Usage:
    cd src
    python get_portfolio.py                                  # All quarters in input data
    python get_portfolio.py --quarter 202602                  # Single quarter
    python get_portfolio.py --config my_config.yaml -o out.csv
    python get_portfolio.py --validate-prices --quarter 202602

The script reuses the same strategy_config.yaml used for backtesting.
Only the stock-selection portion of the config is used; TP/SL, reporting,
and simulation settings are ignored.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from backtest_strategy import (
    load_config,
    load_data,
    filter_data_by_quarters,
    run_stock_selection,
)
from selection import filter_tradeable_stocks
from config.defaults import DEFAULT_CONFIG_PATH
from utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = "../selected_stocks_data"


def _resolve_quarters(config, input_data, cli_quarter):
    """Determine first/last quarter and filter input data accordingly.

    Returns (filtered_df, first_quarter, last_quarter).
    """
    first_q = config.first_quarter
    last_q = config.last_quarter

    # Apply config-level quarter bounds if set
    if first_q is not None or last_q is not None:
        if first_q is None:
            first_q = input_data['quarter'].min()
        if last_q is None:
            last_q = input_data['quarter'].max()
        input_data = filter_data_by_quarters(input_data, first_q, last_q)

    # Narrow to a single CLI quarter if specified
    if cli_quarter is not None:
        input_data = input_data[input_data['quarter'] == cli_quarter].copy()
        first_q = cli_quarter
        last_q = cli_quarter

    if input_data.empty:
        quarters_msg = f"quarter {cli_quarter}" if cli_quarter else "the specified range"
        logger.error("No input data found for %s. Check input file and quarter filters.", quarters_msg)
        sys.exit(1)

    # Ensure bounds are set for downstream use
    if first_q is None:
        first_q = input_data['quarter'].min()
    if last_q is None:
        last_q = input_data['quarter'].max()

    return input_data, first_q, last_q


def _merge_prob_column(selected_stocks, input_data):
    """Merge the probability score back into the selection output for context."""
    if 'prob' not in selected_stocks.columns and 'prob' in input_data.columns:
        prob_lookup = input_data[['quarter', 'co_name', 'prob']].drop_duplicates()
        selected_stocks = selected_stocks.merge(prob_lookup, on=['quarter', 'co_name'], how='left')
    return selected_stocks


def _print_summary(selected_stocks, output_path):
    """Print a human-readable summary of the selection to the console."""
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS — Selected Stocks")
    print("=" * 70)

    quarters = sorted(selected_stocks['quarter'].unique())
    for q in quarters:
        q_data = selected_stocks[selected_stocks['quarter'] == q]
        print(f"\nQuarter {q}: {len(q_data)} stocks selected")
        print("-" * 50)

        # Choose display columns
        display_cols = ['co_name', 'stock_weight']
        if 'prob' in q_data.columns:
            display_cols.append('prob')
        if 'cat' in q_data.columns:
            display_cols.append('cat')
        if 'risk_adj_score' in q_data.columns:
            display_cols.append('risk_adj_score')

        display = q_data[display_cols].copy()

        # Format for readability
        display['stock_weight'] = display['stock_weight'].apply(lambda x: f"{x:.4f}")
        if 'prob' in display.columns:
            display['prob'] = display['prob'].apply(lambda x: f"{x:.4f}")
        if 'risk_adj_score' in display.columns:
            display['risk_adj_score'] = display['risk_adj_score'].apply(lambda x: f"{x:.4f}")

        print(display.to_string(index=False))

        # Weight sanity check
        total_weight = q_data['stock_weight'].sum()
        print(f"\n  Total weight: {total_weight:.4f}")

    print("\n" + "=" * 70)
    print(f"Output saved to: {output_path}")
    print("=" * 70 + "\n")


def get_portfolio(
    config_path=DEFAULT_CONFIG_PATH,
    quarter=None,
    output_path=None,
    validate_prices=False,
    quiet=False,
):
    """Run forward-looking stock selection and save results.

    Parameters:
        config_path: Path to strategy_config.yaml
        quarter: Optional single quarter (YYYYMM int) to select for
        output_path: Output CSV path. Auto-generated if None.
        validate_prices: If True, load price data and filter untradeable stocks
        quiet: If True, suppress console summary

    Returns:
        (output_path, selected_stocks_df)
    """
    # --- 1. Load config ---
    logger.info("Loading config from: %s", config_path)
    config = load_config(config_path)

    if not config.run_stock_selection:
        logger.error(
            "run_stock_selection is set to false in the config. "
            "Inference mode requires run_stock_selection: true."
        )
        sys.exit(1)

    # --- 2. Load input data ---
    logger.info("Loading input data from: %s", config.input_data_path)
    input_data = load_data(config.input_data_path)
    logger.info("Loaded %d rows across %d quarters",
                len(input_data), input_data['quarter'].nunique())

    # --- 3. Resolve quarters ---
    input_data, first_q, last_q = _resolve_quarters(config, input_data, quarter)
    logger.info("Quarters in scope: %d to %d (%d quarters)",
                first_q, last_q, input_data['quarter'].nunique())

    # --- 4. Optional price validation ---
    if validate_prices:
        logger.info("Validating stock tradeability against price data...")
        price_data = load_data(config.price_data_path)
        input_data, data_issues = filter_tradeable_stocks(
            input_data, price_data, first_q, last_q
        )
        if data_issues is not None and not data_issues.empty:
            logger.warning(
                "Filtered %d stock-quarter pairs with price data issues",
                len(data_issues),
            )
        else:
            logger.info("All stocks have complete price data coverage.")

    # --- 5. Run stock selection ---
    logger.info("Running stock selection (type=%s, sort_by=%s)...",
                config.selection_type, config.sort_by)

    selected_stocks = run_stock_selection(
        input_data,
        config.category_scheme,
        config.category_counts,
        config.category_weights,
        sort_by=config.sort_by,
        min_prob_threshold=config.min_prob_threshold,
        selection_type=config.selection_type,
        top_k_config=(
            config.top_k_config.model_dump() if config.top_k_config else None
        ),
        category_based_weighting_scheme=config.category_based_selection_weighting_scheme,
        selection_dimension=config.selection_dimension,
        weighting_dimension=config.weighting_dimension,
    )

    if selected_stocks.empty:
        logger.error("Stock selection returned no results. Check input data and config.")
        sys.exit(1)

    logger.info("Selected %d stock positions across %d quarter(s)",
                len(selected_stocks), selected_stocks['quarter'].nunique())

    # --- 6. Enrich with probability scores ---
    selected_stocks = _merge_prob_column(selected_stocks, input_data)

    # --- 7. Save output ---
    if output_path is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"selection_{timestamp}.csv")

    # Ensure parent directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    selected_stocks.to_csv(output_path, index=False)
    logger.info("Selection saved to: %s", output_path)

    # --- 8. Console summary ---
    if not quiet:
        _print_summary(selected_stocks, output_path)

    return output_path, selected_stocks


def main():
    parser = argparse.ArgumentParser(
        description="Run forward-looking stock selection without backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select stocks for all quarters in input data
  python get_portfolio.py

  # Select stocks for a specific quarter
  python get_portfolio.py --quarter 202602

  # Use a custom config and output path
  python get_portfolio.py --config my_strategy.yaml -o portfolio.csv

  # Validate that selected stocks have tradeable price data
  python get_portfolio.py --validate-prices --quarter 202602

  # Suppress console output (just save file)
  python get_portfolio.py --quarter 202602 -q

The script reads the stock-selection portion of strategy_config.yaml
and outputs a CSV with columns: quarter, co_name, stock_weight, prob, etc.
TP/SL, simulation, and reporting settings are ignored.
        """
    )

    parser.add_argument(
        '--config', '-c',
        default=DEFAULT_CONFIG_PATH,
        help=f'Path to strategy config YAML (default: {DEFAULT_CONFIG_PATH})'
    )
    parser.add_argument(
        '--quarter',
        type=int,
        metavar='YYYYMM',
        default=None,
        help='Select stocks for a single quarter (e.g., 202602). '
             'If omitted, runs for all quarters in input data.'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output CSV file path (default: inference_results/selection_<timestamp>.csv)'
    )
    parser.add_argument(
        '--validate-prices',
        action='store_true',
        default=False,
        help='Load price data and filter out stocks with incomplete price coverage'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        default=False,
        help='Suppress console summary table'
    )

    args = parser.parse_args()

    get_portfolio(
        config_path=args.config,
        quarter=args.quarter,
        output_path=args.output,
        validate_prices=args.validate_prices,
        quiet=args.quiet,
    )


if __name__ == '__main__':
    setup_logging(console_level=logging.INFO)
    main()
