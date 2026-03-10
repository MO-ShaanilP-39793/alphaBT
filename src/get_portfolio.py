"""
Get Portfolio — Forward-Looking Stock Selection

Runs the stock selection pipeline from strategy_config.yaml (or a custom config)
WITHOUT backtesting. Outputs a CSV of selected stocks with portfolio weights
for the coming quarter(s).

With ``--with-levels``, also computes entry/TP/SL levels, detects TP/SL
triggers and exit prices/dates through the data's as-of date, and outputs
daily portfolio and index series — supporting weekly re-runs and dashboard use.

Usage:
    cd src
    python get_portfolio.py                                  # All quarters in input data
    python get_portfolio.py --quarter 202602                  # Single quarter
    python get_portfolio.py --config my_config.yaml -o out.csv
    python get_portfolio.py --validate-prices --quarter 202602
    python get_portfolio.py --with-levels --quarter 202602    # Levels + exits + daily series
    python get_portfolio.py --with-levels --as-of 2026-03-06  # Explicit as-of override

The script reuses the same strategy_config.yaml used for backtesting.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from backtest_strategy import (
    load_config,
    load_data,
    filter_data_by_quarters,
    run_stock_selection,
)
from backtest.tpsl import (
    _resolve_entry,
    _resolve_exit,
    calculate_dynamic_thresholds,
    _resolve_atr_for_quarter,
)
from backtest.simulation import (
    calculate_position_sizes,
    generate_quarter_equity_curve,
)
from selection import filter_tradeable_stocks
from config.defaults import DEFAULT_CONFIG_PATH, INITIAL_CAPITAL
from utils.quarter import get_quarter_dates
from utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = "../selected_stocks_data"


# ---------------------------------------------------------------------------
# TP/SL helpers (mirrors backtest_strategy logic)
# ---------------------------------------------------------------------------

def _resolve_tpsl_scheme(config):
    """Resolve which dimension scheme (volatility/mcap) drives TP/SL thresholds."""
    dim = config.tpsl_category_dimension
    if dim == 'selection':
        return config.selection_dimension
    if dim == 'weighting':
        return config.weighting_dimension
    if dim in ('volatility', 'mcap'):
        return dim
    raise ValueError(
        f"tpsl_category_dimension must be 'selection', 'weighting', "
        f"'volatility', or 'mcap', got '{dim}'"
    )


def _add_tpsl_cat(selected_stocks, config, tpsl_scheme):
    """Add ``tpsl_cat`` column mirroring logic in backtest_strategy._simulate_and_compute."""
    df = selected_stocks.copy()
    tpsl_dim = config.tpsl_category_dimension
    sel_dim = config.selection_dimension
    wgt_dim = config.weighting_dimension

    if 'selection_cat' in df.columns:
        if tpsl_dim in ('selection', sel_dim):
            df['tpsl_cat'] = df['selection_cat']
        elif tpsl_dim in ('weighting', wgt_dim):
            df['tpsl_cat'] = df['weight_cat']
        else:
            if tpsl_scheme == sel_dim:
                df['tpsl_cat'] = df['selection_cat']
            else:
                df['tpsl_cat'] = df['weight_cat']

    if 'cat' not in df.columns:
        df['cat'] = '_default'

    return df


# ---------------------------------------------------------------------------
# Per-row levels + exit monitoring
# ---------------------------------------------------------------------------

def _compute_levels_and_exits(selected_stocks, config, tpsl_scheme,
                              price_data, index_df, as_of):
    """Compute entry/TP/SL levels and exit monitoring for every selected row.

    Appends columns: entry_price, entry_date, num_shares, tp_price, sl_price,
    tp_pct, sl_pct, price_data_adequate, exit_date, exit_price, TP_triggered,
    SL_triggered, regime_exit, holding_period.
    """
    entry_price_window = config.entry_price_window
    tp_enabled = config.tp_enabled
    sl_enabled = config.sl_enabled
    tp_mode = config.tp_mode
    sl_mode = config.sl_mode

    tiered_cfg = config.tiered_config.model_dump() if config.tiered_config else None
    atr_cfg = config.atr_config.model_dump() if config.atr_config else None
    pivot_cfg = config.pivot_config.model_dump() if config.pivot_config else None
    flat_cfg = config.flat_config.model_dump() if config.flat_config else {}
    idx_exit_cfg = config.index_exit.model_dump() if config.index_exit else {
        'regime_filter': {'enabled': False},
        'vol_adjustment': {'enabled': False},
    }

    results = []
    for _, row in selected_stocks.iterrows():
        quarter = str(row['quarter'])
        co_name = row['co_name']
        stock_weight = row['stock_weight']
        cat = row.get('tpsl_cat') or row.get('cat', '_default')

        entry_start, mandatory_exit = get_quarter_dates(quarter)
        effective_exit = min(mandatory_exit, as_of)

        co_prices = price_data[
            (price_data['co_name'] == co_name)
            & (price_data['date'] >= entry_start)
            & (price_data['date'] <= effective_exit)
        ].sort_values('date')

        rec = {}

        entry_price, entry_date, last_entry_calc_date = _resolve_entry(
            co_prices, entry_start, entry_price_window,
        )

        if entry_price is None:
            rec.update({
                'entry_price': np.nan, 'entry_date': pd.NaT,
                'num_shares': np.nan, 'tp_price': np.nan, 'sl_price': np.nan,
                'tp_pct': np.nan, 'sl_pct': np.nan, 'price_data_adequate': False,
                'exit_date': pd.NaT, 'exit_price': np.nan,
                'TP_triggered': False, 'SL_triggered': False,
                'regime_exit': False, 'holding_period': np.nan,
            })
            results.append(rec)
            continue

        rec['entry_price'] = entry_price
        rec['entry_date'] = entry_date
        rec['num_shares'] = int((INITIAL_CAPITAL * stock_weight) / entry_price)
        rec['price_data_adequate'] = True

        # --- TP/SL thresholds ---
        tp_price = None
        sl_price = None
        tp_pct_used = np.nan
        sl_pct_used = np.nan

        effective_atr = _resolve_atr_for_quarter(atr_cfg, row['quarter'])

        if tp_enabled or sl_enabled:
            tp_price, sl_price, meta = calculate_dynamic_thresholds(
                price_data, co_name, entry_price, entry_date,
                tpsl_scheme, cat, tp_mode, sl_mode, index_df,
                atr_config=effective_atr, pivot_config=pivot_cfg,
                flat_config=flat_cfg, index_exit_config=idx_exit_cfg,
                tiered_config=tiered_cfg,
            )
            if tp_enabled:
                tp_pct_used = meta.get('tp_pct', np.nan)
            else:
                tp_price = None
            if sl_enabled:
                sl_pct_used = meta.get('sl_pct', np.nan)
            else:
                sl_price = None

        rec['tp_price'] = tp_price if tp_price is not None else np.nan
        rec['sl_price'] = sl_price if sl_price is not None else np.nan
        rec['tp_pct'] = tp_pct_used
        rec['sl_pct'] = sl_pct_used

        # --- Exit monitoring through as_of ---
        monitoring_df = co_prices[
            (co_prices['date'] > last_entry_calc_date)
            & (co_prices['date'] <= effective_exit)
        ]

        exit_date, exit_price, sl_trig, tp_trig, regime_exit, holding = _resolve_exit(
            monitoring_df, tp_price, sl_price, entry_date,
            index_df, idx_exit_cfg,
        )

        # If no TP/SL/regime triggered and we haven't reached the quarter
        # end yet, the position is still open — report exit as NA rather
        # than using the as-of date as a pseudo time-exit.
        if exit_date is not None and not tp_trig and not sl_trig and not regime_exit:
            if as_of < mandatory_exit:
                exit_date = None
                exit_price = None
                holding = None

        if exit_date is None:
            exit_date = pd.NaT
            exit_price = np.nan

        rec['exit_date'] = exit_date
        rec['exit_price'] = exit_price
        rec['TP_triggered'] = tp_trig
        rec['SL_triggered'] = sl_trig
        rec['regime_exit'] = regime_exit
        rec['holding_period'] = holding if holding is not None else np.nan

        results.append(rec)

    level_cols = pd.DataFrame(results)
    out = pd.concat(
        [selected_stocks.reset_index(drop=True), level_cols.reset_index(drop=True)],
        axis=1,
    )
    return out


# ---------------------------------------------------------------------------
# Daily portfolio + index series
# ---------------------------------------------------------------------------

def _build_daily_series(selected_stocks, config, price_data, index_data, as_of,
                        first_q, last_q):
    """Build daily portfolio value (and index) series through as_of.

    Returns a detailed DataFrame with columns:
        date, Total_Portfolio_Value, <stock_1>, ..., <stock_N>,
        Cash_In_Hand, index_value
    """
    valid = selected_stocks[selected_stocks['price_data_adequate'].eq(True)].copy()
    if valid.empty:
        logger.warning("No positions with valid entry — skipping daily series.")
        return None

    entry_price_window = config.entry_price_window

    trades = valid[['quarter', 'co_name', 'cat', 'stock_weight',
                     'entry_price', 'exit_date', 'exit_price']].copy()

    price_capped = price_data[price_data['date'] <= as_of].copy()

    quarters = sorted(trades['quarter'].unique())
    all_curves = []
    current_capital = INITIAL_CAPITAL

    for q in quarters:
        q_trades = trades[trades['quarter'] == q].copy()
        if q_trades.empty:
            continue

        q_trades_sized = calculate_position_sizes(q_trades, current_capital)

        try:
            curve = generate_quarter_equity_curve(
                q_trades_sized, price_capped, int(q), entry_price_window,
                risk_free_rate_annual=config.cash_appreciation_rate,
            )
        except (ValueError, KeyError) as exc:
            logger.warning("Equity curve failed for quarter %s: %s", q, exc)
            continue

        curve['quarter'] = q
        all_curves.append(curve)
        current_capital = curve['Total_Portfolio_Value'].iloc[-1]

    if not all_curves:
        logger.warning("Could not generate any equity curves.")
        return None

    daily_pf = pd.concat(all_curves).sort_index()

    # Add index value by direct join
    if index_data is not None and not index_data.empty:
        idx_capped = index_data[index_data['date'] <= as_of].copy()
        idx_series = idx_capped.set_index('date')['value']
        daily_pf['index_value'] = daily_pf.index.map(idx_series)

    # Arrange columns: date first (from index), then Total_Portfolio_Value,
    # then per-stock columns, Cash_In_Hand, index_value.
    meta_cols = {'Total_Portfolio_Value', 'Cash_In_Hand', 'quarter', 'index_value'}
    stock_cols = [c for c in daily_pf.columns if c not in meta_cols]

    ordered = ['Total_Portfolio_Value'] + sorted(stock_cols) + ['Cash_In_Hand']
    if 'index_value' in daily_pf.columns:
        ordered.append('index_value')

    result = daily_pf[ordered].copy()
    result.index.name = 'date'
    result = result.reset_index()

    return result


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


def _print_summary(selected_stocks, output_path, daily_series_path=None):
    """Print a human-readable summary of the selection to the console."""
    has_levels = 'entry_price' in selected_stocks.columns

    print("\n" + "=" * 70)
    title = "INFERENCE RESULTS — Selected Stocks"
    if has_levels:
        title += " (with Levels & Exits)"
    print(title)
    print("=" * 70)

    quarters = sorted(selected_stocks['quarter'].unique())
    for q in quarters:
        q_data = selected_stocks[selected_stocks['quarter'] == q]
        print(f"\nQuarter {q}: {len(q_data)} stocks selected")
        print("-" * 50)

        display_cols = ['co_name', 'stock_weight']
        if 'prob' in q_data.columns:
            display_cols.append('prob')
        if 'cat' in q_data.columns:
            display_cols.append('cat')
        if 'risk_adj_score' in q_data.columns:
            display_cols.append('risk_adj_score')

        if has_levels:
            display_cols += ['entry_price', 'tp_price', 'sl_price',
                             'num_shares', 'exit_date', 'exit_price',
                             'TP_triggered', 'SL_triggered']

        available = [c for c in display_cols if c in q_data.columns]
        display = q_data[available].copy()

        display['stock_weight'] = display['stock_weight'].apply(lambda x: f"{x:.4f}")
        if 'prob' in display.columns:
            display['prob'] = display['prob'].apply(lambda x: f"{x:.4f}")
        if 'risk_adj_score' in display.columns:
            display['risk_adj_score'] = display['risk_adj_score'].apply(lambda x: f"{x:.4f}")

        if has_levels:
            for col in ('entry_price', 'tp_price', 'sl_price', 'exit_price'):
                if col in display.columns:
                    display[col] = display[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
                    )
            if 'num_shares' in display.columns:
                display['num_shares'] = display['num_shares'].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else "—"
                )
            if 'exit_date' in display.columns:
                display['exit_date'] = display['exit_date'].apply(
                    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "—"
                )

        print(display.to_string(index=False))

        total_weight = q_data['stock_weight'].sum()
        print(f"\n  Total weight: {total_weight:.4f}")

        if has_levels:
            adequate = q_data.get('price_data_adequate')
            if adequate is not None:
                n_ok = adequate.sum()
                print(f"  Positions with entry: {n_ok}/{len(q_data)}")
            tp_count = q_data['TP_triggered'].sum() if 'TP_triggered' in q_data.columns else 0
            sl_count = q_data['SL_triggered'].sum() if 'SL_triggered' in q_data.columns else 0
            if tp_count or sl_count:
                print(f"  TP triggered: {int(tp_count)}  |  SL triggered: {int(sl_count)}")

    print("\n" + "=" * 70)
    print(f"Selection saved to: {output_path}")
    if daily_series_path:
        print(f"Daily series saved to: {daily_series_path}")
    print("=" * 70 + "\n")


def get_portfolio(
    config_path=DEFAULT_CONFIG_PATH,
    quarter=None,
    output_path=None,
    validate_prices=False,
    with_levels=False,
    as_of=None,
    quiet=False,
):
    """Run forward-looking stock selection and save results.

    Parameters:
        config_path: Path to strategy_config.yaml
        quarter: Optional single quarter (YYYYMM int) to select for
        output_path: Output CSV path. Auto-generated if None.
        validate_prices: If True, load price data and filter untradeable stocks
        with_levels: If True, compute entry/TP/SL/shares, exit monitoring, and
            daily portfolio+index series through the as-of date.
        as_of: Explicit as-of date (str or Timestamp). Defaults to max date in
            the loaded price data when ``with_levels`` is True.
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

    if with_levels and not config.price_data_path:
        logger.error("--with-levels requires price_data_path in config.")
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
    price_data = None
    if validate_prices or with_levels:
        logger.info("Loading price data from: %s", config.price_data_path)
        price_data = load_data(config.price_data_path)
        price_data['date'] = pd.to_datetime(price_data['date'])

    if validate_prices:
        logger.info("Validating stock tradeability against price data...")
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

    # --- 7. Levels, exit monitoring, and daily series ---
    daily_series_path = None

    if with_levels:
        # Resolve as-of date
        if as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
        else:
            as_of_ts = price_data['date'].max()
        logger.info("As-of date: %s", as_of_ts.date())

        # Restrict price data to <= as_of
        price_data = price_data[price_data['date'] <= as_of_ts].copy()

        # Load index data if available
        index_df = None
        if config.index_data_path:
            try:
                index_raw = load_data(config.index_data_path)
                index_raw['date'] = pd.to_datetime(index_raw['date'])
                index_df = index_raw[index_raw['date'] <= as_of_ts].copy()
                logger.info("Loaded index data (%d rows through %s)",
                            len(index_df), as_of_ts.date())
            except Exception as exc:
                logger.warning("Could not load index data: %s", exc)

        # Resolve TP/SL scheme and add tpsl_cat
        tpsl_scheme = _resolve_tpsl_scheme(config)
        selected_stocks = _add_tpsl_cat(selected_stocks, config, tpsl_scheme)

        # Compute levels + exit monitoring
        logger.info("Computing entry/TP/SL levels and exit monitoring...")
        selected_stocks = _compute_levels_and_exits(
            selected_stocks, config, tpsl_scheme,
            price_data, index_df, as_of_ts,
        )

        n_with_entry = selected_stocks['price_data_adequate'].sum()
        n_tp = selected_stocks['TP_triggered'].sum()
        n_sl = selected_stocks['SL_triggered'].sum()
        logger.info(
            "Levels: %d/%d with entry | TP triggered: %d | SL triggered: %d",
            n_with_entry, len(selected_stocks), n_tp, n_sl,
        )

        # Build daily series
        logger.info("Building daily portfolio/index series through %s...",
                     as_of_ts.date())
        daily_series = _build_daily_series(
            selected_stocks, config, price_data, index_df,
            as_of_ts, first_q, last_q,
        )

        if daily_series is not None and not daily_series.empty:
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            daily_series_path = os.path.join(
                DEFAULT_OUTPUT_DIR,
                f"daily_series_{first_q}_{as_of_ts.strftime('%Y%m%d')}_{timestamp}.csv",
            )
            daily_series.to_csv(daily_series_path, index=False)
            logger.info("Daily series saved to: %s", daily_series_path)

    # --- 8. Save selection output ---
    if output_path is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"selection_{timestamp}.csv")

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    selected_stocks.to_csv(output_path, index=False)
    logger.info("Selection saved to: %s", output_path)

    # --- 9. Console summary ---
    if not quiet:
        _print_summary(selected_stocks, output_path, daily_series_path)

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

  # Compute levels, exit monitoring, and daily portfolio series
  python get_portfolio.py --with-levels --quarter 202602

  # Same, with an explicit as-of cutoff date
  python get_portfolio.py --with-levels --as-of 2026-03-06 --quarter 202602

  # Suppress console output (just save file)
  python get_portfolio.py --quarter 202602 -q

The script reads strategy_config.yaml. Without --with-levels only the
stock-selection portion is used. With --with-levels the TP/SL and
price settings are also consumed.
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
        help='Output CSV file path (default: selected_stocks_data/selection_<timestamp>.csv)'
    )
    parser.add_argument(
        '--validate-prices',
        action='store_true',
        default=False,
        help='Load price data and filter out stocks with incomplete price coverage'
    )
    parser.add_argument(
        '--with-levels',
        action='store_true',
        default=False,
        help='Compute entry/TP/SL levels, exit monitoring, and daily series'
    )
    parser.add_argument(
        '--as-of',
        type=str,
        default=None,
        metavar='DATE',
        help='As-of cutoff date (YYYY-MM-DD). Defaults to max date in price data.'
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
        with_levels=args.with_levels,
        as_of=args.as_of,
        quiet=args.quiet,
    )


if __name__ == '__main__':
    setup_logging(console_level=logging.INFO)
    main()
