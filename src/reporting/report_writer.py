"""Main report orchestrator — generates the consolidated backtest Excel workbook."""

from typing import Optional
import io
import math
import struct

import pandas as pd

from config.defaults import INITIAL_CAPITAL
from utils.logging_config import get_logger

logger = get_logger(__name__)

from .metrics import (
    compute_portfolio_metrics,
    compute_benchmark_metrics,
    compute_cash_metrics,
)
from .analytics import (
    compute_portfolio_performance,
    compute_rolling_performance,
    compute_calendar_year_performance,
    compute_trailing_returns,
    compute_monthly_returns_from_daily,
    compute_up_down_months,
    compute_crisis_regime_returns,
    compute_market_regime_returns,
    compute_quarterly_alpha,
    compute_churn_analysis,
    get_stock_counts_by_mcap,
    get_comprehensive_quarter_analysis,
)
from .charts import (
    create_portfolio_vs_index_chart,
    create_daily_drawdown_chart,
    create_monthly_returns_heatmap,
    create_calendar_year_heatmap,
    create_correlation_heatmap,
    create_distribution_chart,
    create_box_plot,
    create_cash_pct_chart,
    create_churn_chart,
)


def _png_image_rows(
    buffer: io.BytesIO,
    excel_row_pts: float = 15.0,
    padding: int = 2,
) -> int:
    """
    Compute how many Excel rows a PNG image occupies, based on its actual pixel height
    and the DPI embedded in the PNG's pHYs chunk (written by matplotlib at save time).

    PNG binary layout used here:
      IHDR chunk  (always first after the 8-byte signature):
        bytes 16–19 : image height (px)
      pHYs chunk  (always before IDAT; present when matplotlib saves with dpi=N):
        4 bytes pixels-per-unit-X  (pixels per metre when unit==1)
        4 bytes pixels-per-unit-Y
        1 byte  unit specifier     (0=unknown, 1=metre)

    Conversion:
      dpi            = pixels_per_metre / 39.3701   (if unit == 1)
      height_inches  = height_px / dpi
      rows           = ceil(height_inches / (excel_row_pts / 72)) + padding

    The buffer is seeked back to 0 afterwards so it remains usable by insert_image.
    """
    data = buffer.read()
    buffer.seek(0)

    # Height in pixels from IHDR
    height_px = struct.unpack('>I', data[20:24])[0]

    # Walk PNG chunks to find pHYs (always appears before IDAT)
    dpi = 96  # sensible fallback if pHYs is absent
    pos = 33  # first byte of the chunk after IHDR (8 sig + 4 len + 4 type + 13 data + 4 crc)
    while pos + 12 <= len(data):
        chunk_len = struct.unpack('>I', data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        if chunk_type == b'pHYs':
            px_per_metre = struct.unpack('>I', data[pos + 8:pos + 12])[0]
            unit = data[pos + 16]
            if unit == 1 and px_per_metre > 0:
                dpi = round(px_per_metre / 39.3701)
            break
        if chunk_type == b'IDAT':
            break  # pHYs always precedes IDAT; stop early
        pos += 12 + chunk_len

    height_inches = height_px / dpi
    excel_row_height_inches = excel_row_pts / 72.0
    return math.ceil(height_inches / excel_row_height_inches) + padding


def _format_monthly_returns_for_excel(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format monthly returns DataFrame for Excel export.
    
    - Converts Date index to 'MMM YYYY' format (e.g., 'Jan 2020')
    - Multiplies returns by 100 (0.0234 → 2.34, formatted as 2.34% in Excel)
    - Rounds to 2 decimal places
    - Resets index to make Date a column
    
    Parameters:
    - monthly_df: DataFrame with date index and return columns (decimal format)
    
    Returns:
    - DataFrame ready for .to_excel() with index=False
    """
    if monthly_df.empty:
        return monthly_df
    
    formatted_df = monthly_df.copy()
    
    # Convert date index to 'MMM YYYY' format
    formatted_df.index = pd.to_datetime(formatted_df.index).strftime('%b %Y')
    formatted_df.index.name = 'Month'
    
    # Convert returns from decimal to percentage (multiply by 100)
    # Excel percentage format will display 2.34 as '2.34%'
    formatted_df = formatted_df * 100
    
    # Round to 2 decimal places
    formatted_df = formatted_df.round(2)
    
    # Reset index to make Month a column
    formatted_df = formatted_df.reset_index()
    
    return formatted_df


def generate_backtest_report(
    daily_pf: pd.DataFrame,
    trade_results: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    sub_periods: Optional[list[list[int]]] = None,
    input_frequency: str = "daily",
    data_issues: Optional[pd.DataFrame] = None,
    first_quarter: Optional[int] = None,
    last_quarter: Optional[int] = None,
) -> str:
    """
    Generate the single consolidated backtest report Excel workbook.

    Parameters:
    - daily_pf: DataFrame with columns ['date', 'portfolio_value', 'quarter']
    - trade_results: DataFrame from trade simulation
    - comparison_df: DataFrame with portfolio vs index comparison (can be None)
    - output_path: Full path for the output Excel file
    - sub_periods: List of [start_year, end_year] pairs for sub-period analysis
    - input_frequency: 'daily' or 'monthly'
    - report_title: Title for the report
    - data_issues: DataFrame with data quality issues (can be None)
    - first_quarter: First quarter (int) for chart titles
    - last_quarter: Last quarter (int) for chart titles

    Returns:
    - Path to the generated Excel file
    """
    logger.info("Generating consolidated backtest report...")

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    # Prepare daily returns from portfolio values
    daily_pf_sorted = daily_pf.copy()
    daily_pf_sorted['date'] = pd.to_datetime(daily_pf_sorted['date'])
    daily_pf_sorted = daily_pf_sorted.sort_values('date')

    # Daily returns DataFrame (for returns-based analysis)
    returns_series = daily_pf_sorted.set_index('date')['portfolio_value'].pct_change().dropna()
    returns_series.name = 'Portfolio'

    # Build combined returns (Portfolio + Benchmark)
    if comparison_df is not None and not comparison_df.empty:
        comp = comparison_df.copy()
        comp['date'] = pd.to_datetime(comp['date'])
        comp = comp.set_index('date').sort_index()
        index_returns = comp['index_fund_value'].pct_change().dropna()
        index_returns.name = 'Benchmark'
        combined_returns = pd.concat([returns_series, index_returns], axis=1).dropna()
    else:
        combined_returns = returns_series.to_frame()

    combined_returns.index.name = 'Date'

    # Monthly returns (for charts)
    monthly_df = compute_monthly_returns_from_daily(combined_returns, input_frequency)

    # Format monthly returns for Excel sheet
    monthly_performance_sheet = _format_monthly_returns_for_excel(monthly_df)

    # Date range
    data_start = combined_returns.index.min()
    data_end = combined_returns.index.max()

    # =========================================================================
    # COMPUTE ALL ANALYTICS
    # =========================================================================

    logger.info("Computing portfolio metrics...")
    portfolio_metrics = None
    try:
        portfolio_metrics = compute_portfolio_metrics(daily_pf)
    except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
        logger.warning("Could not compute portfolio metrics: %s", e)

    logger.info("Computing benchmark metrics...")
    benchmark_metrics = None
    if comparison_df is not None:
        try:
            benchmark_metrics = compute_benchmark_metrics(comparison_df)
        except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
            logger.warning("Could not compute benchmark metrics: %s", e)

    logger.info("Computing periodic performance...")
    periodic_inception = compute_portfolio_performance(combined_returns, input_frequency)
    periodic_inception.index.name = "Since_Inception"

    sub_period_dfs = []
    if sub_periods:
        for start_y, end_y in sub_periods:
            sp_df = compute_portfolio_performance(combined_returns, input_frequency, start_y, end_y)
            if not sp_df.empty:
                sp_df.index.name = f"{start_y}_{end_y}"
                sub_period_dfs.append(sp_df)

    logger.info("Computing rolling performance...")
    rolling_3m = compute_rolling_performance(combined_returns, input_frequency, 3, "monthly", annualize=False)
    rolling_3m.index.name = "Rolling_3M" if not rolling_3m.empty else None
    rolling_6m = compute_rolling_performance(combined_returns, input_frequency, 6, "monthly", annualize=False)
    rolling_6m.index.name = "Rolling_6M" if not rolling_6m.empty else None
    rolling_1y = compute_rolling_performance(combined_returns, input_frequency, 1, "yearly")
    rolling_1y.index.name = "Rolling_1Y" if not rolling_1y.empty else None
    rolling_3y = compute_rolling_performance(combined_returns, input_frequency, 3, "yearly")
    rolling_3y.index.name = "Rolling_3Y_ann" if not rolling_3y.empty else None
    rolling_5y = compute_rolling_performance(combined_returns, input_frequency, 5, "yearly")
    rolling_5y.index.name = "Rolling_5Y_ann" if not rolling_5y.empty else None

    logger.info("Computing calendar year performance...")
    calendar_df = compute_calendar_year_performance(combined_returns, input_frequency)
    calendar_df.index.name = "Calendar_Year" if not calendar_df.empty else None

    logger.info("Computing trailing returns...")
    trailing_df = compute_trailing_returns(combined_returns, input_frequency)
    last_date = pd.to_datetime(data_end).strftime("%Y-%m-%d")
    trailing_df.index.name = f"Trailing_Returns_{last_date}"

    logger.info("Computing up/down months...")
    up_down_df = compute_up_down_months(monthly_df)

    logger.info("Computing quarter analysis...")
    quarter_analysis = None
    try:
        quarter_analysis = get_comprehensive_quarter_analysis(trade_results)
    except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
        logger.warning("Could not compute quarter analysis: %s", e)

    logger.info("Computing regime returns...")
    crisis_df = compute_crisis_regime_returns(combined_returns, data_start, data_end)
    market_df = compute_market_regime_returns(combined_returns, data_start, data_end)

    # Quarterly alpha (needs benchmark)
    quarterly_alpha_df = None
    if comparison_df is not None and trade_results is not None:
        logger.info("Computing quarterly alpha...")
        try:
            quarterly_alpha_df = compute_quarterly_alpha(combined_returns, trade_results)
        except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
            logger.warning("Could not compute quarterly alpha: %s", e)

    logger.info("Computing cash metrics...")
    cash_metrics = None
    try:
        cash_metrics = compute_cash_metrics(daily_pf)
    except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
        logger.warning("Could not compute cash metrics: %s", e)

    logger.info("Computing churn analysis...")
    churn_df = None
    if trade_results is not None:
        try:
            churn_df = compute_churn_analysis(trade_results)
        except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
            logger.warning("Could not compute churn analysis: %s", e)

    # Stock counts by mcap (conditional)
    mcap_counts_df = None
    if trade_results is not None:
        has_mcap_category = 'mcap_category' in trade_results.columns
        has_cat_with_mcap = False
        if 'cat' in trade_results.columns:
            cat_values = set(trade_results['cat'].dropna().unique())
            mcap_values = {'largecap', 'midcap', 'smallcap'}
            has_cat_with_mcap = bool(mcap_values.intersection(cat_values))
        if has_mcap_category or has_cat_with_mcap:
            logger.info("Computing stock counts by market cap...")
            try:
                mcap_counts_df = get_stock_counts_by_mcap(trade_results)
            except ValueError as e:
                logger.warning("Could not compute mcap counts: %s", e)

    # =========================================================================
    # CREATE CHARTS
    # =========================================================================

    logger.info("Creating charts...")

    pf_vs_index_chart = None
    if comparison_df is not None:
        pf_vs_index_chart = create_portfolio_vs_index_chart(
            comparison_df, first_quarter, last_quarter, INITIAL_CAPITAL
        )

    drawdown_chart = create_daily_drawdown_chart(daily_pf)
    cash_pct_chart = create_cash_pct_chart(daily_pf)
    heatmap_chart = create_monthly_returns_heatmap(daily_pf)
    cy_heatmap = create_calendar_year_heatmap(calendar_df) if not calendar_df.empty else None
    corr_heatmap = create_correlation_heatmap(combined_returns)
    bell_curve = create_distribution_chart(monthly_df)
    box_plot = create_box_plot(monthly_df)
    churn_chart = create_churn_chart(churn_df) if churn_df is not None and not churn_df.empty else None

    # =========================================================================
    # WRITE EXCEL WORKBOOK
    # =========================================================================

    logger.info("Writing Excel workbook...")

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Format definitions
        format_decimal = workbook.add_format({'num_format': '0.00'})
        percent_format = workbook.add_format({'num_format': '0.0%', 'align': 'right'})

        # ----- Sheet: portfolio_metrics -----
        if portfolio_metrics is not None:
            pd.DataFrame([portfolio_metrics]).to_excel(
                writer, sheet_name="portfolio_metrics", index=False
            )

        # ----- Sheet: benchmark_metrics -----
        if benchmark_metrics is not None:
            pd.DataFrame([benchmark_metrics]).to_excel(
                writer, sheet_name="benchmark_metrics", index=False
            )

        # ----- Sheet: periodic_returns -----
        start_row = 0
        periodic_inception.reset_index().to_excel(
            writer, startrow=start_row, sheet_name="periodic_returns", index=False
        )
        for sp_df in sub_period_dfs:
            start_row += len(sp_df) + 5
            sp_df.reset_index().to_excel(
                writer, startrow=start_row, sheet_name="periodic_returns", index=False
            )

        # ----- Sheet: rolling_returns -----
        start_row = 0
        if not rolling_3m.empty:
            rolling_3m.reset_index().to_excel(
                writer, startrow=start_row, sheet_name="rolling_returns", index=False
            )
            start_row += len(rolling_3m) + 5
        if not rolling_6m.empty:
            rolling_6m.reset_index().to_excel(
                writer, startrow=start_row, sheet_name="rolling_returns", index=False
            )
            start_row += len(rolling_6m) + 5
        if not rolling_1y.empty:
            rolling_1y.reset_index().to_excel(
                writer, startrow=start_row, sheet_name="rolling_returns", index=False
            )
            start_row += len(rolling_1y) + 5
        if not rolling_3y.empty:
            rolling_3y.reset_index().to_excel(
                writer, startrow=start_row, sheet_name="rolling_returns", index=False
            )
            start_row += len(rolling_3y) + 5
        if not rolling_5y.empty:
            rolling_5y.reset_index().to_excel(
                writer, startrow=start_row, sheet_name="rolling_returns", index=False
            )

        # ----- Sheet: calendar_returns -----
        if not calendar_df.empty:
            calendar_df.reset_index().to_excel(
                writer, sheet_name="calendar_returns", index=False
            )

        # ----- Sheet: up_down_months -----
        up_down_df.reset_index().to_excel(
            writer, sheet_name="up_down_months", index=False
        )

        # ----- Sheet: trailing_returns -----
        trailing_df.reset_index().to_excel(
            writer, sheet_name="trailing_returns", index=False
        )

        # ----- Sheet: quarter_analysis -----
        if quarter_analysis is not None:
            quarter_analysis.to_excel(
                writer, sheet_name="quarter_analysis", index=False
            )

        # ----- Sheet: quarterly_alpha -----
        if quarterly_alpha_df is not None and not quarterly_alpha_df.empty:
            quarterly_alpha_df.to_excel(
                writer, sheet_name="quarterly_alpha", index=False
            )

        # ----- Sheet: crisis_regimes -----
        if not crisis_df.empty:
            crisis_df.reset_index().to_excel(
                writer, sheet_name="crisis_regimes", index=False
            )

        # ----- Sheet: market_regimes -----
        if not market_df.empty:
            market_df.reset_index().to_excel(
                writer, sheet_name="market_regimes", index=False
            )

        # ----- Sheet: cash_metrics -----
        if cash_metrics is not None:
            summary_df = pd.DataFrame([{
                'avg_cash_pct': cash_metrics['avg_cash_pct'],
                'avg_cash': cash_metrics['avg_cash'],
            }])
            summary_df.to_excel(
                writer, sheet_name="cash_metrics", startrow=0, index=False
            )
            cash_metrics['cash_by_quarter'].to_excel(
                writer, sheet_name="cash_metrics", startrow=4, index=False
            )

        # ----- Sheet: churn_analysis -----
        if churn_df is not None and not churn_df.empty:
            churn_df.to_excel(
                writer, sheet_name="churn_analysis", index=False
            )

        # ----- Sheet: stock_counts_by_mcap -----
        if mcap_counts_df is not None:
            mcap_counts_df.to_excel(
                writer, sheet_name="stock_counts_by_mcap", index=False
            )

        # ----- Sheet: charts (all embedded PNGs) -----
        charts_sheet = workbook.add_worksheet("charts")
        writer.sheets["charts"] = charts_sheet

        row_offset = 2

        if pf_vs_index_chart:
            charts_sheet.insert_image(f'A{row_offset}', "plot.png", {"image_data": pf_vs_index_chart})
            row_offset += _png_image_rows(pf_vs_index_chart)

        charts_sheet.insert_image(f'A{row_offset}', "plot.png", {"image_data": drawdown_chart})
        row_offset += _png_image_rows(drawdown_chart)

        if cash_pct_chart:
            charts_sheet.insert_image(f'A{row_offset}', "plot.png", {"image_data": cash_pct_chart})
            row_offset += _png_image_rows(cash_pct_chart)

        charts_sheet.insert_image(f'A{row_offset}', "plot.png", {"image_data": heatmap_chart})
        row_offset += _png_image_rows(heatmap_chart)  # dynamic: scales with number of years

        if cy_heatmap:
            charts_sheet.insert_image(f'A{row_offset}', "plot.png", {"image_data": cy_heatmap})
            row_offset += _png_image_rows(cy_heatmap)

        charts_sheet.insert_image(f'A{row_offset}', "plot.png", {"image_data": corr_heatmap})
        row_offset += _png_image_rows(corr_heatmap)

        charts_sheet.insert_image(f'A{row_offset}', "plot.png", {"image_data": bell_curve})
        row_offset += _png_image_rows(bell_curve)

        charts_sheet.insert_image(f'A{row_offset}', "plot.png", {"image_data": box_plot})
        row_offset += _png_image_rows(box_plot)

        charts_sheet.insert_image(f'A{row_offset}', "plot.png", {"image_data": churn_chart})
        row_offset += _png_image_rows(churn_chart)

        # ----- Sheet: monthly_performance (monthly returns data) -----
        if not monthly_performance_sheet.empty:
            monthly_performance_sheet.to_excel(
                writer, sheet_name="monthly_performance", index=False
            )

        # ----- Sheet: trade_results (raw data) -----
        trade_results.to_excel(
            writer, sheet_name="trade_results", index=False
        )

        # ----- Sheet: daily_portfolio_values (raw data) -----
        daily_pf.to_excel(
            writer, sheet_name="daily_portfolio_values", index=False
        )

        # ----- Sheet: portfolio_vs_index (raw data, conditional) -----
        if comparison_df is not None:
            comparison_df.to_excel(
                writer, sheet_name="portfolio_vs_index", index=False
            )

        # ----- Sheet: data_quality_issues (conditional) -----
        if data_issues is not None and not data_issues.empty:
            data_issues.to_excel(
                writer, sheet_name="data_quality_issues", index=False
            )

        # ----- Apply formatting -----
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            if sheet_name in ["calendar_returns"]:
                worksheet.set_column('A:Z', 15, format_decimal)
            elif sheet_name in ["trailing_returns"]:
                worksheet.set_column('A:A', 25)
                worksheet.set_column('B:Z', 12, format_decimal)
            elif sheet_name in ["crisis_regimes", "market_regimes"]:
                worksheet.set_column('A:A', 25)
            elif sheet_name == "churn_analysis":
                worksheet.set_column('A:A', 12)
                worksheet.set_column('B:E', 16, format_decimal)
            elif sheet_name == "stock_counts_by_mcap":
                worksheet.set_column('A:A', 12)
                worksheet.set_column('B:E', 15, format_decimal)
            elif sheet_name == "quarterly_alpha":
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:D', 18, percent_format)
            elif sheet_name == "portfolio_metrics":
                worksheet.set_column('A:Z', 18)
            elif sheet_name == "benchmark_metrics":
                worksheet.set_column('A:Z', 22)
            elif sheet_name == "cash_metrics":
                worksheet.set_column('A:C', 18, format_decimal)
            elif sheet_name == "monthly_performance":
                worksheet.set_column('A:A', 15)  # Month column
                worksheet.set_column('B:Z', 12, percent_format)  # Return columns
            elif sheet_name not in ["charts", "trade_results",
                                     "daily_portfolio_values", "portfolio_vs_index",
                                     "data_quality_issues"]:
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:Z', 12, format_decimal)

    logger.info("Report saved to: %s", output_path)
    return output_path
