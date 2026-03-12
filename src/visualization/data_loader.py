"""
Load backtest artefacts for the interactive dashboard.

Three modes:
1. **From a saved backtest run directory** — reads config_used.yaml, trade
   results and daily values from the Excel report, plus the original price
   and index data via paths stored in the config.
2. **From a get_portfolio --with-levels run directory** — reads selection.csv
   and daily_series_*.csv (partial-quarter / live portfolio monitoring).
3. **From live DataFrames** — passed in-memory when launched programmatically.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class DashboardData:
    """Container for all data the dashboard needs."""

    trade_results: pd.DataFrame
    daily_pf_values: pd.DataFrame
    price_data: pd.DataFrame
    index_data: Optional[pd.DataFrame] = None
    comparison_df: Optional[pd.DataFrame] = None
    config: dict = field(default_factory=dict)
    is_live_quarter: bool = False
    as_of_date: Optional[pd.Timestamp] = None

    @property
    def quarters(self) -> list[int]:
        return sorted(self.trade_results["quarter"].unique().tolist())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_report_xlsx(run_dir: str) -> Optional[str]:
    """Return the first backtest_report*.xlsx found in *run_dir*."""
    for f in sorted(os.listdir(run_dir)):
        if f.startswith("backtest_report") and f.endswith(".xlsx"):
            return os.path.join(run_dir, f)
    return None


def _find_daily_series_csv(run_dir: str) -> Optional[str]:
    """Return the first daily_series_*.csv found in *run_dir*."""
    pattern = os.path.join(run_dir, "daily_series_*.csv")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def _resolve_relative_path(path: str, config_dir: str) -> str:
    """Resolve a possibly-relative path against the directory the config sat in."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(config_dir, path))


def _load_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_price_and_index(config: dict, run_dir: str):
    """Load price and index data from paths stored in config_used.yaml."""
    src_dir = str(Path(run_dir).parent.parent / "src")

    price_path = config.get("price_data_path", "")
    price_data = _load_file(_resolve_relative_path(price_path, src_dir))
    price_data["date"] = pd.to_datetime(price_data["date"])

    index_data = None
    index_path = config.get("index_data_path", "")
    if index_path:
        try:
            index_data = _load_file(_resolve_relative_path(index_path, src_dir))
            index_data["date"] = pd.to_datetime(index_data["date"])
        except Exception:
            pass

    return price_data, index_data


def _build_comparison_from_daily_series(daily_pf: pd.DataFrame, initial_capital: float) -> Optional[pd.DataFrame]:
    """Build a comparison_df from the daily series' index_fund column."""
    if "index_fund" not in daily_pf.columns or "date" not in daily_pf.columns:
        return None

    pf_col = "Total_Portfolio_Value" if "Total_Portfolio_Value" in daily_pf.columns else "portfolio_value"
    if pf_col not in daily_pf.columns:
        return None

    comp = daily_pf[["date", pf_col, "index_fund"]].dropna(subset=["index_fund"]).copy()
    if comp.empty:
        return None

    comp = comp.rename(columns={pf_col: "pf_value", "index_fund": "index_fund_value"})
    comp["pf_return"] = ((comp["pf_value"] - initial_capital) / initial_capital) * 100
    comp["index_return"] = ((comp["index_fund_value"] - initial_capital) / initial_capital) * 100
    comp["alpha"] = comp["pf_return"] - comp["index_return"]
    return comp


# ---------------------------------------------------------------------------
# Loader: backtest run dir (Excel report)
# ---------------------------------------------------------------------------

def _load_from_backtest_dir(run_dir: str, config: dict) -> DashboardData:
    """Load from a full backtest run with backtest_report*.xlsx."""
    report_path = _find_report_xlsx(run_dir)
    if report_path is None:
        raise FileNotFoundError(
            f"No backtest_report*.xlsx found in {run_dir}. "
            "Re-run the backtest with generate_report: true."
        )

    trade_results = pd.read_excel(report_path, sheet_name="trade_results")
    daily_pf = pd.read_excel(report_path, sheet_name="daily_portfolio_values")

    comparison_df = None
    try:
        comparison_df = pd.read_excel(report_path, sheet_name="portfolio_vs_index")
        if "date" in comparison_df.columns:
            comparison_df["date"] = pd.to_datetime(comparison_df["date"])
    except Exception:
        pass

    for col in ("entry_date", "exit_date"):
        if col in trade_results.columns:
            trade_results[col] = pd.to_datetime(trade_results[col])

    if "date" in daily_pf.columns:
        daily_pf["date"] = pd.to_datetime(daily_pf["date"])

    price_data, index_data = _load_price_and_index(config, run_dir)

    return DashboardData(
        trade_results=trade_results,
        daily_pf_values=daily_pf,
        price_data=price_data,
        index_data=index_data,
        comparison_df=comparison_df,
        config=config,
        is_live_quarter=False,
    )


# ---------------------------------------------------------------------------
# Loader: get_portfolio --with-levels run dir (CSVs)
# ---------------------------------------------------------------------------

def _load_from_portfolio_dir(run_dir: str, config: dict) -> DashboardData:
    """Load from a get_portfolio --with-levels output folder.

    Expected layout::

        run_dir/
            config_used.yaml
            selection.csv
            daily_series_<quarter>_<date>.csv   (optional)
    """
    from config.defaults import INITIAL_CAPITAL

    selection_path = os.path.join(run_dir, "selection.csv")
    trade_results = pd.read_csv(selection_path)

    # Column renames to match backtest trade_results schema
    rename_map = {"tp_pct": "tp_pct_used", "sl_pct": "sl_pct_used"}
    trade_results = trade_results.rename(
        columns={k: v for k, v in rename_map.items() if k in trade_results.columns}
    )

    for col in ("entry_date", "exit_date"):
        if col in trade_results.columns:
            trade_results[col] = pd.to_datetime(trade_results[col])

    # Compute derived columns for closed positions
    if "stock_return" not in trade_results.columns:
        has_exit = trade_results["exit_price"].notna() & trade_results["entry_price"].notna()
        trade_results["stock_return"] = np.where(
            has_exit,
            (trade_results["exit_price"] - trade_results["entry_price"]) / trade_results["entry_price"],
            np.nan,
        )

    if "holding_period" not in trade_results.columns or trade_results["holding_period"].isna().all():
        has_dates = trade_results["exit_date"].notna() & trade_results["entry_date"].notna()
        trade_results["holding_period"] = np.where(
            has_dates,
            (trade_results["exit_date"] - trade_results["entry_date"]).dt.days,
            np.nan,
        )

    # Load daily series if available
    daily_series_path = _find_daily_series_csv(run_dir)
    if daily_series_path is not None:
        daily_pf = pd.read_csv(daily_series_path)
        daily_pf["date"] = pd.to_datetime(daily_pf["date"])
    else:
        daily_pf = pd.DataFrame(columns=["date", "Total_Portfolio_Value", "Cash_In_Hand"])

    # Infer as-of date from the daily series filename or last date
    as_of_date = None
    if daily_series_path is not None:
        basename = os.path.basename(daily_series_path)
        # daily_series_202602_20260224.csv → extract 20260224
        parts = basename.replace(".csv", "").split("_")
        if len(parts) >= 4:
            try:
                as_of_date = pd.Timestamp(parts[3])
            except Exception:
                pass
    if as_of_date is None and not daily_pf.empty:
        as_of_date = daily_pf["date"].max()

    # Build comparison_df from the daily series' index_fund column
    comparison_df = _build_comparison_from_daily_series(daily_pf, INITIAL_CAPITAL)

    price_data, index_data = _load_price_and_index(config, run_dir)

    return DashboardData(
        trade_results=trade_results,
        daily_pf_values=daily_pf,
        price_data=price_data,
        index_data=index_data,
        comparison_df=comparison_df,
        config=config,
        is_live_quarter=True,
        as_of_date=as_of_date,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_from_run_dir(run_dir: str) -> DashboardData:
    """Auto-detect folder type and load dashboard data.

    Supports both backtest run directories (with .xlsx report) and
    get_portfolio --with-levels output (with selection.csv + daily_series CSVs).
    """
    run_dir = str(Path(run_dir).resolve())

    config_path = os.path.join(run_dir, "config_used.yaml")
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    # Auto-detect: xlsx present → full backtest, else try portfolio CSVs
    if _find_report_xlsx(run_dir) is not None:
        return _load_from_backtest_dir(run_dir, config)

    selection_path = os.path.join(run_dir, "selection.csv")
    if os.path.isfile(selection_path):
        return _load_from_portfolio_dir(run_dir, config)

    raise FileNotFoundError(
        f"Could not find backtest_report*.xlsx or selection.csv in {run_dir}. "
        "Point --run-dir at a backtest output or get_portfolio --with-levels folder."
    )


def load_from_dataframes(
    trade_results: pd.DataFrame,
    daily_pf_values: pd.DataFrame,
    price_data: pd.DataFrame,
    index_data: Optional[pd.DataFrame] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    config: Optional[dict] = None,
) -> DashboardData:
    """Wrap pre-loaded DataFrames into a :class:`DashboardData` container."""
    return DashboardData(
        trade_results=trade_results,
        daily_pf_values=daily_pf_values,
        price_data=price_data,
        index_data=index_data,
        comparison_df=comparison_df,
        config=config or {},
    )


def get_quarter_trades(data: DashboardData, quarter: int) -> pd.DataFrame:
    return data.trade_results[data.trade_results["quarter"] == quarter].copy()


def get_quarter_daily(data: DashboardData, quarter: int) -> pd.DataFrame:
    df = data.daily_pf_values
    if "quarter" in df.columns:
        return df[df["quarter"] == quarter].copy()
    return df.copy()


def classify_exit(row: pd.Series) -> str:
    if row.get("TP_triggered"):
        return "TP"
    if row.get("SL_triggered"):
        return "SL"
    if row.get("regime_exit"):
        return "Regime"
    if pd.isna(row.get("exit_date")):
        return "Open"
    return "Time"
