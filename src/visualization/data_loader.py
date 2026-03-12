"""
Load backtest artefacts for the interactive dashboard.

Two modes:
1. **From a saved backtest run directory** — reads config_used.yaml, trade
   results and daily values from the Excel report, plus the original price
   and index data via paths stored in the config.
2. **From live DataFrames** — passed in-memory when launched programmatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

    # Derived helpers
    @property
    def quarters(self) -> list[int]:
        return sorted(self.trade_results["quarter"].unique().tolist())


def _find_report_xlsx(run_dir: str) -> Optional[str]:
    """Return the first backtest_report*.xlsx found in *run_dir*."""
    for f in sorted(os.listdir(run_dir)):
        if f.startswith("backtest_report") and f.endswith(".xlsx"):
            return os.path.join(run_dir, f)
    return None


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


def load_from_run_dir(run_dir: str) -> DashboardData:
    """Load all dashboard data from a backtest run output folder.

    Expected layout::

        run_dir/
            config_used.yaml
            backtest_report_*.xlsx   (with sheets: trade_results,
                                      daily_portfolio_values,
                                      portfolio_vs_index)
    """
    run_dir = str(Path(run_dir).resolve())

    # --- Config ---
    config_path = os.path.join(run_dir, "config_used.yaml")
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    config_src_dir = os.path.join(run_dir)

    # --- Excel report ---
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
        for col in ("date",):
            if col in comparison_df.columns:
                comparison_df[col] = pd.to_datetime(comparison_df[col])
    except Exception:
        pass

    # Normalise date columns
    for col in ("entry_date", "exit_date"):
        if col in trade_results.columns:
            trade_results[col] = pd.to_datetime(trade_results[col])

    if "date" in daily_pf.columns:
        daily_pf["date"] = pd.to_datetime(daily_pf["date"])

    # --- Price & index data (via paths in config) ---
    # Paths in config_used.yaml are relative to *src/* (the working dir when
    # the backtest ran). We try the run-dir first, then fall back to src/.
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

    return DashboardData(
        trade_results=trade_results,
        daily_pf_values=daily_pf,
        price_data=price_data,
        index_data=index_data,
        comparison_df=comparison_df,
        config=config,
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
    return "Time"
