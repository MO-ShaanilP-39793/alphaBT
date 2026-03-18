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
    quarterly_alpha: Optional[pd.DataFrame] = None
    config: dict = field(default_factory=dict)
    is_live_quarter: bool = False
    as_of_date: Optional[pd.Timestamp] = None

    @property
    def quarters(self) -> list[int]:
        return sorted(self.trade_results["quarter"].unique().tolist())


@dataclass
class CrossQuarterData:
    """Pre-computed artefacts for the cross-quarter dashboard mode."""

    portfolio_metrics: dict
    benchmark_metrics: Optional[dict]
    periodic_returns: pd.DataFrame
    churn_df: pd.DataFrame
    monthly_returns: pd.DataFrame
    calendar_year_df: pd.DataFrame
    drawdown_series: pd.DataFrame
    index_drawdown_series: Optional[pd.DataFrame]
    cash_by_quarter: Optional[pd.DataFrame]
    rolling_3m: pd.DataFrame
    rolling_6m: pd.DataFrame
    rolling_1y: pd.DataFrame
    rolling_3y: pd.DataFrame
    rolling_5y: pd.DataFrame


def compute_cross_quarter_data(data: DashboardData) -> CrossQuarterData:
    """Derive all cross-quarter analytics from the already-loaded data.

    Reuses functions from ``reporting.metrics`` and ``reporting.analytics``
    so the numbers match the Excel report exactly.
    """
    from reporting.metrics import (
        compute_portfolio_metrics,
        compute_benchmark_metrics,
        compute_drawdown_series,
        compute_cash_metrics,
    )
    from reporting.analytics import (
        compute_churn_analysis,
        compute_calendar_year_performance,
        compute_monthly_returns_from_daily,
        compute_portfolio_performance,
        compute_rolling_performance,
    )

    daily_pf = data.daily_pf_values.copy()
    daily_pf["date"] = pd.to_datetime(daily_pf["date"])

    pf_col = "portfolio_value" if "portfolio_value" in daily_pf.columns else "Total_Portfolio_Value"
    if pf_col != "portfolio_value":
        daily_pf = daily_pf.rename(columns={pf_col: "portfolio_value"})

    if "quarter" not in daily_pf.columns:
        daily_pf["quarter"] = 0

    # --- Portfolio-level summary metrics ---
    portfolio_metrics = compute_portfolio_metrics(daily_pf)

    # --- Benchmark metrics (optional) ---
    benchmark_metrics = None
    if data.comparison_df is not None and not data.comparison_df.empty:
        try:
            benchmark_metrics = compute_benchmark_metrics(data.comparison_df)
        except Exception:
            pass

    # --- Drawdown series ---
    drawdown_series = compute_drawdown_series(daily_pf)

    # --- Index drawdown series (optional) ---
    index_drawdown_series = None
    if data.comparison_df is not None and "index_fund_value" in data.comparison_df.columns:
        idx = data.comparison_df[["date", "index_fund_value"]].copy()
        idx = idx.rename(columns={"index_fund_value": "portfolio_value"})
        index_drawdown_series = compute_drawdown_series(idx)

    # --- Churn analysis ---
    churn_df = compute_churn_analysis(data.trade_results)

    # --- Monthly returns (from daily portfolio values) ---
    daily_pf_sorted = daily_pf.sort_values("date").set_index("date")
    daily_pf_sorted["pf_return"] = daily_pf_sorted["portfolio_value"].pct_change()

    returns_cols = {"Portfolio": daily_pf_sorted["pf_return"]}
    if data.comparison_df is not None and "index_fund_value" in data.comparison_df.columns:
        comp = data.comparison_df.copy()
        comp["date"] = pd.to_datetime(comp["date"])
        comp = comp.sort_values("date").set_index("date")
        comp["idx_return"] = comp["index_fund_value"].pct_change()
        returns_cols["Benchmark"] = comp["idx_return"]

    returns_df = pd.DataFrame(returns_cols).dropna(how="all")
    returns_df.index.name = "Date"

    monthly_returns = compute_monthly_returns_from_daily(returns_df, input_frequency="daily")

    # --- Calendar year returns ---
    calendar_year_df = compute_calendar_year_performance(returns_df, input_frequency="daily")

    # --- Periodic returns (AReturns, ARisk, Ret/Risk, DDown for each column) ---
    periodic_returns = compute_portfolio_performance(returns_df, input_frequency="daily")

    # --- Rolling returns ---
    rolling_3m = compute_rolling_performance(returns_df, "daily", 3, "monthly", annualize=False)
    rolling_6m = compute_rolling_performance(returns_df, "daily", 6, "monthly", annualize=False)
    rolling_1y = compute_rolling_performance(returns_df, "daily", 1, "yearly", annualize=True)
    rolling_3y = compute_rolling_performance(returns_df, "daily", 3, "yearly", annualize=True)
    rolling_5y = compute_rolling_performance(returns_df, "daily", 5, "yearly", annualize=True)

    # --- Cash metrics by quarter ---
    cash_by_quarter = None
    cash_result = compute_cash_metrics(daily_pf)
    if cash_result is not None:
        cash_by_quarter = cash_result["cash_by_quarter"]

    return CrossQuarterData(
        portfolio_metrics=portfolio_metrics,
        benchmark_metrics=benchmark_metrics,
        periodic_returns=periodic_returns,
        churn_df=churn_df,
        monthly_returns=monthly_returns,
        calendar_year_df=calendar_year_df,
        drawdown_series=drawdown_series,
        index_drawdown_series=index_drawdown_series,
        cash_by_quarter=cash_by_quarter,
        rolling_3m=rolling_3m,
        rolling_6m=rolling_6m,
        rolling_1y=rolling_1y,
        rolling_3y=rolling_3y,
        rolling_5y=rolling_5y,
    )


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
    """Load price and index data from paths stored in config_used.yaml.

    Returns (price_data, index_data) — either may be None if the
    underlying files are unavailable (e.g. deployed without the data/ tree).
    """
    src_dir = str(Path(run_dir).parent.parent / "src")

    price_data = None
    price_path = config.get("price_data_path", "")
    if price_path:
        try:
            price_data = _load_file(_resolve_relative_path(price_path, src_dir))
            price_data["date"] = pd.to_datetime(price_data["date"])
        except Exception:
            pass

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

    quarterly_alpha = None
    try:
        qa = pd.read_excel(report_path, sheet_name="quarterly_alpha")
        qa["Quarter"] = pd.to_numeric(qa["Quarter"], errors="coerce")
        quarterly_alpha = qa.dropna(subset=["Quarter"]).copy()
        quarterly_alpha["Quarter"] = quarterly_alpha["Quarter"].astype(int)
        for col in ("Portfolio_Return", "Benchmark_Return", "Outperformance"):
            if col in quarterly_alpha.columns:
                quarterly_alpha[col] = pd.to_numeric(quarterly_alpha[col], errors="coerce")
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
        quarterly_alpha=quarterly_alpha,
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

def _summarise_strategy(config: dict) -> str:
    """Build a short human-readable strategy label from config keys."""
    # Try explicit name first
    explicit = config.get("strategy_config", config.get("strategy", ""))
    if isinstance(explicit, str) and explicit:
        return explicit
    if isinstance(explicit, dict) and explicit.get("name"):
        return explicit["name"]

    parts: list[str] = []

    tp = config.get("tp_mode", "")
    sl = config.get("sl_mode", "")
    if tp and sl:
        if tp == sl:
            parts.append(f"{tp} TP/SL")
        else:
            parts.append(f"{tp} TP / {sl} SL")
    elif tp:
        parts.append(f"{tp} TP")
    elif sl:
        parts.append(f"{sl} SL")

    sel = config.get("selection_type", "")
    if sel == "top_k":
        k = (config.get("top_k_config") or {}).get("k", "")
        parts.append(f"top-{k}" if k else "top_k")
    elif sel:
        parts.append(sel)

    return " | ".join(parts) if parts else ""


def probe_run_dir(run_dir: str) -> Optional[dict]:
    """Return lightweight metadata for a run directory without loading data.

    Returns ``None`` if *run_dir* is not a valid run directory (i.e. missing
    ``config_used.yaml`` and no recognised data files).

    Returned dict keys:
        path, name, run_type ("backtest" | "portfolio"),
        strategy (strategy name from config), timestamp (dir mtime as ISO str).
    """
    run_dir = str(Path(run_dir).resolve())
    config_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.isfile(config_path):
        return None

    try:
        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh) or {}
    except Exception:
        return None

    if _find_report_xlsx(run_dir) is not None:
        run_type = "backtest"
    elif os.path.isfile(os.path.join(run_dir, "selection.csv")):
        run_type = "portfolio"
    else:
        return None

    strategy_name = _summarise_strategy(config)

    dir_name = os.path.basename(run_dir)
    mtime = os.path.getmtime(config_path)
    timestamp = pd.Timestamp.fromtimestamp(mtime).isoformat()

    return {
        "path": run_dir,
        "name": dir_name,
        "run_type": run_type,
        "strategy": strategy_name,
        "timestamp": timestamp,
    }


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
