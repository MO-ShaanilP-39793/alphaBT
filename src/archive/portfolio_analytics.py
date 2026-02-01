"""
Portfolio Analytics - Unified Analysis Module
=============================================

Combines the strengths of:
- Daily_Summary_Stats notebook: Excel reports, regime analysis, multi-strategy comparison
- addtl_bt_fns.py: Benchmark metrics, calendar-based rolling, trade-level analysis

Usage:
    from portfolio_analytics import PortfolioAnalyzer
    
    # From returns DataFrame (like notebook)
    analyzer = PortfolioAnalyzer.from_returns(returns_df, timeline="daily")
    
    # From portfolio values (like module)
    analyzer = PortfolioAnalyzer.from_portfolio_values(daily_pf_df)
    
    # Generate comprehensive Excel report
    analyzer.generate_excel_report("output.xlsx")
    
    # Or access individual metrics
    metrics = analyzer.compute_all_metrics()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from datetime import datetime
from typing import Optional, Dict, List, Union, Tuple
from matplotlib.ticker import StrMethodFormatter


# =============================================================================
# CORE PORTFOLIO ANALYZER CLASS
# =============================================================================

class PortfolioAnalyzer:
    """
    Unified portfolio analysis combining notebook and module capabilities.
    
    Supports:
    - Multiple strategies/return series comparison
    - Benchmark comparison with relative metrics
    - Crisis and market regime analysis
    - Comprehensive Excel report generation
    - Individual metric computation and plotting
    """
    
    # Default crisis regimes (can be customized)
    DEFAULT_CRISIS_REGIMES = {
        "Global Financial Crisis": {
            "crisis_start": "08-01-2008", "crisis_end": "27-10-2008",
            "recovery_start": "28-10-2008", "recovery_end": "31-10-2009"
        },
        "Taper Tantrum": {
            "crisis_start": "01-01-2013", "crisis_end": "30-08-2013",
            "recovery_start": "31-08-2013", "recovery_end": "31-08-2014"
        },
        "Yuan Devaluation": {
            "crisis_start": "03-08-2015", "crisis_end": "29-02-2016",
            "recovery_start": "01-03-2016", "recovery_end": "28-02-2017"
        },
        "Covid Crash": {
            "crisis_start": "20-02-2020", "crisis_end": "31-03-2020",
            "recovery_start": "01-04-2020", "recovery_end": "31-12-2020"
        }
    }
    
    # Default market regimes
    DEFAULT_MARKET_REGIMES = {
        "Bear Regime 01": {"start": "01-01-2008", "end": "30-11-2008"},
        "Recovery Regime 01": {"start": "01-12-2008", "end": "31-10-2010"},
        "Bear Regime 02": {"start": "01-11-2010", "end": "31-12-2011"},
        "Recovery Regime 02": {"start": "01-01-2012", "end": "28-02-2014"},
        "Bull Regime 02": {"start": "01-03-2014", "end": "28-02-2015"},
        "Bear Regime 03": {"start": "01-03-2015", "end": "29-02-2016"},
        "Recovery Regime 03": {"start": "01-03-2016", "end": "31-12-2016"},
        "Bull Regime 03": {"start": "01-01-2017", "end": "31-01-2020"},
        "Bear Regime 04": {"start": "01-02-2020", "end": "31-03-2020"},
        "Recovery Regime 04": {"start": "01-04-2020", "end": "31-10-2020"},
        "Bull Regime 04": {"start": "01-11-2020", "end": "28-02-2022"},
        "Bear Regime 05": {"start": "01-03-2022", "end": "31-07-2022"},
        "Bull Regime 05": {"start": "01-08-2022", "end": "30-09-2024"},
        "Bear Regime 06": {"start": "01-10-2024", "end": "28-02-2025"}
    }
    
    def __init__(
        self,
        returns_df: pd.DataFrame,
        timeline: str = "daily",
        benchmark_col: Optional[str] = None,
        risk_free_rate: float = 0.065
    ):
        """
        Initialize PortfolioAnalyzer.
        
        Parameters:
        - returns_df: DataFrame with DatetimeIndex and return columns
        - timeline: "daily", "monthly", or "yearly"
        - benchmark_col: Column name to use as benchmark (for relative metrics)
        - risk_free_rate: Annual risk-free rate (default 6.5% for India)
        """
        self.returns_df = returns_df.copy()
        self.timeline = timeline.strip().lower()
        self.benchmark_col = benchmark_col
        self.risk_free_rate = risk_free_rate
        
        # Validate timeline
        if self.timeline not in ["daily", "monthly", "yearly"]:
            raise ValueError("timeline must be 'daily', 'monthly', or 'yearly'")
        
        # Set annualization factor
        self.annual_factor = {"daily": 252, "monthly": 12, "yearly": 1}[self.timeline]
        
        # Ensure datetime index
        if not isinstance(self.returns_df.index, pd.DatetimeIndex):
            self.returns_df.index = pd.to_datetime(self.returns_df.index)
        
        # Store strategy columns (excluding benchmark if specified)
        self.strategy_cols = [c for c in self.returns_df.columns if c != benchmark_col]
        
        # Pre-compute monthly returns if not already monthly
        self._monthly_returns = None
        
    @classmethod
    def from_returns(
        cls,
        returns_df: pd.DataFrame,
        timeline: str = "daily",
        benchmark_col: Optional[str] = None,
        risk_free_rate: float = 0.065
    ) -> "PortfolioAnalyzer":
        """
        Create analyzer from returns DataFrame (like notebook input).
        
        Parameters:
        - returns_df: DataFrame with date index and return columns (as decimals, e.g., 0.01 for 1%)
        """
        return cls(returns_df, timeline, benchmark_col, risk_free_rate)
    
    @classmethod
    def from_portfolio_values(
        cls,
        values_df: pd.DataFrame,
        date_col: str = "date",
        value_cols: Optional[List[str]] = None,
        benchmark_col: Optional[str] = None,
        risk_free_rate: float = 0.065
    ) -> "PortfolioAnalyzer":
        """
        Create analyzer from portfolio values DataFrame (like module input).
        
        Parameters:
        - values_df: DataFrame with date column and value columns
        - date_col: Name of date column
        - value_cols: List of value columns to analyze (None = all numeric except date)
        - benchmark_col: Column name to use as benchmark
        """
        df = values_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Convert values to returns
        returns_df = df[value_cols].pct_change().dropna()
        
        return cls(returns_df, "daily", benchmark_col, risk_free_rate)
    
    # =========================================================================
    # MONTHLY RETURNS COMPUTATION
    # =========================================================================
    
    @property
    def monthly_returns(self) -> pd.DataFrame:
        """Get or compute monthly returns."""
        if self._monthly_returns is None:
            if self.timeline == "monthly":
                self._monthly_returns = self.returns_df.copy()
            else:
                self._monthly_returns = self._compute_monthly_returns()
        return self._monthly_returns
    
    def _compute_monthly_returns(self) -> pd.DataFrame:
        """Convert daily/yearly returns to monthly."""
        df = self.returns_df.copy()
        df = df.reset_index()
        date_col = df.columns[0]
        df["YM"] = df[date_col].dt.to_period("M")
        df = df.set_index(date_col)
        
        # Filter months with sufficient data
        min_days = 15 if self.timeline == "daily" else 1
        mask = df["YM"].value_counts() > min_days
        valid_months = mask[mask].index
        df = df[df["YM"].isin(valid_months)]
        
        # Compute monthly returns
        monthly = df.groupby("YM").apply(
            lambda x: (1 + x.drop(columns=["YM"])).prod() - 1
        )
        monthly.index = monthly.index.to_timestamp("M")
        monthly.index.name = "Date"
        
        return monthly
    
    # =========================================================================
    # CORE PERFORMANCE METRICS (from notebook)
    # =========================================================================
    
    def compute_performance_metrics(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute comprehensive performance metrics for all strategies.
        
        Parameters:
        - start_year: Optional start year filter
        - end_year: Optional end year filter
        
        Returns:
        - DataFrame with metrics as columns, strategies as rows
        """
        df = self._filter_by_years(self.returns_df, start_year, end_year)
        
        # Cumulative returns
        cum_ret = (1 + df).cumprod().iloc[-1] - 1
        
        # Annualized returns
        ann_ret = (1 + cum_ret) ** (self.annual_factor / len(df)) - 1
        
        # Annualized volatility
        ann_vol = df.std() * np.sqrt(self.annual_factor)
        
        # Sharpe ratio
        sharpe = ann_ret / ann_vol
        
        # Downside deviation and Sortino
        downside_dev = df[df < 0].std() * np.sqrt(self.annual_factor)
        sortino = ann_ret / downside_dev
        
        # Maximum drawdown
        max_dd = self._compute_max_drawdown(df)
        
        # Calmar ratio (from module)
        calmar = ann_ret / abs(max_dd)
        
        # VaR 95% (from module)
        var_95 = df.quantile(0.05)
        
        # Skewness and Kurtosis
        skewness = df.skew()
        kurtosis = df.kurt()
        
        # Up/Down periods
        up_pct = (df > 0).sum() / len(df) * 100
        down_pct = 100 - up_pct
        
        # Best/Worst periods (from module)
        best_period = df.max() * 100
        worst_period = df.min() * 100
        
        # Build results DataFrame
        metrics = pd.DataFrame({
            "G-Rs100": (cum_ret * 100) + 100,
            "AReturns": ann_ret * 100,
            "ARisk": ann_vol * 100,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "DDown": max_dd * 100,
            "VaR_95": var_95 * 100,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "%Up_Periods": up_pct,
            "%Down_Periods": down_pct,
            "Best_Period": best_period,
            "Worst_Period": worst_period
        }).round(2)
        
        return metrics
    
    def _compute_max_drawdown(self, returns_df: pd.DataFrame) -> pd.Series:
        """Compute maximum drawdown for each column."""
        log_dd = np.log(1 + returns_df).cumsum() - np.log(1 + returns_df).cumsum().cummax()
        return (np.exp(log_dd) - 1).min()
    
    def _filter_by_years(
        self,
        df: pd.DataFrame,
        start_year: Optional[int],
        end_year: Optional[int]
    ) -> pd.DataFrame:
        """Filter DataFrame by year range."""
        if start_year is None and end_year is None:
            return df
        
        result = df.copy()
        if start_year:
            result = result[result.index.year >= start_year]
        if end_year:
            result = result[result.index.year <= end_year]
        
        return result
    
    # =========================================================================
    # BENCHMARK COMPARISON METRICS (from module)
    # =========================================================================
    
    def compute_benchmark_metrics(self, strategy_col: Optional[str] = None) -> Dict:
        """
        Compute benchmark-relative metrics for a strategy.
        
        Parameters:
        - strategy_col: Strategy column to analyze (uses first non-benchmark if None)
        
        Returns:
        - Dictionary with beta, tracking_error, information_ratio, up/down_capture
        """
        if self.benchmark_col is None:
            raise ValueError("No benchmark column specified")
        
        if strategy_col is None:
            strategy_col = self.strategy_cols[0]
        
        df = self.returns_df[[strategy_col, self.benchmark_col]].dropna()
        pf_ret = df[strategy_col]
        bm_ret = df[self.benchmark_col]
        
        # Beta
        covariance = pf_ret.cov(bm_ret)
        bm_variance = bm_ret.var()
        beta = covariance / bm_variance if bm_variance > 0 else 1
        
        # Tracking Error
        excess_ret = pf_ret - bm_ret
        tracking_error = excess_ret.std() * np.sqrt(self.annual_factor)
        
        # Information Ratio
        excess_annual = excess_ret.mean() * self.annual_factor
        info_ratio = excess_annual / tracking_error if tracking_error > 0 else 0
        
        # Up/Down Capture
        up_days = df[bm_ret > 0]
        down_days = df[bm_ret < 0]
        
        up_capture = (up_days[strategy_col].mean() / up_days[self.benchmark_col].mean() * 100
                      if len(up_days) > 0 and up_days[self.benchmark_col].mean() != 0 else 100)
        down_capture = (down_days[strategy_col].mean() / down_days[self.benchmark_col].mean() * 100
                        if len(down_days) > 0 and down_days[self.benchmark_col].mean() != 0 else 100)
        
        # Correlation
        correlation = pf_ret.corr(bm_ret)
        
        # Alpha (annualized excess return)
        pf_annual = (1 + pf_ret).prod() ** (self.annual_factor / len(pf_ret)) - 1
        bm_annual = (1 + bm_ret).prod() ** (self.annual_factor / len(bm_ret)) - 1
        alpha = pf_annual - bm_annual
        
        return {
            "strategy": strategy_col,
            "benchmark": self.benchmark_col,
            "alpha_pct": round(alpha * 100, 2),
            "beta": round(beta, 3),
            "tracking_error_pct": round(tracking_error * 100, 2),
            "information_ratio": round(info_ratio, 3),
            "up_capture_pct": round(up_capture, 2),
            "down_capture_pct": round(down_capture, 2),
            "correlation": round(correlation, 3)
        }
    
    def compute_all_benchmark_metrics(self) -> pd.DataFrame:
        """Compute benchmark metrics for all strategies."""
        if self.benchmark_col is None:
            raise ValueError("No benchmark column specified")
        
        results = []
        for col in self.strategy_cols:
            metrics = self.compute_benchmark_metrics(col)
            results.append(metrics)
        
        return pd.DataFrame(results).set_index("strategy")
    
    # =========================================================================
    # ROLLING RETURNS (both trading-day and calendar-based from module)
    # =========================================================================
    
    def compute_rolling_returns(
        self,
        roll_period: int = 1,
        roll_type: str = "yearly",
        calendar_based: bool = False
    ) -> pd.DataFrame:
        """
        Compute rolling returns with probability distributions.
        
        Parameters:
        - roll_period: Number of periods to roll
        - roll_type: "daily", "monthly", or "yearly"
        - calendar_based: If True, use calendar months instead of trading days
        
        Returns:
        - DataFrame with rolling return statistics
        """
        roll_type = roll_type.strip().lower()
        
        if calendar_based:
            return self._compute_calendar_rolling(roll_period)
        else:
            return self._compute_trading_day_rolling(roll_period, roll_type)
    
    def _compute_trading_day_rolling(self, roll_period: int, roll_type: str) -> pd.DataFrame:
        """Compute rolling returns based on trading days (from notebook)."""
        # Scale factor mapping
        scale_factors = {
            ("daily", "daily"): 1, ("daily", "monthly"): 21, ("daily", "yearly"): 252,
            ("monthly", "monthly"): 1, ("monthly", "yearly"): 12,
            ("yearly", "yearly"): 1
        }
        
        key = (self.timeline, roll_type)
        if key not in scale_factors:
            raise ValueError(f"Cannot compute {roll_type} rolling from {self.timeline} data")
        
        scale = scale_factors[key]
        window = roll_period * scale
        annualization = self.annual_factor / window
        
        if len(self.returns_df) < window:
            raise ValueError("Data size is less than roll period")
        
        # Compute rolling returns
        rolling_ret = (1 + self.returns_df).rolling(window).apply(np.prod, raw=True) - 1
        rolling_ret = rolling_ret.dropna()
        ann_rolling = (1 + rolling_ret) ** annualization - 1
        
        # Compute statistics
        summary = rolling_ret.describe().drop("std")
        summary.iloc[1:] = ((1 + summary.iloc[1:]) ** annualization - 1)
        
        # Probability buckets
        summary.loc["<0%P"] = (ann_rolling < 0).mean()
        summary.loc["0-10%P"] = ((ann_rolling >= 0) & (ann_rolling < 0.1)).mean()
        summary.loc["10-20%P"] = ((ann_rolling >= 0.1) & (ann_rolling < 0.2)).mean()
        summary.loc[">20%P"] = (ann_rolling >= 0.2).mean()
        
        # Scale to percentages
        summary.iloc[1:] = summary.iloc[1:] * 100
        
        return summary.round(2).T
    
    def _compute_calendar_rolling(self, months: int) -> pd.DataFrame:
        """Compute rolling returns based on calendar months (from module)."""
        from pandas.tseries.offsets import DateOffset
        
        df = self.returns_df.copy()
        results = {}
        
        for col in df.columns:
            returns = []
            for current_date in df.index:
                target_date = current_date - DateOffset(months=months)
                available = df.index[df.index <= target_date]
                
                if len(available) == 0:
                    returns.append(np.nan)
                else:
                    closest = available[-1]
                    period_data = df.loc[closest:current_date, col]
                    cum_ret = (1 + period_data).prod() - 1
                    returns.append(cum_ret)
            
            results[col] = returns
        
        result_df = pd.DataFrame(results, index=df.index)
        
        # Annualize if period > 12 months
        if months >= 12:
            years = months / 12
            result_df = (1 + result_df) ** (1 / years) - 1
        
        return result_df.dropna()
    
    # =========================================================================
    # CALENDAR YEAR RETURNS (from notebook)
    # =========================================================================
    
    def compute_calendar_year_returns(self) -> pd.DataFrame:
        """Compute calendar year returns."""
        df = self.returns_df.copy()
        df = df.reset_index()
        date_col = df.columns[0]
        df["Year"] = pd.to_datetime(df[date_col]).dt.to_period("Y")
        
        # Filter years with sufficient data
        cutoff = 230 if self.timeline == "daily" else 11
        valid_years = df["Year"].value_counts()[df["Year"].value_counts() > cutoff].index
        df = df[df["Year"].isin(valid_years)]
        
        # Compute returns by year
        cy_returns = df.groupby("Year").apply(
            lambda x: (1 + x.drop(columns=[date_col, "Year"])).prod() - 1
        ) * 100
        
        cy_returns.index = cy_returns.index.astype(str)
        return cy_returns.round(2)
    
    # =========================================================================
    # TRAILING RETURNS (from notebook)
    # =========================================================================
    
    def compute_trailing_returns(self) -> pd.DataFrame:
        """Compute point-in-time trailing returns."""
        periods = {
            "1-m": self.annual_factor // 12,
            "3-m": self.annual_factor // 4,
            "6-m": self.annual_factor // 2,
            "1-year": self.annual_factor,
            "3-years": self.annual_factor * 3,
            "5-years": self.annual_factor * 5,
            "10-years": self.annual_factor * 10
        }
        
        results = {}
        for period_name, lookback in periods.items():
            if len(self.returns_df) >= lookback:
                ret = (1 + self.returns_df.tail(lookback)).prod() - 1
                
                # Annualize multi-year periods
                if "year" in period_name and lookback > self.annual_factor:
                    years = lookback / self.annual_factor
                    ret = (1 + ret) ** (1 / years) - 1
                
                results[period_name] = ret * 100
            else:
                results[period_name] = pd.Series(0, index=self.returns_df.columns)
        
        return pd.DataFrame(results).round(2)
    
    # =========================================================================
    # REGIME ANALYSIS (from notebook)
    # =========================================================================
    
    def compute_crisis_regime_returns(
        self,
        crisis_regimes: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Compute returns during crisis and recovery periods.
        
        Parameters:
        - crisis_regimes: Dict of regime definitions (uses defaults if None)
        """
        if crisis_regimes is None:
            crisis_regimes = self.DEFAULT_CRISIS_REGIMES
        
        results = []
        for regime_name, periods in crisis_regimes.items():
            # Crisis period
            crisis_start = pd.to_datetime(periods["crisis_start"], format="%d-%m-%Y")
            crisis_end = pd.to_datetime(periods["crisis_end"], format="%d-%m-%Y")
            
            crisis_ret = self._period_return(crisis_start, crisis_end)
            if crisis_ret is not None and not (crisis_ret == 0).all():
                row = {"Regime": regime_name, "Start_Date": crisis_start, "End_Date": crisis_end}
                row.update((crisis_ret * 100).round(2).to_dict())
                results.append(row)
            
            # Recovery period
            recovery_start = pd.to_datetime(periods["recovery_start"], format="%d-%m-%Y")
            recovery_end = pd.to_datetime(periods["recovery_end"], format="%d-%m-%Y")
            
            recovery_ret = self._period_return(recovery_start, recovery_end)
            if recovery_ret is not None and not (recovery_ret == 0).all():
                row = {"Regime": f"Recovery {regime_name}", "Start_Date": recovery_start, "End_Date": recovery_end}
                row.update((recovery_ret * 100).round(2).to_dict())
                results.append(row)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.set_index("Regime")
        return df
    
    def compute_market_regime_returns(
        self,
        market_regimes: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Compute returns during market regimes (bull/bear/recovery).
        
        Parameters:
        - market_regimes: Dict of regime definitions (uses defaults if None)
        """
        if market_regimes is None:
            market_regimes = self.DEFAULT_MARKET_REGIMES
        
        results = []
        for regime_name, periods in market_regimes.items():
            reg_start = pd.to_datetime(periods["start"], format="%d-%m-%Y")
            reg_end = pd.to_datetime(periods["end"], format="%d-%m-%Y")
            
            ret = self._period_return(reg_start, reg_end)
            if ret is not None and not (ret == 0).all():
                row = {"Regime": regime_name, "Start_Date": reg_start, "End_Date": reg_end}
                row.update((ret * 100).round(2).to_dict())
                results.append(row)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.set_index("Regime")
        return df
    
    def _period_return(self, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
        """Compute return for a specific period."""
        mask = (self.returns_df.index >= start) & (self.returns_df.index <= end)
        period_data = self.returns_df[mask]
        
        if len(period_data) == 0:
            return None
        
        return (1 + period_data).prod() - 1
    
    # =========================================================================
    # UP/DOWN PERIODS ANALYSIS (from notebook)
    # =========================================================================
    
    def compute_up_down_periods(self) -> pd.DataFrame:
        """Compute count of up and down periods."""
        monthly = self.monthly_returns
        
        up_months = (monthly > 0).sum()
        down_months = len(monthly) - up_months
        
        return pd.DataFrame({
            "Up_Periods": up_months,
            "Down_Periods": down_months
        })
    
    # =========================================================================
    # DRAWDOWN ANALYSIS (enhanced from both)
    # =========================================================================
    
    def compute_drawdown_series(self) -> pd.DataFrame:
        """Compute drawdown series for all strategies."""
        cum_ret = (1 + self.returns_df).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        
        return drawdown
    
    def get_drawdown_details(self) -> pd.DataFrame:
        """Get detailed drawdown statistics."""
        dd = self.compute_drawdown_series()
        
        results = {}
        for col in dd.columns:
            col_dd = dd[col]
            max_dd_idx = col_dd.idxmin()
            
            # Find drawdown start (last peak before max drawdown)
            prior_data = col_dd.loc[:max_dd_idx]
            peak_idx = (prior_data == 0).iloc[::-1].idxmax() if (prior_data == 0).any() else prior_data.index[0]
            
            # Find recovery (first time back to 0 after max drawdown)
            post_data = col_dd.loc[max_dd_idx:]
            recovery_idx = (post_data >= 0).idxmax() if (post_data >= 0).any() else None
            
            results[col] = {
                "max_drawdown_pct": round(col_dd.min() * 100, 2),
                "max_dd_date": max_dd_idx,
                "dd_start_date": peak_idx,
                "recovery_date": recovery_idx,
                "dd_duration_days": (max_dd_idx - peak_idx).days if peak_idx else None,
                "recovery_days": (recovery_idx - max_dd_idx).days if recovery_idx else None
            }
        
        return pd.DataFrame(results).T
    
    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    
    def compute_correlation_matrix(self) -> pd.DataFrame:
        """Compute correlation matrix between all strategies."""
        return self.returns_df.corr().round(3)
    
    # =========================================================================
    # PLOTTING FUNCTIONS
    # =========================================================================
    
    def plot_growth_of_wealth(
        self,
        initial_value: float = 10000,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> io.BytesIO:
        """Plot growth of wealth chart."""
        monthly = self.monthly_returns
        growth = (1 + monthly).cumprod() * initial_value
        growth = growth.reset_index()
        
        date_col = growth.columns[0]
        
        plt.figure(figsize=(12, 6))
        colors = plt.colormaps["tab10"].colors
        
        for idx, col in enumerate(growth.columns[1:]):
            plt.plot(growth[date_col], growth[col], label=col, 
                     linewidth=1.5, color=colors[idx % len(colors)])
        
        plt.ylabel(f"Growth of Rs {initial_value:,.0f}", fontsize=10)
        plt.title(title or f"Growth of Rs {initial_value:,.0f}", fontsize=12)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('Rs. {x:,.0f}'))
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.legend(fontsize=8)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(buf.getvalue())
            buf.seek(0)
        
        return buf
    
    def plot_drawdown(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> io.BytesIO:
        """Plot drawdown chart."""
        monthly = self.monthly_returns
        dd = self.compute_drawdown_series()
        
        # Use monthly for smoother chart
        dd_monthly = dd.resample('ME').last() if self.timeline == "daily" else dd
        dd_monthly = dd_monthly.reset_index()
        date_col = dd_monthly.columns[0]
        
        plt.figure(figsize=(12, 6))
        colors = plt.colormaps["tab10"].colors
        
        for idx, col in enumerate(dd_monthly.columns[1:]):
            plt.plot(dd_monthly[date_col], dd_monthly[col] * 100, label=col,
                     linewidth=0.75, color=colors[idx % len(colors)])
        
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.ylabel("Drawdown (%)", fontsize=10)
        plt.title(title or "Drawdowns", fontsize=12)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}%'))
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.legend(fontsize=8)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(buf.getvalue())
            buf.seek(0)
        
        return buf
    
    def plot_calendar_year_heatmap(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> io.BytesIO:
        """Plot calendar year returns heatmap."""
        cy_ret = self.compute_calendar_year_returns()
        
        n_rows, n_cols = cy_ret.shape
        fig_width = max(6, n_cols * 0.8)
        fig_height = max(4, n_rows * 0.3)
        
        plt.figure(figsize=(fig_width, fig_height))
        
        ax = sns.heatmap(
            cy_ret,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            cbar_kws={"label": "Return (%)"}
        )
        
        plt.title(title or "Calendar Year Returns Heatmap", fontsize=12)
        ax.set_ylabel("Year", fontsize=10)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(buf.getvalue())
            buf.seek(0)
        
        return buf
    
    def plot_correlation_heatmap(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> io.BytesIO:
        """Plot correlation matrix heatmap."""
        corr = self.compute_correlation_matrix()
        
        plt.figure(figsize=(8, 6))
        
        ax = sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            cbar_kws={"label": "Correlation"}
        )
        
        plt.title(title or "Correlation Matrix", fontsize=12)
        plt.xticks(rotation=45, fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(buf.getvalue())
            buf.seek(0)
        
        return buf
    
    def plot_return_distribution(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> io.BytesIO:
        """Plot return distribution (bell curve)."""
        monthly = self.monthly_returns
        long_df = monthly.melt(var_name="Strategy", value_name="Return")
        
        plt.figure(figsize=(12, 6))
        colors = plt.colormaps["tab10"].colors
        palette = {col: colors[i % len(colors)] for i, col in enumerate(monthly.columns)}
        
        sns.kdeplot(
            data=long_df,
            x='Return',
            hue='Strategy',
            palette=palette,
            common_norm=False,
            linewidth=2
        )
        
        plt.title(title or "Distribution of Monthly Returns", fontsize=12)
        plt.xlabel("Monthly Return")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(buf.getvalue())
            buf.seek(0)
        
        return buf
    
    def plot_box_whisker(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> io.BytesIO:
        """Plot box-whisker chart of monthly returns."""
        monthly = self.monthly_returns
        
        plt.figure(figsize=(12, 6))
        colors = plt.colormaps["tab10"].colors
        palette = {col: colors[i % len(colors)] for i, col in enumerate(monthly.columns)}
        
        sns.boxplot(data=monthly, palette=palette)
        
        plt.title(title or "Box-Whisker Plot of Monthly Returns", fontsize=12)
        plt.ylabel("Monthly Return")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(buf.getvalue())
            buf.seek(0)
        
        return buf
    
    def plot_rolling_sharpe(
        self,
        window: int = 126,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> io.BytesIO:
        """Plot rolling Sharpe ratio (from module)."""
        df = self.returns_df.copy()
        daily_rf = self.risk_free_rate / self.annual_factor
        
        excess_ret = df - daily_rf
        rolling_mean = excess_ret.rolling(window).mean() * self.annual_factor
        rolling_std = df.rolling(window).std() * np.sqrt(self.annual_factor)
        rolling_sharpe = rolling_mean / rolling_std
        
        rolling_sharpe = rolling_sharpe.dropna()
        
        plt.figure(figsize=(14, 5))
        colors = plt.colormaps["tab10"].colors
        
        for idx, col in enumerate(rolling_sharpe.columns):
            plt.plot(rolling_sharpe.index, rolling_sharpe[col], 
                     label=col, linewidth=1, color=colors[idx % len(colors)])
        
        plt.axhline(0, color='black', linestyle='-', alpha=0.5)
        plt.axhline(1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        
        plt.xlabel("Date")
        plt.ylabel("Sharpe Ratio")
        plt.title(title or f"{window}-Day Rolling Sharpe Ratio", fontsize=12)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(buf.getvalue())
            buf.seek(0)
        
        return buf
    
    # =========================================================================
    # COMPREHENSIVE EXCEL REPORT (from notebook, enhanced)
    # =========================================================================
    
    def generate_excel_report(
        self,
        output_path: str,
        sub_periods: Optional[List[Tuple[int, int]]] = None,
        include_charts: bool = True,
        chart_title_suffix: str = ""
    ):
        """
        Generate comprehensive Excel report combining all analyses.
        
        Parameters:
        - output_path: Path for output Excel file
        - sub_periods: List of (start_year, end_year) tuples for sub-period analysis
        - include_charts: Whether to embed charts in Excel
        - chart_title_suffix: Suffix to add to chart titles
        """
        # Compute all data
        inception_metrics = self.compute_performance_metrics()
        inception_metrics.index.name = "Since_Inception"
        
        # Sub-periods
        sub_period_metrics = []
        if sub_periods:
            for start_y, end_y in sub_periods:
                metrics = self.compute_performance_metrics(start_y, end_y)
                metrics.index.name = f"{start_y}_{end_y}"
                sub_period_metrics.append(metrics)
        
        # Rolling returns
        roll_1y = self.compute_rolling_returns(1, "yearly")
        roll_1y.index.name = "Rolling_1Y"
        
        roll_3y = self.compute_rolling_returns(3, "yearly")
        roll_3y.index.name = "Rolling_3Y"
        
        roll_5y = self.compute_rolling_returns(5, "yearly")
        roll_5y.index.name = "Rolling_5Y"
        
        # Calendar year returns
        cy_returns = self.compute_calendar_year_returns()
        cy_returns.index.name = "Calendar_Year"
        
        # Monthly returns
        monthly = self.monthly_returns.copy()
        monthly.index = monthly.index.date
        monthly.index.name = "Date"
        
        # Up/down months
        up_down = self.compute_up_down_periods()
        
        # Trailing returns
        trailing = self.compute_trailing_returns()
        trailing.index.name = "Strategy"
        
        # Regime analysis
        crisis_regimes = self.compute_crisis_regime_returns()
        market_regimes = self.compute_market_regime_returns()
        
        # Benchmark metrics (if applicable)
        benchmark_metrics = None
        if self.benchmark_col:
            benchmark_metrics = self.compute_all_benchmark_metrics()
        
        # Correlation matrix
        correlation = self.compute_correlation_matrix()
        
        # Generate charts
        charts = {}
        if include_charts:
            charts['growth'] = self.plot_growth_of_wealth(
                title=f"Growth of Rs 10,000 {chart_title_suffix}"
            )
            charts['drawdown'] = self.plot_drawdown(
                title=f"Drawdowns {chart_title_suffix}"
            )
            charts['cy_heatmap'] = self.plot_calendar_year_heatmap()
            charts['correlation'] = self.plot_correlation_heatmap()
            charts['distribution'] = self.plot_return_distribution()
            charts['box_plot'] = self.plot_box_whisker()
            
            if self.timeline == "daily":
                charts['rolling_sharpe'] = self.plot_rolling_sharpe()
        
        # Write to Excel
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Formats
            decimal_fmt = workbook.add_format({'num_format': '0.00'})
            percent_fmt = workbook.add_format({'num_format': '0.0%'})
            date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            
            # Input returns
            input_returns = self.returns_df.copy()
            input_returns.index = input_returns.index.date
            input_returns.reset_index().to_excel(writer, sheet_name="input_returns", index=False)
            
            # Periodic returns
            start_row = 0
            inception_metrics.reset_index().to_excel(
                writer, sheet_name="periodic_returns", startrow=start_row, index=False
            )
            
            for metrics in sub_period_metrics:
                start_row += len(inception_metrics) + 5
                metrics.reset_index().to_excel(
                    writer, sheet_name="periodic_returns", startrow=start_row, index=False
                )
            
            # Benchmark metrics
            if benchmark_metrics is not None:
                start_row += len(sub_period_metrics[-1]) + 5 if sub_period_metrics else len(inception_metrics) + 5
                benchmark_metrics.reset_index().to_excel(
                    writer, sheet_name="periodic_returns", startrow=start_row, index=False
                )
            
            # Rolling returns
            start_row = 0
            roll_1y.reset_index().to_excel(
                writer, sheet_name="rolling_returns", startrow=start_row, index=False
            )
            start_row += len(roll_1y) + 5
            roll_3y.reset_index().to_excel(
                writer, sheet_name="rolling_returns", startrow=start_row, index=False
            )
            start_row += len(roll_3y) + 5
            roll_5y.reset_index().to_excel(
                writer, sheet_name="rolling_returns", startrow=start_row, index=False
            )
            
            # Calendar year returns
            cy_returns.reset_index().to_excel(writer, sheet_name="calendar_returns", index=False)
            
            # Monthly returns
            monthly.reset_index().to_excel(writer, sheet_name="monthly_returns", index=False)
            
            # Up/down months
            up_down.reset_index().to_excel(writer, sheet_name="up_down_periods", index=False)
            
            # Trailing returns
            trailing.reset_index().to_excel(writer, sheet_name="trailing_returns", index=False)
            
            # Correlation
            correlation.reset_index().to_excel(writer, sheet_name="correlation", index=False)
            
            # Crisis regimes
            if len(crisis_regimes) > 0:
                crisis_regimes.reset_index().to_excel(writer, sheet_name="crisis_regimes", index=False)
            
            # Market regimes
            if len(market_regimes) > 0:
                market_regimes.reset_index().to_excel(writer, sheet_name="market_regimes", index=False)
            
            # Charts
            if include_charts:
                # Charts sheet 1
                ws1 = workbook.add_worksheet("charts_01")
                writer.sheets["charts_01"] = ws1
                ws1.insert_image('A2', "plot.png", {"image_data": charts['distribution']})
                ws1.insert_image('A35', "plot.png", {"image_data": charts['box_plot']})
                ws1.insert_image('N2', "plot.png", {"image_data": charts['cy_heatmap']})
                
                # Charts sheet 2
                ws2 = workbook.add_worksheet("charts_02")
                writer.sheets["charts_02"] = ws2
                ws2.insert_image('A2', "plot.png", {"image_data": charts['drawdown']})
                ws2.insert_image('A35', "plot.png", {"image_data": charts['growth']})
                ws2.insert_image('N2', "plot.png", {"image_data": charts['correlation']})
                
                if 'rolling_sharpe' in charts:
                    ws2.insert_image('N35', "plot.png", {"image_data": charts['rolling_sharpe']})
            
            # Apply formatting to sheets
            for sheet_name in writer.sheets:
                ws = writer.sheets[sheet_name]
                if sheet_name in ["input_returns", "monthly_returns"]:
                    ws.set_column('A:A', 15, date_fmt)
                    ws.set_column('B:Z', 12, percent_fmt)
                elif sheet_name not in ["charts_01", "charts_02"]:
                    ws.set_column('A:A', 20)
                    ws.set_column('B:Z', 12, decimal_fmt)
        
        print(f"Report saved to: {output_path}")
    
    # =========================================================================
    # COMPUTE ALL METRICS (convenience method)
    # =========================================================================
    
    def compute_all_metrics(self) -> Dict:
        """
        Compute all available metrics and return as a dictionary.
        
        Returns:
        - Dictionary with all computed metrics and DataFrames
        """
        results = {
            "performance_metrics": self.compute_performance_metrics(),
            "calendar_year_returns": self.compute_calendar_year_returns(),
            "trailing_returns": self.compute_trailing_returns(),
            "rolling_1y": self.compute_rolling_returns(1, "yearly"),
            "rolling_3y": self.compute_rolling_returns(3, "yearly"),
            "rolling_5y": self.compute_rolling_returns(5, "yearly"),
            "up_down_periods": self.compute_up_down_periods(),
            "correlation_matrix": self.compute_correlation_matrix(),
            "drawdown_details": self.get_drawdown_details(),
            "crisis_regime_returns": self.compute_crisis_regime_returns(),
            "market_regime_returns": self.compute_market_regime_returns(),
        }
        
        if self.benchmark_col:
            results["benchmark_metrics"] = self.compute_all_benchmark_metrics()
        
        return results


# =============================================================================
# TRADE-LEVEL ANALYZER (from module)
# =============================================================================

class TradeAnalyzer:
    """
    Trade-level analysis for backtesting results.
    Analyzes individual trades from trade_results.csv.
    """
    
    def __init__(self, trade_results: pd.DataFrame):
        """
        Initialize TradeAnalyzer.
        
        Parameters:
        - trade_results: DataFrame with columns including:
            ['quarter', 'cat', 'co_name', 'holding_period', 'stock_return', 
             'cat_weight', 'entry_price', 'exit_price']
        """
        self.trade_results = trade_results.copy()
        self.has_tpsl = {'SL_triggered', 'TP_triggered'}.issubset(trade_results.columns)
    
    def get_category_returns_by_quarter(self) -> pd.DataFrame:
        """Get category returns by quarter."""
        valid = self.trade_results[
            self.trade_results['holding_period'].notna() &
            self.trade_results['stock_return'].notna()
        ]
        
        stats = valid.groupby(['quarter', 'cat']).agg({
            'stock_return': 'mean',
            'cat_weight': 'first',
            'co_name': 'count'
        }).reset_index()
        
        stats.columns = ['quarter', 'category', 'category_return', 'cat_weight', 'stock_count']
        return stats
    
    def get_stock_counts_by_quarter(self) -> pd.DataFrame:
        """Get stock counts by category per quarter."""
        valid = self.trade_results[self.trade_results['holding_period'].notna()]
        counts = valid.groupby(['quarter', 'cat']).size().reset_index(name='stock_count')
        counts.columns = ['quarter', 'category', 'stock_count']
        return counts
    
    def get_holding_period_stats(self) -> pd.DataFrame:
        """Get holding period statistics by quarter and category."""
        valid = self.trade_results[self.trade_results['holding_period'].notna()]
        
        results = []
        for quarter, group in valid.groupby('quarter'):
            row = {
                'quarter': quarter,
                'avg_holding_period': group['holding_period'].mean(),
                'total_stocks': len(group)
            }
            
            for cat in group['cat'].unique():
                cat_data = group[group['cat'] == cat]
                row[f'avg_hp_{cat}'] = cat_data['holding_period'].mean()
                row[f'count_{cat}'] = len(cat_data)
            
            if self.has_tpsl:
                row['TP_count'] = group['TP_triggered'].sum()
                row['SL_count'] = group['SL_triggered'].sum()
                row['time_exit_count'] = len(group) - row['TP_count'] - row['SL_count']
            
            results.append(row)
        
        return pd.DataFrame(results).round(2)
    
    def get_comprehensive_analysis(self) -> pd.DataFrame:
        """Get comprehensive quarter-level analysis."""
        cat_returns = self.get_category_returns_by_quarter()
        hp_stats = self.get_holding_period_stats()
        
        # Pivot category returns
        ret_pivot = cat_returns.pivot(
            index='quarter', columns='category', values='category_return'
        ).add_suffix('_return')
        
        # Merge
        result = hp_stats.set_index('quarter').join(ret_pivot)
        
        # Compute portfolio return
        cat_ret_df = cat_returns.groupby('quarter').apply(
            lambda x: (x['category_return'] * x['cat_weight']).sum()
        )
        result['portfolio_return'] = cat_ret_df
        
        return result.reset_index().round(4)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_analysis(
    returns_df: pd.DataFrame,
    timeline: str = "daily",
    benchmark_col: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict:
    """
    Quick one-liner analysis function.
    
    Parameters:
    - returns_df: DataFrame with date index and return columns
    - timeline: "daily", "monthly", or "yearly"
    - benchmark_col: Optional benchmark column name
    - output_path: Optional path for Excel report
    
    Returns:
    - Dictionary with all computed metrics
    """
    analyzer = PortfolioAnalyzer.from_returns(
        returns_df, timeline, benchmark_col
    )
    
    metrics = analyzer.compute_all_metrics()
    
    if output_path:
        analyzer.generate_excel_report(output_path)
    
    return metrics


def compare_strategies(
    returns_df: pd.DataFrame,
    timeline: str = "daily",
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Quick comparison of multiple strategies.
    
    Returns:
    - DataFrame with performance metrics for all strategies
    """
    analyzer = PortfolioAnalyzer.from_returns(returns_df, timeline)
    metrics = analyzer.compute_performance_metrics()
    
    if output_path:
        analyzer.generate_excel_report(output_path)
    
    return metrics
