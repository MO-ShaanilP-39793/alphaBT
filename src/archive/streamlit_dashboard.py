"""
Comprehensive Backtesting Dashboard - Streamlit Application

A feature-rich dashboard for analyzing backtesting results with:
- Run folder selection from backtesting_results/
- Optional: Run new backtests directly from the UI
- Complete analysis using all functions from addtl_bt_fns.py
- Interactive charts and metrics

Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import analysis functions
try:
    from addtl_bt_fns import (
        # Portfolio-level analysis
        compute_portfolio_metrics,
        compute_drawdown_series,
        compute_rolling_returns,
        compute_calendar_rolling_returns,
        compute_monthly_returns,
        # Benchmark comparison
        compute_benchmark_metrics,
        compute_rolling_alpha,
        compute_rolling_alpha_calendar,
        # Category-level analysis
        get_stock_counts_by_category,
        get_stock_counts_pivot,
        get_category_returns_by_quarter,
        get_category_returns_pivot,
        get_holding_period_by_quarter,
        get_comprehensive_quarter_analysis,
    )
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    ANALYSIS_AVAILABLE = False
    IMPORT_ERROR = str(e)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="QPF Backtesting Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING - Professional Theme
# =============================================================================

st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    /* Global font settings */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    }
    
    /* Main theme colors - Professional dark blue palette */
    :root {
        --primary-color: #1a365d;
        --secondary-color: #2c5282;
        --accent-color: #3182ce;
        --positive-color: #276749;
        --negative-color: #c53030;
        --neutral-color: #4a5568;
        --text-primary: #1a202c;
        --text-secondary: #718096;
        --bg-light: #f7fafc;
        --bg-card: #ffffff;
        --border-color: #e2e8f0;
    }
    
    /* Header styling */
    .main-header {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 0.25rem;
        letter-spacing: -0.025em;
    }
    
    .sub-header {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1rem;
        font-weight: 400;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
    }
    
    /* Section headers */
    h3, .stMarkdown h3 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        color: var(--primary-color);
        font-size: 1.125rem;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    
    h4, .stMarkdown h4 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        color: var(--secondary-color);
        font-size: 0.95rem;
        margin-top: 1rem;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: var(--bg-light);
        padding: 0.25rem;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        color: var(--text-secondary);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-card);
        color: var(--primary-color);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-light);
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-secondary);
        margin-top: 1.5rem;
    }
    
    /* Code blocks */
    code {
        font-family: 'IBM Plex Mono', 'Consolas', monospace;
        font-size: 0.85rem;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85rem;
    }
    
    /* Button styling */
    .stButton > button {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        border-radius: 4px;
    }
    
    /* Selectbox and inputs */
    .stSelectbox label, .stMultiSelect label, .stNumberInput label, .stSlider label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        color: var(--text-primary);
    }
    
    /* Caption styling */
    .stCaption {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--border-color);
        margin: 1.5rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Download button */
    .stDownloadButton > button {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_backtest_runs(base_dir: str = "../backtesting_results") -> list:
    """Get list of available backtest run folders."""
    if not os.path.exists(base_dir):
        return []
    
    runs = []
    for folder in sorted(os.listdir(base_dir), reverse=True):
        run_path = os.path.join(base_dir, folder)
        if os.path.isdir(run_path) and folder.startswith("run_"):
            # Check if it has required files
            required_files = ["daily_portfolio_values.csv", "trade_results.csv"]
            has_files = all(os.path.exists(os.path.join(run_path, f)) for f in required_files)
            if has_files:
                # Extract timestamp from folder name
                try:
                    timestamp_str = folder.replace("run_", "")
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    display_name = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    display_name = folder
                runs.append({"path": run_path, "name": folder, "display": display_name})
    
    return runs


def load_run_data(run_path: str) -> dict:
    """Load all data files from a backtest run folder."""
    data = {}
    
    # Load daily portfolio values
    pf_path = os.path.join(run_path, "daily_portfolio_values.csv")
    if os.path.exists(pf_path):
        data["daily_pf"] = pd.read_csv(pf_path)
        data["daily_pf"]["date"] = pd.to_datetime(data["daily_pf"]["date"])
    
    # Load trade results
    trade_path = os.path.join(run_path, "trade_results.csv")
    if os.path.exists(trade_path):
        data["trade_results"] = pd.read_csv(trade_path)
    
    # Load benchmark comparison
    bench_path = os.path.join(run_path, "portfolio_vs_index.csv")
    if os.path.exists(bench_path):
        data["comparison_df"] = pd.read_csv(bench_path)
        data["comparison_df"]["date"] = pd.to_datetime(data["comparison_df"]["date"])
    
    # Load config
    config_path = os.path.join(run_path, "config_used.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data["config"] = yaml.safe_load(f)
    
    return data


def format_currency(value: float) -> str:
    """Format large numbers as currency with Indian notation (Cr, L)."""
    if abs(value) >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif abs(value) >= 1e5:
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a percentage value."""
    return f"{value:.{decimals}f}%"


def color_metric(value: float, threshold: float = 0, inverse: bool = False) -> str:
    """Return CSS class based on value."""
    if inverse:
        return "metric-positive" if value < threshold else "metric-negative"
    return "metric-positive" if value > threshold else "metric-negative"


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with run selection and settings."""
    
    st.sidebar.markdown("## Select Backtest Run")
    
    # Get available runs
    runs = get_backtest_runs()
    
    if not runs:
        st.sidebar.warning("No backtest runs found in `backtesting_results/`")
        st.sidebar.info("Run a backtest first using `backtest_strategy.py`")
        return None
    
    # Run selector
    run_options = {r["display"]: r["path"] for r in runs}
    selected_display = st.sidebar.selectbox(
        "Choose a run:",
        options=list(run_options.keys()),
        help="Runs are sorted newest first"
    )
    selected_path = run_options[selected_display]
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.markdown("## Settings")
    
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=6.5,
        step=0.1,
        help="Annual risk-free rate for Sharpe/Sortino calculations"
    ) / 100
    
    rolling_window = st.sidebar.slider(
        "Rolling Window (days)",
        min_value=21,
        max_value=252,
        value=63,
        step=21,
        help="Window for rolling metrics (21=1M, 63=3M, 126=6M, 252=1Y)"
    )
    
    use_calendar_rolling = st.sidebar.checkbox(
        "Use calendar-based rolling periods",
        value=False,
        help="Use calendar months instead of trading days for rolling calculations"
    )
    
    st.sidebar.markdown("---")
    
    # Run new backtest section
    st.sidebar.markdown("## Run New Backtest")
    
    if st.sidebar.button("Run Backtest", type="primary", use_container_width=True):
        with st.sidebar.status("Running backtest...", expanded=True) as status:
            try:
                # Import and run backtest
                from backtest_strategy import run_backtest
                output_dir, _, _ = run_backtest()
                status.update(label="Backtest complete!", state="complete")
                st.sidebar.success(f"Results saved to: {output_dir}")
                st.rerun()
            except Exception as e:
                status.update(label="Backtest failed", state="error")
                st.sidebar.error(f"Error: {str(e)}")
    
    st.sidebar.caption("Uses strategy_config.yaml")
    
    return {
        "run_path": selected_path,
        "risk_free_rate": risk_free_rate,
        "rolling_window": rolling_window,
        "use_calendar_rolling": use_calendar_rolling
    }


# =============================================================================
# TAB RENDERERS
# =============================================================================

def render_executive_summary(data: dict, settings: dict):
    """Render the Executive Summary tab."""
    
    daily_pf = data.get("daily_pf")
    config = data.get("config", {})
    comparison_df = data.get("comparison_df")
    
    if daily_pf is None:
        st.error("Daily portfolio values not found.")
        return
    
    # Compute metrics
    metrics = compute_portfolio_metrics(daily_pf, risk_free_rate=settings["risk_free_rate"])
    
    # Benchmark metrics if available
    bench_metrics = None
    if comparison_df is not None:
        bench_metrics = compute_benchmark_metrics(comparison_df)
    
    # Header row with config summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Performance Overview")
        
        # Period info
        st.markdown(f"""
        **Period:** {metrics['start_date']} to {metrics['end_date']} ({metrics['years']} years, {metrics['trading_days']} trading days)
        """)
    
    with col2:
        st.markdown("### Strategy Configuration")
        scheme = config.get("category_scheme", "N/A")
        counts = config.get("category_counts", [])
        quarters = f"{config.get('first_quarter', 'N/A')} - {config.get('last_quarter', 'N/A')}"
        st.code(f"Scheme: {scheme}\nCounts: {counts}\nQuarters: {quarters}", language=None)
    
    st.markdown("---")
    
    # KPI Row 1 - Returns
    st.markdown("#### Returns")
    kpi_cols = st.columns(5)
    
    with kpi_cols[0]:
        val = metrics['total_return_pct']
        st.metric("Total Return", format_pct(val), delta=None)
    
    with kpi_cols[1]:
        val = metrics['cagr_pct']
        st.metric("CAGR", format_pct(val))
    
    with kpi_cols[2]:
        st.metric("Initial Value", format_currency(metrics['initial_value']))
    
    with kpi_cols[3]:
        st.metric("Final Value", format_currency(metrics['final_value']))
    
    with kpi_cols[4]:
        if bench_metrics:
            st.metric("Alpha", format_pct(bench_metrics['alpha_pct']), 
                     delta=f"vs Index: {format_pct(bench_metrics['index_return_pct'])}")
        else:
            st.metric("Best Day", format_pct(metrics['best_day_pct']))
    
    # KPI Row 2 - Risk
    st.markdown("#### Risk Metrics")
    risk_cols = st.columns(5)
    
    with risk_cols[0]:
        st.metric("Volatility", format_pct(metrics['volatility_pct']))
    
    with risk_cols[1]:
        val = metrics['max_drawdown_pct']
        st.metric("Max Drawdown", format_pct(val), delta_color="inverse")
    
    with risk_cols[2]:
        st.metric("VaR (95%)", format_pct(metrics['var_95_pct']), delta_color="inverse")
    
    with risk_cols[3]:
        st.metric("Worst Day", format_pct(metrics['worst_day_pct']), delta_color="inverse")
    
    with risk_cols[4]:
        st.metric("Win Days %", format_pct(metrics['positive_days_pct']))
    
    # KPI Row 3 - Risk-Adjusted
    st.markdown("#### Risk-Adjusted Performance")
    adj_cols = st.columns(5)
    
    with adj_cols[0]:
        sharpe = metrics['sharpe_ratio']
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
    
    with adj_cols[1]:
        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
    
    with adj_cols[2]:
        st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.3f}")
    
    with adj_cols[3]:
        if bench_metrics:
            st.metric("Beta", f"{bench_metrics['beta']:.3f}")
        else:
            st.metric("—", "—")
    
    with adj_cols[4]:
        if bench_metrics:
            st.metric("Info Ratio", f"{bench_metrics['information_ratio']:.3f}")
        else:
            st.metric("—", "—")
    
    st.markdown("---")
    
    # Equity Curve
    st.markdown("### Equity Curve")
    
    chart_data = daily_pf.set_index("date")[["portfolio_value"]].rename(
        columns={"portfolio_value": "Portfolio"}
    )
    
    if comparison_df is not None:
        chart_data["Benchmark"] = comparison_df.set_index("date")["index_fund_value"].reindex(chart_data.index)
    
    st.line_chart(chart_data, use_container_width=True)


def render_risk_analysis(data: dict, settings: dict):
    """Render the Risk Analysis tab."""
    
    daily_pf = data.get("daily_pf")
    if daily_pf is None:
        st.error("Daily portfolio values not found.")
        return
    
    st.markdown("### Drawdown Analysis")
    
    # Compute drawdown
    dd_series = compute_drawdown_series(daily_pf)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Drawdown chart
        st.markdown("#### Underwater Chart")
        dd_chart = dd_series.set_index("date")[["drawdown_pct"]]
        st.area_chart(dd_chart, use_container_width=True, color="#c53030")
    
    with col2:
        # Worst drawdowns table
        st.markdown("#### Worst Drawdown Days")
        worst = dd_series.nsmallest(10, "drawdown_pct")[["date", "portfolio_value", "drawdown_pct"]].copy()
        worst["date"] = worst["date"].dt.strftime("%Y-%m-%d")
        worst["drawdown_pct"] = worst["drawdown_pct"].apply(lambda x: f"{x:.2f}%")
        worst["portfolio_value"] = worst["portfolio_value"].apply(format_currency)
        worst.columns = ["Date", "Value", "Drawdown"]
        st.dataframe(worst, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Rolling Volatility
    st.markdown("### Rolling Volatility Analysis")
    
    window = settings["rolling_window"]
    
    df = daily_pf.copy()
    df["daily_return"] = df["portfolio_value"].pct_change()
    df["rolling_vol"] = df["daily_return"].rolling(window).std() * np.sqrt(252) * 100
    
    vol_chart = df.set_index("date")[["rolling_vol"]].dropna()
    vol_chart.columns = [f"{window}-Day Rolling Volatility (%)"]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.line_chart(vol_chart, use_container_width=True, color="#6b46c1")
    
    with col2:
        avg_vol = vol_chart.iloc[:, 0].mean()
        max_vol = vol_chart.iloc[:, 0].max()
        min_vol = vol_chart.iloc[:, 0].min()
        current_vol = vol_chart.iloc[-1, 0]
        
        st.metric("Average", format_pct(avg_vol))
        st.metric("Current", format_pct(current_vol))
        st.metric("Max", format_pct(max_vol))
        st.metric("Min", format_pct(min_vol))
    
    st.markdown("---")
    
    # Rolling Sharpe
    st.markdown("### Rolling Sharpe Ratio")
    
    risk_free_rate = settings["risk_free_rate"]
    daily_rf = risk_free_rate / 252
    
    df["excess_return"] = df["daily_return"] - daily_rf
    roll_mean = df["excess_return"].rolling(window).mean() * 252
    roll_std = df["daily_return"].rolling(window).std() * np.sqrt(252)
    df["rolling_sharpe"] = roll_mean / roll_std
    
    sharpe_chart = df.set_index("date")[["rolling_sharpe"]].dropna()
    sharpe_chart.columns = [f"{window}-Day Rolling Sharpe"]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.line_chart(sharpe_chart, use_container_width=True, color="#2b6cb0")
    
    with col2:
        avg_sharpe = sharpe_chart.iloc[:, 0].mean()
        current_sharpe = sharpe_chart.iloc[-1, 0]
        pct_positive = (sharpe_chart.iloc[:, 0] > 0).mean() * 100
        
        st.metric("Average", f"{avg_sharpe:.3f}")
        st.metric("Current", f"{current_sharpe:.3f}")
        st.metric("% Positive", format_pct(pct_positive))


def render_returns_analysis(data: dict, settings: dict):
    """Render the Returns Analysis tab."""
    
    daily_pf = data.get("daily_pf")
    if daily_pf is None:
        st.error("Daily portfolio values not found.")
        return
    
    # Monthly Returns Heatmap
    st.markdown("### Monthly Returns Heatmap")
    
    monthly = compute_monthly_returns(daily_pf)
    
    # Style the dataframe
    def color_returns(val):
        if pd.isna(val):
            return "background-color: #f0f0f0"
        elif val > 5:
            return "background-color: #276749; color: white"
        elif val > 0:
            return "background-color: #c6f6d5"
        elif val > -5:
            return "background-color: #fed7d7"
        else:
            return "background-color: #c53030; color: white"
    
    styled_monthly = monthly.style.applymap(color_returns).format("{:.2f}%", na_rep="—")
    st.dataframe(styled_monthly, use_container_width=True)
    
    st.markdown("---")
    
    # Rolling Returns
    st.markdown("### Rolling Returns")
    
    if settings["use_calendar_rolling"]:
        rolling_returns = compute_calendar_rolling_returns(daily_pf)
        return_cols = [c for c in rolling_returns.columns if c.startswith("return_") and "_cal" in c]
    else:
        rolling_returns = compute_rolling_returns(daily_pf)
        return_cols = [c for c in rolling_returns.columns if c.startswith("return_")]
    
    if return_cols:
        chart_data = rolling_returns.set_index("date")[return_cols].dropna(how="all")
        chart_data.columns = [c.replace("return_", "").replace("_cal", " (Cal)") for c in chart_data.columns]
        st.line_chart(chart_data, use_container_width=True)
    
    st.markdown("---")
    
    # Return Distribution
    st.markdown("### Daily Return Distribution")
    
    df = daily_pf.copy()
    df["daily_return"] = df["portfolio_value"].pct_change() * 100
    returns = df["daily_return"].dropna()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Histogram using Streamlit's native chart (approximation)
        hist_data = pd.DataFrame({"Daily Return (%)": returns})
        st.bar_chart(returns.value_counts(bins=50).sort_index(), use_container_width=True)
    
    with col2:
        st.markdown("#### Distribution Statistics")
        st.metric("Mean", f"{returns.mean():.3f}%")
        st.metric("Std Dev", f"{returns.std():.3f}%")
        st.metric("Skewness", f"{returns.skew():.3f}")
        st.metric("Kurtosis", f"{returns.kurtosis():.3f}")
        st.metric("VaR 95%", f"{returns.quantile(0.05):.3f}%")
        st.metric("VaR 99%", f"{returns.quantile(0.01):.3f}%")


def render_benchmark_comparison(data: dict, settings: dict):
    """Render the Benchmark Comparison tab."""
    
    comparison_df = data.get("comparison_df")
    daily_pf = data.get("daily_pf")
    
    if comparison_df is None:
        st.warning("No benchmark comparison data available for this run.")
        st.info("Make sure `portfolio_vs_index.csv` exists in the run folder.")
        return
    
    # Compute benchmark metrics
    bench_metrics = compute_benchmark_metrics(comparison_df)
    
    st.markdown("### Portfolio vs Benchmark")
    
    # Metrics row
    cols = st.columns(5)
    
    with cols[0]:
        st.metric("Portfolio Return", format_pct(bench_metrics['portfolio_return_pct']))
    
    with cols[1]:
        st.metric("Index Return", format_pct(bench_metrics['index_return_pct']))
    
    with cols[2]:
        alpha = bench_metrics['alpha_pct']
        st.metric("Alpha", format_pct(alpha), 
                 delta="Outperformed" if alpha > 0 else "Underperformed")
    
    with cols[3]:
        st.metric("Beta", f"{bench_metrics['beta']:.3f}")
    
    with cols[4]:
        st.metric("Correlation", f"{bench_metrics['correlation']:.3f}")
    
    st.markdown("---")
    
    # Second metrics row
    cols2 = st.columns(5)
    
    with cols2[0]:
        st.metric("Tracking Error", format_pct(bench_metrics['tracking_error_pct']))
    
    with cols2[1]:
        st.metric("Information Ratio", f"{bench_metrics['information_ratio']:.3f}")
    
    with cols2[2]:
        st.metric("Up Capture", format_pct(bench_metrics['up_capture_pct']))
    
    with cols2[3]:
        st.metric("Down Capture", format_pct(bench_metrics['down_capture_pct']))
    
    with cols2[4]:
        st.metric("Outperform Days %", format_pct(bench_metrics['outperformance_days_pct']))
    
    st.markdown("---")
    
    # Comparison chart
    st.markdown("### Cumulative Returns Comparison")
    
    chart_data = comparison_df.set_index("date")[["pf_return", "index_return"]].copy()
    chart_data.columns = ["Portfolio", "Index"]
    st.line_chart(chart_data, use_container_width=True)
    
    st.markdown("---")
    
    # Rolling Alpha
    st.markdown("### Rolling Alpha")
    
    window = settings["rolling_window"]
    
    if settings["use_calendar_rolling"]:
        # Convert window (days) to approximate months
        months = max(1, window // 21)
        rolling_alpha = compute_rolling_alpha_calendar(comparison_df, period=months)
        title = f"{months}-Month Rolling Alpha (Calendar)"
    else:
        rolling_alpha = compute_rolling_alpha(comparison_df, window=window)
        title = f"{window}-Day Rolling Alpha"
    
    if not rolling_alpha.empty:
        alpha_chart = rolling_alpha.set_index("date")[["rolling_alpha"]]
        alpha_chart.columns = [title]
        st.line_chart(alpha_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Alpha timeline
    st.markdown("### Cumulative Alpha Over Time")
    alpha_timeline = comparison_df.set_index("date")[["alpha"]]
    alpha_timeline.columns = ["Cumulative Alpha (%)"]
    st.area_chart(alpha_timeline, use_container_width=True, color="#276749")


def render_category_analysis(data: dict, settings: dict):
    """Render the Category Analysis tab."""
    
    trade_results = data.get("trade_results")
    
    if trade_results is None:
        st.error("Trade results not found.")
        return
    
    st.markdown("### Category-Level Performance")
    
    # Get category returns
    cat_returns = get_category_returns_by_quarter(trade_results)
    cat_pivot = cat_returns.pivot(index='quarter', columns='category', values='category_return')
    
    # Stock counts
    counts_pivot = get_stock_counts_pivot(trade_results)
    
    # Comprehensive quarter analysis
    quarter_analysis = get_comprehensive_quarter_analysis(trade_results)
    
    # Category returns chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Returns by Category per Quarter")
        
        # Prepare data for bar chart
        chart_df = cat_pivot.copy() * 100  # Convert to percentage
        st.bar_chart(chart_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Average Category Returns")
        avg_returns = cat_pivot.mean() * 100
        for cat, ret in avg_returns.items():
            indicator = "+" if ret > 0 else ""
            st.markdown(f"**{cat.title()}**: {indicator}{ret:.2f}%")
    
    st.markdown("---")
    
    # Cumulative returns
    st.markdown("#### Cumulative Returns by Category")
    cumulative = (1 + cat_pivot).cumprod() - 1
    cumulative_chart = cumulative * 100
    st.line_chart(cumulative_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Stock counts distribution
    st.markdown("#### Stock Distribution by Category")
    
    counts_for_chart = counts_pivot.set_index("quarter")
    st.bar_chart(counts_for_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Holding periods
    st.markdown("#### Average Holding Period by Quarter")
    holding = get_holding_period_by_quarter(trade_results)
    
    hp_cols = [c for c in holding.columns if c.startswith("avg_hp_")]
    if hp_cols:
        hp_chart = holding.set_index("quarter")[hp_cols]
        hp_chart.columns = [c.replace("avg_hp_", "").title() for c in hp_chart.columns]
        st.bar_chart(hp_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Comprehensive table
    st.markdown("#### Detailed Quarter Analysis")
    st.dataframe(quarter_analysis, use_container_width=True, hide_index=True)


def render_trade_details(data: dict, settings: dict):
    """Render the Trade Details tab."""
    
    trade_results = data.get("trade_results")
    
    if trade_results is None:
        st.error("Trade results not found.")
        return
    
    st.markdown("### Trade-Level Details")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    quarters = sorted(trade_results["quarter"].unique())
    categories = sorted(trade_results["cat"].unique())
    
    with col1:
        selected_quarters = st.multiselect(
            "Filter by Quarter",
            options=quarters,
            default=quarters,
            key="trade_quarters"
        )
    
    with col2:
        selected_categories = st.multiselect(
            "Filter by Category",
            options=categories,
            default=categories,
            key="trade_categories"
        )
    
    with col3:
        exit_type = st.selectbox(
            "Exit Type",
            options=["All", "TP Hit", "SL Hit", "Time Exit"],
            key="exit_type"
        )
    
    with col4:
        return_filter = st.selectbox(
            "Return Filter",
            options=["All", "Positive Only", "Negative Only"],
            key="return_filter"
        )
    
    # Apply filters
    filtered = trade_results[
        (trade_results["quarter"].isin(selected_quarters)) &
        (trade_results["cat"].isin(selected_categories))
    ].copy()
    
    if exit_type == "TP Hit" and "TP_triggered" in filtered.columns:
        filtered = filtered[filtered["TP_triggered"] == True]
    elif exit_type == "SL Hit" and "SL_triggered" in filtered.columns:
        filtered = filtered[filtered["SL_triggered"] == True]
    elif exit_type == "Time Exit" and "TP_triggered" in filtered.columns and "SL_triggered" in filtered.columns:
        filtered = filtered[(filtered["TP_triggered"] == False) & (filtered["SL_triggered"] == False)]
    
    if return_filter == "Positive Only" and "stock_return" in filtered.columns:
        filtered = filtered[filtered["stock_return"] > 0]
    elif return_filter == "Negative Only" and "stock_return" in filtered.columns:
        filtered = filtered[filtered["stock_return"] < 0]
    
    # Summary stats
    st.markdown("---")
    
    stats_cols = st.columns(5)
    
    with stats_cols[0]:
        st.metric("Total Trades", len(filtered))
    
    with stats_cols[1]:
        if "stock_return" in filtered.columns:
            avg_return = filtered["stock_return"].mean() * 100
            st.metric("Avg Return", format_pct(avg_return))
    
    with stats_cols[2]:
        if "holding_period" in filtered.columns:
            avg_hp = filtered["holding_period"].mean()
            st.metric("Avg Holding (days)", f"{avg_hp:.1f}")
    
    with stats_cols[3]:
        if "TP_triggered" in filtered.columns:
            tp_count = filtered["TP_triggered"].sum()
            st.metric("TP Hits", int(tp_count))
    
    with stats_cols[4]:
        if "SL_triggered" in filtered.columns:
            sl_count = filtered["SL_triggered"].sum()
            st.metric("SL Hits", int(sl_count))
    
    st.markdown("---")
    
    # Display columns
    display_cols = ["quarter", "co_name", "cat", "entry_date", "exit_date", 
                    "entry_price", "exit_price", "holding_period", "stock_return"]
    if "TP_triggered" in filtered.columns:
        display_cols.extend(["TP_triggered", "SL_triggered"])
    
    display_cols = [c for c in display_cols if c in filtered.columns]
    
    # Format for display
    display_df = filtered[display_cols].copy()
    
    if "stock_return" in display_df.columns:
        display_df["stock_return"] = display_df["stock_return"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
    
    if "entry_price" in display_df.columns:
        display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "—")
    
    if "exit_price" in display_df.columns:
        display_df["exit_price"] = display_df["exit_price"].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "—")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)
    
    # Download button
    csv = filtered.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_trades.csv",
        mime="text/csv"
    )


def render_config_viewer(data: dict):
    """Render the Configuration tab."""
    
    config = data.get("config", {})
    
    if not config:
        st.warning("No configuration data found for this run.")
        return
    
    st.markdown("### Strategy Configuration Used")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Data Paths")
        st.code(f"""
Input Data: {config.get('input_data_path', 'N/A')}
Price Data: {config.get('price_data_path', 'N/A')}
Index Data: {config.get('index_data_path', 'N/A')}
        """, language=None)
        
        st.markdown("#### Period")
        st.code(f"""
First Quarter: {config.get('first_quarter', 'N/A')}
Last Quarter: {config.get('last_quarter', 'N/A')}
        """, language=None)
        
        st.markdown("#### Stock Selection")
        st.code(f"""
Category Scheme: {config.get('category_scheme', 'N/A')}
Category Counts: {config.get('category_counts', 'N/A')}
Category Weights: {config.get('category_weights', 'N/A')}
Selection Method: {config.get('selection_method', 'probability')}
Min Prob Threshold: {config.get('min_prob_threshold', 'None')}
        """, language=None)
    
    with col2:
        st.markdown("#### TP/SL Configuration")
        tpsl_mode = config.get('tpsl_mode', 'fixed')
        st.code(f"Mode: {tpsl_mode}", language=None)
        
        if tpsl_mode == 'fixed':
            tp_config = config.get('TP_CONFIG', {})
            sl_config = config.get('SL_CONFIG', {})
            
            scheme = config.get('category_scheme', 'volatility')
            
            st.markdown("**Take Profit Thresholds:**")
            if scheme in tp_config:
                for cat, val in tp_config[scheme].items():
                    st.code(f"  {cat}: {val*100:.1f}%", language=None)
            
            st.markdown("**Stop Loss Thresholds:**")
            if scheme in sl_config:
                for cat, val in sl_config[scheme].items():
                    st.code(f"  {cat}: {val*100:.1f}%", language=None)
        
        elif tpsl_mode == 'atr':
            atr_config = config.get('atr_config', {})
            st.code(f"""
ATR Period: {atr_config.get('period', 'N/A')}
TP Multiplier: {atr_config.get('tp_multiplier', 'N/A')}
SL Multiplier: {atr_config.get('sl_multiplier', 'N/A')}
            """, language=None)
    
    st.markdown("---")
    
    # Full YAML view
    with st.expander("View Full Configuration (YAML)"):
        st.code(yaml.dump(config, default_flow_style=False), language="yaml")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Check if analysis functions are available
    if not ANALYSIS_AVAILABLE:
        st.error(f"Failed to import analysis functions: {IMPORT_ERROR}")
        st.info("Make sure `addtl_bt_fns.py` is in the same directory.")
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">QPF Backtesting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive analysis of quantitative portfolio strategies</p>', unsafe_allow_html=True)
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    if settings is None:
        st.info("No backtest runs found. Run a backtest first using `backtest_strategy.py`")
        st.stop()
    
    # Load data for selected run
    with st.spinner("Loading backtest data..."):
        data = load_run_data(settings["run_path"])
    
    if not data:
        st.error("Failed to load data from selected run.")
        st.stop()
    
    # Display run info
    run_name = os.path.basename(settings["run_path"])
    st.caption(f"Viewing: {run_name}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Executive Summary",
        "Risk Analysis",
        "Returns Analysis",
        "Benchmark",
        "Category Analysis",
        "Trade Details",
        "Configuration"
    ])
    
    with tab1:
        render_executive_summary(data, settings)
    
    with tab2:
        render_risk_analysis(data, settings)
    
    with tab3:
        render_returns_analysis(data, settings)
    
    with tab4:
        render_benchmark_comparison(data, settings)
    
    with tab5:
        render_category_analysis(data, settings)
    
    with tab6:
        render_trade_details(data, settings)
    
    with tab7:
        render_config_viewer(data)


if __name__ == "__main__":
    main()
