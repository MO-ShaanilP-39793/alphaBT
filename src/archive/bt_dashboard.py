import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from addtl_bt_fns import generate_full_analysis_report, compute_portfolio_metrics
except ImportError:
    st.error("Could not import 'addtl_bt_fns'. Make sure this script is in the 'src' folder.")
    st.stop()

st.set_page_config(page_title="Portfolio Analysis Dashboard", layout="wide", page_icon="📈")

# --- CSS Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stMetric {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("📈 Quantitative Portfolio Analysis")
st.markdown("Interactive dashboard for strategy performance evaluation.")

# --- Sidebar: Data Input ---
with st.sidebar:
    st.header("Data Input")
    uploaded_pf = st.file_uploader("Upload Portfolio CSV", type=['csv'], help="Must contain: date, portfolio_value")
    uploaded_bench = st.file_uploader("Upload Benchmark CSV (Optional)", type=['csv'], help="Must contain: date, close/value")
    
    st.markdown("---")
    st.caption("Settings")
    risk_free_rate = st.number_input("Risk Free Rate (%)", value=6.5, step=0.1) / 100
    rolling_window = st.slider("Rolling Window (Days)", 21, 252, 126)

# --- Main Analysis ---
if uploaded_pf:
    # 1. Load Data
    try:
        daily_pf = pd.read_csv(uploaded_pf)
        daily_pf['date'] = pd.to_datetime(daily_pf['date'])
        
        # Basic validation
        if 'portfolio_value' not in daily_pf.columns:
            st.error("CSV must contain 'portfolio_value' column.")
            st.stop()
            
        daily_pf = daily_pf.sort_values('date')
    except Exception as e:
        st.error(f"Error loading portfolio file: {e}")
        st.stop()

    # Load Benchmark if present
    benchmark_df = None
    if uploaded_bench:
        try:
            benchmark_df = pd.read_csv(uploaded_bench)
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            # Standardize column name
            col_map = {c: 'value' for c in benchmark_df.columns if c.lower() in ['close', 'adj close', 'value']}
            if col_map:
                benchmark_df = benchmark_df.rename(columns=col_map)
            benchmark_df = benchmark_df.sort_values('date')
        except Exception as e:
            st.warning(f"Error loading benchmark: {e}")

    # 2. Run Analysis
    with st.spinner("Crunching numbers..."):
        # We generate the standard report dict
        report = generate_full_analysis_report(daily_pf, comparison_df=benchmark_df)
        
        # Extract metrics
        metrics = report['portfolio_metrics']
        
    # 3. KPI Row
    st.subheader("Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    kpi1.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
    kpi2.metric("CAGR", f"{metrics.get('cagr', 0):.2f}%")
    kpi3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    kpi4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%", delta_color="inverse")
    kpi5.metric("Volatility", f"{metrics.get('volatility', 0):.2f}%", delta_color="inverse")

    if benchmark_df is not None:
        st.caption(f"Benchmark Alpha: {report.get('benchmark_metrics', {}).get('alpha', 0):.2f} | Beta: {report.get('benchmark_metrics', {}).get('beta', 0):.2f}")

    # 4. Charts Section
    tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve", "Drawdowns", "Rolling Analysis", "Monthly Returns"])

    with tab1:
        st.subheader("Portfolio Value Over Time")
        
        # Prepare comparison data
        chart_data = daily_pf.set_index('date')[['portfolio_value']].rename(columns={'portfolio_value': 'Portfolio'})
        
        if benchmark_df is not None:
            # Reindex benchmark to match portfolio dates
            bench_series = benchmark_df.set_index('date')['value']
            bench_series = bench_series.reindex(chart_data.index, method='ffill')
            
            # Normalize benchmark to start at portfolio's initial value
            start_val = chart_data['Portfolio'].iloc[0]
            bench_norm = bench_series / bench_series.iloc[0] * start_val
            chart_data['Benchmark'] = bench_norm

        st.line_chart(chart_data, color=["#00CC96", "#EF553B"])

    with tab2:
        st.subheader("Underwater Plot (Drawdowns)")
        dd_data = report['drawdown_series']
        st.area_chart(dd_data.set_index('date')['drawdown_pct'], color="#FF4B4B")
        
        # Worst drawdowns table
        st.markdown("**Worst Drawdown Periods**")
        # Simple logic to find worst days (could be expanded to periods)
        worst_days = dd_data.nsmallest(5, 'drawdown_pct')[['date', 'drawdown_pct']]
        worst_days['drawdown_pct'] = worst_days['drawdown_pct'].apply(lambda x: f"{x:.2f}%")
        worst_days['date'] = worst_days['date'].dt.date
        st.table(worst_days.set_index('date'))

    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{rolling_window}-Day Rolling Volatility**")
            # Recalculate for interactive plotting since report doesn't store the series
            daily_ret = daily_pf.set_index('date')['portfolio_value'].pct_change()
            roll_vol = daily_ret.rolling(rolling_window).std() * np.sqrt(252) * 100
            st.line_chart(roll_vol)
            
        with col2:
            st.markdown(f"**{rolling_window}-Day Rolling Sharpe**")
            daily_rf = risk_free_rate / 252
            excess_ret = daily_ret - daily_rf
            roll_mean = excess_ret.rolling(rolling_window).mean() * 252
            roll_std = daily_ret.rolling(rolling_window).std() * np.sqrt(252)
            roll_sharpe = roll_mean / roll_std
            
            # Color logic is hard in st.line_chart, so we just plot the line
            st.line_chart(roll_sharpe)
            st.caption("Green line = positive risk-adjusted returns")

    with tab4:
        st.subheader("Monthly Returns Heatmap")
        monthly = report['monthly_returns']
        
        # Display as a styled dataframe
        st.dataframe(
            monthly.style
            .background_gradient(cmap='RdYlGn', vmin=-10, vmax=10)
            .format("{:.2f}%")
            .highlight_null(color='lightgray'),
            use_container_width=True,
            height=400
        )

else:
    # Landing page state
    st.info("👋 Welcome! Please upload your 'daily_portfolio_value.csv' file to begin analysis.")
    
    st.markdown("""
    ### Expected Data Format
    The CSV file should be the output from your backtest, typically containing:
    - `date`: YYYY-MM-DD
    - `portfolio_value`: Numerical value of portfolio
    - `quarter`: (Optional) Quarter identifier
    """)
