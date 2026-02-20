"""Chart generation functions — all return BytesIO buffers for embedding in Excel."""

import io

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter, FuncFormatter

from config.defaults import INITIAL_CAPITAL, DAYS_PER_YEAR
from utils.formatting import crores_formatter

from .metrics import compute_drawdown_series


def create_pf_vs_index_chart(comparison_df, first_quarter, last_quarter, initial_capital=INITIAL_CAPITAL):
    """
    Create Portfolio vs Index chart (2-panel: value comparison + alpha).
    Extracted from simulation.py plot_pf_vs_index.

    Returns:
    - BytesIO buffer containing the PNG image, or None if no data
    """
    if comparison_df is None or comparison_df.empty:
        return None

    df = comparison_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True)

    # Top: Portfolio Value vs Index
    ax1.plot(df['date'], df['pf_value'], color='#1f77b4', linewidth=2, label='Portfolio')
    ax1.plot(df['date'], df['index_fund_value'], color='#ff7f0e', linewidth=2, label='Index Fund')
    ax1.axhline(y=initial_capital, color='black', linestyle='--', alpha=0.5, label='Initial Capital')

    ax1.fill_between(df['date'], df['pf_value'], df['index_fund_value'],
                     where=(df['pf_value'] >= df['index_fund_value']),
                     interpolate=True, color='green', alpha=0.1, label='Outperformance')
    ax1.fill_between(df['date'], df['pf_value'], df['index_fund_value'],
                     where=(df['pf_value'] < df['index_fund_value']),
                     interpolate=True, color='red', alpha=0.1, label='Underperformance')

    ax1.set_title(f'Portfolio vs Index: {first_quarter} to {last_quarter} (Initial Capital: \u20B9{initial_capital/10000000:.0f} Cr)',
                  fontsize=14, pad=15)
    ax1.set_ylabel('Value (INR)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')

    ax1.yaxis.set_major_formatter(FuncFormatter(crores_formatter))

    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    years = (end_date - start_date).days / DAYS_PER_YEAR

    final_pf_value = df['pf_value'].iloc[-1]
    final_index_value = df['index_fund_value'].iloc[-1]

    if years > 0:
        pf_cagr = ((final_pf_value / initial_capital) ** (1 / years) - 1) * 100
        index_cagr = ((final_index_value / initial_capital) ** (1 / years) - 1) * 100
    else:
        pf_cagr = 0
        index_cagr = 0

    annotation_text = (f"Portfolio CAGR: {pf_cagr:+.2f}%\n"
                       f"Index CAGR: {index_cagr:+.2f}%")
    ax1.text(0.98, 0.95, annotation_text, transform=ax1.transAxes,
             fontsize=11, fontweight='bold', verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

    # Bottom: Alpha
    ax2.fill_between(df['date'], df['alpha'], 0,
                     where=(df['alpha'] >= 0), interpolate=True, color='green', alpha=0.3)
    ax2.fill_between(df['date'], df['alpha'], 0,
                     where=(df['alpha'] < 0), interpolate=True, color='red', alpha=0.3)
    ax2.plot(df['date'], df['alpha'], color='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Alpha (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    date_range_days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    if date_range_days > 365:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    elif date_range_days > 180:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    else:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return buffer


def create_growth_of_wealth_chart(monthly_returns_df, chart_title="Growth of Rs 10,000"):
    """Create Growth of Wealth chart. Returns BytesIO buffer."""
    gofwealth = monthly_returns_df.add(1).cumprod().multiply(10000).reset_index()
    date_col = gofwealth.columns[0]

    plt.figure(figsize=(12, 6))
    colors = plt.colormaps["tab10"].colors

    for idx, col in enumerate(gofwealth.columns[1:]):
        plt.plot(gofwealth[date_col], gofwealth[col], label=col, linewidth=1, color=colors[idx % len(colors)])

    plt.ylabel("Growth of Rs 10,000", fontsize=10)
    plt.title(chart_title, fontsize=12)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('Rs. {x:,.0f}'))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    plt.close()
    buffer.seek(0)
    return buffer


def create_daily_drawdown_chart(daily_pf):
    """
    Create daily drawdown chart (2-panel: value + drawdown).
    Refactored from analytics.py plot_drawdown to return BytesIO buffer.
    """
    dd = compute_drawdown_series(daily_pf)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1], sharex=True)

    # Portfolio value
    ax1.plot(dd['date'], dd['portfolio_value'], color='#1f77b4', linewidth=1.5, label='Portfolio')
    ax1.plot(dd['date'], dd['cummax'], color='gray', linestyle='--', alpha=0.7, label='Peak')
    ax1.fill_between(dd['date'], dd['portfolio_value'], dd['cummax'], alpha=0.3, color='red')
    ax1.set_ylabel('Portfolio Value (Cr)')

    def crores_formatter_local(x, pos):
        return f'{x/1e7:.0f}'
    ax1.yaxis.set_major_formatter(FuncFormatter(crores_formatter_local))
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Portfolio Value and Drawdown Analysis', fontsize=14)

    # Drawdown
    ax2.fill_between(dd['date'], dd['drawdown_pct'], 0, color='red', alpha=0.5)
    ax2.plot(dd['date'], dd['drawdown_pct'], color='darkred', linewidth=1)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    max_dd_idx = dd['drawdown_pct'].idxmin()
    max_dd_date = dd.loc[max_dd_idx, 'date']
    max_dd_val = dd.loc[max_dd_idx, 'drawdown_pct']
    ax2.annotate(f'Max DD: {max_dd_val:.1f}%', xy=(max_dd_date, max_dd_val),
                 xytext=(10, 15), textcoords='offset points',
                 fontsize=10, color='darkred', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=1))

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return buffer


def create_monthly_returns_heatmap(daily_pf):
    """
    Create monthly returns heatmap.
    Refactored from analytics.py plot_monthly_returns_heatmap to return BytesIO buffer.
    """
    # Build monthly returns pivot
    df = daily_pf.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    monthly = df['portfolio_value'].resample('ME').last()
    monthly_returns = monthly.pct_change() * 100

    result = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    pivot = result.pivot(index='year', columns='month', values='return')

    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    pivot.columns = [month_names[m] for m in pivot.columns]

    monthly_only = pivot

    fig, ax = plt.subplots(figsize=(14, max(4, len(monthly_only) * 0.5 + 1)))

    im = ax.imshow(monthly_only.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

    ax.set_xticks(range(len(monthly_only.columns)))
    ax.set_xticklabels(monthly_only.columns)
    ax.set_yticks(range(len(monthly_only.index)))
    ax.set_yticklabels(monthly_only.index)

    for i in range(len(monthly_only.index)):
        for j in range(len(monthly_only.columns)):
            val = monthly_only.iloc[i, j]
            if pd.notna(val):
                text_color = 'white' if abs(val) > 5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=text_color, fontsize=9)

    ax.set_title('Monthly Returns (%)', fontsize=14)
    plt.colorbar(im, ax=ax, label='Return (%)')
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return buffer


def create_calendar_year_heatmap(calendar_year_df, chart_title="Calendar Year Returns Heatmap"):
    """Create Calendar Year Returns Heatmap. Returns BytesIO buffer."""
    if calendar_year_df.empty:
        return None

    n_rows, n_cols = calendar_year_df.shape
    cell_width = 0.8
    cell_height = 0.3
    fig_width = max(6, n_cols * cell_width)
    fig_height = max(4, n_rows * cell_height)

    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        calendar_year_df, annot=True, fmt=".1f", cmap="RdYlGn",
        center=0, linewidths=0.5, cbar_kws={"label": "Return (%)"}
    )

    plt.title(chart_title, fontsize=12)
    ax.set_xlabel("Strategies")
    ax.set_ylabel("Calendar Year", fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    plt.close()
    buffer.seek(0)
    return buffer


def create_correlation_heatmap(returns_df, chart_title="Correlation Matrix"):
    """Create Correlation Matrix Heatmap. Returns BytesIO buffer."""
    corr_matrix = returns_df.corr()

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, cbar_kws={"label": "Correlation"}
    )

    plt.title(chart_title, fontsize=12)
    ax.set_xlabel("Strategies")
    ax.set_ylabel("Strategies", fontsize=10)
    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    plt.close()
    buffer.seek(0)
    return buffer


def create_distribution_chart(monthly_returns_df, chart_title="Distribution of Monthly Returns"):
    """Create Bell Curve / Distribution chart. Returns BytesIO buffer."""
    long_df = monthly_returns_df.melt(var_name="Strategies", value_name="Monthly Return")

    plt.figure(figsize=(12, 6))
    colors = plt.colormaps["tab10"].colors
    assets = monthly_returns_df.columns.tolist()
    palette = {asset: colors[i % len(colors)] for i, asset in enumerate(assets)}

    sns.kdeplot(
        data=long_df, x='Monthly Return', hue='Strategies',
        palette=palette, common_norm=False, linewidth=2
    )

    plt.title(chart_title, fontsize=12)
    plt.xlabel('Monthly Return')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    plt.close()
    buffer.seek(0)
    return buffer


def create_box_plot(monthly_returns_df, chart_title="Box-Whisker Plot of Monthly Returns"):
    """Create Box-Whisker plot. Returns BytesIO buffer."""
    plt.figure(figsize=(12, 6))
    colors = plt.colormaps["tab10"].colors
    assets = monthly_returns_df.columns.tolist()
    palette = {asset: colors[i % len(colors)] for i, asset in enumerate(assets)}

    sns.boxplot(data=monthly_returns_df, palette=palette)

    plt.title(chart_title, fontsize=12)
    plt.xlabel('Asset')
    plt.ylabel('Monthly Return')

    handles = [plt.Line2D([], [], marker='s', linestyle='None', color=palette[asset], label=asset)
               for asset in assets]
    plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=300)
    plt.close()
    buffer.seek(0)
    return buffer
