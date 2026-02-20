import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.logging_config import get_logger

logger = get_logger(__name__)

def analyze_labels_category_distribution(
        data: pd.DataFrame, first_quarter: int, last_quarter: int, plot: bool = True
    ) -> dict:
    """
    Analyze label distribution and market cap category breakdown for a given quarter range.
    
    Parameters:
        data : pd.DataFrame with columns [quarter, co_name, cap_cat, label]
            cap_cat should be in ['large', 'mid', 'small']
        first_quarter : int
            Start quarter in YYYYMM format (e.g., 202002)
        last_quarter : int
            End quarter in YYYYMM format (e.g., 202402)
    
    Returns:
        dict : Dictionary containing analysis results
    """
    # Filter data for the quarter range
    data = data[(data['quarter'] >= first_quarter) & (data['quarter'] <= last_quarter)].copy()
    
    num_datapoints = len(data)
    
    if num_datapoints == 0:
        logger.warning("No data found for quarter range %d to %d", first_quarter, last_quarter)
        return {}
    
    # 1. Compute Label distribution (raw numbers and percentages)
    label_counts = data['label'].value_counts().sort_index()
    # Number of data points labeled 1
    label_1_count = label_counts.get(1, 0)
    # Number of data points labeled 0
    label_0_count = label_counts.get(0, 0)

    # Percentage of data points labeled 1
    label_1_pct = (label_1_count / num_datapoints) * 100
    # Percentage of data points labeled 0
    label_0_pct = (label_0_count / num_datapoints) * 100
    
    logger.info("=== Label Distribution (Quarter %d to %d) ===", first_quarter, last_quarter)
    logger.info("Number of Datapoints: %d", num_datapoints)
    logger.info("Label 1: %d (%.2f%%)", label_1_count, label_1_pct)
    logger.info("Label 0: %d (%.2f%%)", label_0_count, label_0_pct)
    
    # 2. Compute Overall market cap category distribution
    logger.info("=== Overall Market Cap Distribution ===")
    cap_counts = data['cap_cat'].value_counts()
    cap_pcts = (cap_counts / num_datapoints) * 100
    overall_cap_dist = {}
    for cap in ['large', 'mid', 'small']:
        count = cap_counts.get(cap, 0)
        pct = cap_pcts.get(cap, 0)
        overall_cap_dist[cap] = {'count': count, 'pct': pct}
        logger.info("%s cap: %d (%.2f%%)", cap.capitalize(), count, pct)
    
    # 3. What is the Market Cap category distribution within label=1 stocks?
    label_1_stocks = data[data['label'] == 1]  # filter for rows labeled 1
    label_1_total = len(label_1_stocks)  # compute total number of positive data points
    
    logger.info("=== Market Cap Category Distribution within Label=1 Stocks ===")
    if label_1_total > 0:
        cap_in_label1 = label_1_stocks['cap_cat'].value_counts()
        cap_in_label1_pct = (cap_in_label1 / label_1_total) * 100
        
        for cap in ['large', 'mid', 'small']:
            count = cap_in_label1.get(cap, 0)
            pct = cap_in_label1_pct.get(cap, 0)
            logger.info("%s cap: %d (%.2f%%)", cap.capitalize(), count, pct)
    else:
        logger.info("No stocks with label=1")
    
    # 4. What is the percentage of positive data points in each market cap category? (Label=1 rate within each cap category)
    logger.info("=== Label=1 Rate by Cap Category ===")
    cap_label1_rates = {}
    for cap in ['large', 'mid', 'small']:
        cap_stocks = data[data['cap_cat'] == cap]  # filter for the category
        cap_total = len(cap_stocks)  # compute total num of data points in this category
        if cap_total > 0:
            cap_label1 = len(cap_stocks[cap_stocks['label'] == 1])  # compute number of positive data points in the cat
            cap_label1_rate = (cap_label1 / cap_total) * 100
            cap_label1_rates[cap] = {
                'total': cap_total,
                'label_1': cap_label1,
                'label_1_pct': cap_label1_rate
            }
            logger.info("%s cap: %d/%d labeled 1 (%.2f%%)", cap.capitalize(), cap_label1, cap_total, cap_label1_rate)
        else:
            cap_label1_rates[cap] = {'total': 0, 'label_1': 0, 'label_1_pct': 0}
            logger.info("%s cap: No data points for this category found", cap.capitalize())
    
    # Return results as dictionary for further use
    results = {
        'quarter_range': (first_quarter, last_quarter),
        'num_datapoints': num_datapoints,
        'label_distribution': {
            'label_1': {'count': label_1_count, 'pct': label_1_pct},
            'label_0': {'count': label_0_count, 'pct': label_0_pct}
        },
        'overall_cap_distribution': overall_cap_dist,
        'cap_in_label1': {
            cap: {
                'count': cap_in_label1.get(cap, 0) if label_1_total > 0 else 0,
                'pct': cap_in_label1_pct.get(cap, 0) if label_1_total > 0 else 0
            } for cap in ['large', 'mid', 'small']
        },
        'label1_rate_by_cap': cap_label1_rates
    }
    
    # Plotting
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        cap_colors = {'large': '#2ecc71', 'mid': '#3498db', 'small': '#e74c3c'}
        label_colors = {0: '#e74c3c', 1: '#2ecc71'}
        
        # ===== Stacked Bar Plot 1: Two bars with market cap category divisions =====
        ax1 = axes[0]
        
        # Get cap breakdown within each label
        label_0_stocks = data[data['label'] == 0]
        cap_in_label0 = label_0_stocks['cap_cat'].value_counts()
        
        # Data for stacked bars (Label 0 and Label 1)
        labels_x = ['Label 0', 'Label 1']
        large_counts = [cap_in_label0.get('large', 0), cap_in_label1.get('large', 0) if label_1_total > 0 else 0]
        mid_counts = [cap_in_label0.get('mid', 0), cap_in_label1.get('mid', 0) if label_1_total > 0 else 0]
        small_counts = [cap_in_label0.get('small', 0), cap_in_label1.get('small', 0) if label_1_total > 0 else 0]
        
        x = np.arange(len(labels_x))
        width = 0.5
        
        bars_large = ax1.bar(x, large_counts, width, label='Large Cap', color=cap_colors['large'])
        bars_mid = ax1.bar(x, mid_counts, width, bottom=large_counts, label='Mid Cap', color=cap_colors['mid'])
        bars_small = ax1.bar(x, small_counts, width, bottom=np.array(large_counts) + np.array(mid_counts), 
                            label='Small Cap', color=cap_colors['small'])
        
        ax1.set_xlabel('Label')
        ax1.set_ylabel('Number of Data Points')
        ax1.set_title(f'Label Distribution by Cap Category\n(Q{first_quarter} to Q{last_quarter})')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels_x)
        ax1.legend()
        
        # Add total counts on top of bars
        for i, (l, m, s) in enumerate(zip(large_counts, mid_counts, small_counts)):
            total = l + m + s
            ax1.annotate(f'{total}', xy=(i, total), ha='center', va='bottom', fontweight='bold')
        
        # ===== Stacked Bar Plot 2: Three bars with label divisions =====
        ax2 = axes[1]
        
        # Data for stacked bars (Large, Mid, Small)
        cap_x = ['Large Cap', 'Mid Cap', 'Small Cap']
        
        label_0_by_cap = []
        label_1_by_cap = []
        for cap in ['large', 'mid', 'small']:
            cap_stocks = data[data['cap_cat'] == cap]
            label_0_by_cap.append(len(cap_stocks[cap_stocks['label'] == 0]))
            label_1_by_cap.append(len(cap_stocks[cap_stocks['label'] == 1]))
        
        x2 = np.arange(len(cap_x))
        
        bars_label0 = ax2.bar(x2, label_0_by_cap, width, label='Label 0', color=label_colors[0])
        bars_label1 = ax2.bar(x2, label_1_by_cap, width, bottom=label_0_by_cap, label='Label 1', color=label_colors[1])
        
        ax2.set_xlabel('Cap Category')
        ax2.set_ylabel('Number of Data Points')
        ax2.set_title(f'Cap Category Distribution by Label\n(Q{first_quarter} to Q{last_quarter})')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(cap_x)
        ax2.legend()
        
        # Add total counts on top of bars
        for i, (l0, l1) in enumerate(zip(label_0_by_cap, label_1_by_cap)):
            total = l0 + l1
            ax2.annotate(f'{total}', xy=(i, total), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    return results


def analyze_labels_stock_level(
        data: pd.DataFrame, first_quarter: int, last_quarter: int, plot: bool = True
    ) -> dict:
    """
    Perform stock-level analysis including unique counts, concentration, and label persistence.
    
    Parameters:
        data : pd.DataFrame
            DataFrame with columns: quarter, co_name, cap_cat, label
        first_quarter : int
            Start quarter in YYYYMM format (e.g., 202002)
        last_quarter : int
            End quarter in YYYYMM format (e.g., 202402)
        plot : bool
            Whether to display the concentration bins bar plot (default: True)
    
    Returns:
        dict : Dictionary containing stock-level analysis results
    """
    # Filter data for the quarter range
    data = data[(data['quarter'] >= first_quarter) & (data['quarter'] <= last_quarter)].copy()
    
    if len(data) == 0:
        logger.warning("No data found for quarter range %d to %d", first_quarter, last_quarter)
        return {}
    
    # 1. Unique stock counts
    num_unique_stocks = data['co_name'].nunique()
    num_unique_stocks_with_label1 = data[data['label'] == 1]['co_name'].nunique()
    stocks_with_label1_pct = (num_unique_stocks_with_label1 / num_unique_stocks) * 100 if num_unique_stocks > 0 else 0
    
    logger.info("=== Unique Stock Analysis (Quarter %d to %d) ===", first_quarter, last_quarter)
    logger.info("Total unique stocks: %d", num_unique_stocks)
    logger.info("Stocks with label=1 at least once: %d (%.2f%%)", num_unique_stocks_with_label1, stocks_with_label1_pct)
    
    # 2. Concentration analysis - how often do stocks get label=1?
    num_quarters = data['quarter'].nunique()
    stock_label1_counts = data[data['label'] == 1].groupby('co_name').size()
    # stock_label1_counts: index: co_name, values: num of quarters the stock was labeled 1
    
    logger.info("=== Concentration Analysis (Label=1 Frequency) ===")
    logger.info("Number of quarters in range: %d", num_quarters)
    
    # Distribution of label=1 counts
    if len(stock_label1_counts) > 0:
        concentration_bins = {
            'once': (stock_label1_counts == 1).sum(),
            '2-3_times': ((stock_label1_counts >= 2) & (stock_label1_counts <= 3)).sum(),
            '4-5_times': ((stock_label1_counts >= 4) & (stock_label1_counts <= 5)).sum(),
            'more_than_5': (stock_label1_counts > 5).sum()
        }
        
        for bin_name, count in concentration_bins.items():
            pct = (count / num_unique_stocks_with_label1) * 100 if num_unique_stocks_with_label1 > 0 else 0
            logger.info("Label=1 %s: %d stocks (%.2f%%)", bin_name.replace('_', ' '), count, pct)
        
        # Top repeat winners
        top_n = min(10, len(stock_label1_counts))
        top_winners = stock_label1_counts.nlargest(top_n)
        logger.info("Top %d most frequent label=1 stocks:", top_n)
        for stock, count in top_winners.items():
            logger.info("  %s: %d times (%.1f%% of quarters)", stock, count, count/num_quarters*100)
    else:
        concentration_bins = {}
        top_winners = pd.Series(dtype=int)
        logger.info("No stocks with label=1")
    
    # 3. Label persistence analysis
    # For each stock, if label=1 in quarter Q, what's the probability of label=1 in Q+1?
    logger.info("=== Label Persistence Analysis ===")
    
    # Get sorted quarters
    quarters = sorted(data['quarter'].unique())
    quarter_to_next = {q: quarters[i+1] for i, q in enumerate(quarters[:-1])}
    
    # Find transitions
    persistence_data = []
    for stock in data['co_name'].unique():
        stock_data = data[data['co_name'] == stock].set_index('quarter')['label']
        for q, next_q in quarter_to_next.items():
            if q in stock_data.index and next_q in stock_data.index:
                persistence_data.append({
                    'current_label': stock_data[q],
                    'next_label': stock_data[next_q]
                })
    
    persistence_df = pd.DataFrame(persistence_data)
    
    if len(persistence_df) > 0:
        # P(label=1 in Q+1 | label=1 in Q)
        label1_current = persistence_df[persistence_df['current_label'] == 1]
        if len(label1_current) > 0:
            persistence_1_to_1 = (label1_current['next_label'] == 1).mean() * 100
            persistence_1_to_0 = (label1_current['next_label'] == 0).mean() * 100
        else:
            persistence_1_to_1 = 0
            persistence_1_to_0 = 0
        
        # P(label=1 in Q+1 | label=0 in Q)
        label0_current = persistence_df[persistence_df['current_label'] == 0]
        if len(label0_current) > 0:
            persistence_0_to_1 = (label0_current['next_label'] == 1).mean() * 100
            persistence_0_to_0 = (label0_current['next_label'] == 0).mean() * 100
        else:
            persistence_0_to_1 = 0
            persistence_0_to_0 = 0
        
        logger.info("Transitions analyzed: %d", len(persistence_df))
        logger.info("P(label=1 next | label=1 current): %.2f%%", persistence_1_to_1)
        logger.info("P(label=0 next | label=1 current): %.2f%%", persistence_1_to_0)
        logger.info("P(label=1 next | label=0 current): %.2f%%", persistence_0_to_1)
        logger.info("P(label=0 next | label=0 current): %.2f%%", persistence_0_to_0)
        
        # Persistence ratio - how much more likely is label=1 if previously label=1?
        if persistence_0_to_1 > 0:
            persistence_ratio = persistence_1_to_1 / persistence_0_to_1
            logger.info("Persistence ratio: %.2fx", persistence_ratio)
            logger.info("(How much more likely to get label=1 if previous quarter was label=1)")
        
        persistence_results = {
            'num_transitions': len(persistence_df),
            'prob_1_given_1': persistence_1_to_1,
            'prob_0_given_1': persistence_1_to_0,
            'prob_1_given_0': persistence_0_to_1,
            'prob_0_given_0': persistence_0_to_0,
            'persistence_ratio': persistence_ratio if persistence_0_to_1 > 0 else None
        }
    else:
        logger.info("Not enough data for persistence analysis")
        persistence_results = {}
    
    # Plotting: Concentration bins bar chart
    if plot and len(concentration_bins) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        bin_labels = ['Once', '2-3 Times', '4-5 Times', 'More than 5']
        bin_keys = ['once', '2-3_times', '4-5_times', 'more_than_5']
        bin_values = [concentration_bins.get(k, 0) for k in bin_keys]
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = ax.bar(bin_labels, bin_values, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add count and percentage labels on top of each bar
        for bar, val in zip(bars, bin_values):
            pct = (val / num_unique_stocks_with_label1) * 100 if num_unique_stocks_with_label1 > 0 else 0
            ax.annotate(f'{val}\n({pct:.1f}%)', 
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Label=1 Frequency')
        ax.set_ylabel('Number of Stocks')
        ax.set_title(f'Concentration Analysis: How Often Stocks Get Label=1\n(Q{first_quarter} to Q{last_quarter})')
        
        plt.tight_layout()
        plt.show()
    
    # Return results
    results = {
        'quarter_range': (first_quarter, last_quarter),
        'num_quarters': num_quarters,
        'unique_stocks': {
            'total': num_unique_stocks,
            'with_label1': num_unique_stocks_with_label1,
            'with_label1_pct': stocks_with_label1_pct
        },
        'concentration': {
            'bins': concentration_bins,
            'top_winners': top_winners.to_dict() if len(top_winners) > 0 else {}
        },
        'persistence': persistence_results
    }
    
    return results


def analyze_labels_temporal(data: pd.DataFrame, first_quarter: int, last_quarter: int) -> dict:
    """
    Create a temporal bar plot showing label=1 stock counts by market cap category for each quarter.
    
    Parameters:
        data : pd.DataFrame
            DataFrame with columns: quarter, co_name, cap_cat, label
            cap_cat should be in ['large', 'mid', 'small']
        first_quarter : int
            Start quarter in YYYYMM format (e.g., 202002)
        last_quarter : int
            End quarter in YYYYMM format (e.g., 202402)
    
    Returns:
        dict : Dictionary containing temporal analysis results with label=1 counts per quarter per cap category
    """
    # Filter data for the quarter range
    data = data[(data['quarter'] >= first_quarter) & (data['quarter'] <= last_quarter)].copy()
    
    if len(data) == 0:
        logger.warning("No data found for quarter range %d to %d", first_quarter, last_quarter)
        return {}
    
    # Get sorted list of quarters
    quarters = sorted(data['quarter'].unique())
    
    # Count label=1 stocks by cap category for each quarter
    temporal_data = {'quarter': [], 'large': [], 'mid': [], 'small': []}
    
    for q in quarters:
        q_data = data[(data['quarter'] == q) & (data['label'] == 1)]
        cap_counts = q_data['cap_cat'].value_counts()
        
        temporal_data['quarter'].append(q)
        temporal_data['large'].append(cap_counts.get('large', 0))
        temporal_data['mid'].append(cap_counts.get('mid', 0))
        temporal_data['small'].append(cap_counts.get('small', 0))
    
    temporal_df = pd.DataFrame(temporal_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(12, len(quarters) * 0.8), 6))
    
    x = np.arange(len(quarters))
    width = 0.25
    
    cap_colors = {'large': '#2ecc71', 'mid': '#3498db', 'small': '#e74c3c'}
    
    bars_large = ax.bar(x - width, temporal_df['large'], width, label='Large Cap', color=cap_colors['large'])
    bars_mid = ax.bar(x, temporal_df['mid'], width, label='Mid Cap', color=cap_colors['mid'])
    bars_small = ax.bar(x + width, temporal_df['small'], width, label='Small Cap', color=cap_colors['small'])
    
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Number of Stocks with Label=1')
    ax.set_title(f'Temporal Distribution of Label=1 Stocks by Cap Category\n(Q{first_quarter} to Q{last_quarter})')
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in quarters], rotation=45, ha='right')
    ax.legend()
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()
    
    # Return results
    results = {
        'quarter_range': (first_quarter, last_quarter),
        'quarters': quarters,
        'temporal_data': temporal_df.to_dict('list')
    }
    
    return results

