"""
Tests for target/analyse_target.py.

Covers the three public analysis functions with plot=False to avoid
interactive matplotlib display.
"""

import pytest
import pandas as pd
import numpy as np

from target.analyse_target import (
    analyze_labels_category_distribution,
    analyze_labels_stock_level,
    analyze_labels_temporal,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_target_data(quarters=None, n_stocks=6):
    """
    Create synthetic target data with columns: quarter, co_name, cap_cat, label.
    Stocks are split equally across large/mid/small.
    Label=1 for every other stock.
    """
    if quarters is None:
        quarters = [202302, 202305, 202308]
    
    cap_cats = ['large', 'mid', 'small']
    rows = []
    for q in quarters:
        for i in range(n_stocks):
            rows.append({
                'quarter': q,
                'co_name': f'STOCK_{i}',
                'cap_cat': cap_cats[i % 3],
                'label': i % 2,  # alternating 0, 1
            })
    return pd.DataFrame(rows)


# =============================================================================
# analyze_labels_category_distribution
# =============================================================================

class TestCategoryDistribution:

    def test_basic_output_structure(self):
        data = _make_target_data()
        result = analyze_labels_category_distribution(data, 202302, 202308, plot=False)
        assert 'quarter_range' in result
        assert 'num_datapoints' in result
        assert 'label_distribution' in result
        assert 'overall_cap_distribution' in result
        assert 'cap_in_label1' in result
        assert 'label1_rate_by_cap' in result

    def test_empty_quarter_range(self):
        data = _make_target_data(quarters=[202302])
        result = analyze_labels_category_distribution(data, 202502, 202508, plot=False)
        assert result == {}

    def test_label_counts_correct(self):
        data = _make_target_data(quarters=[202302], n_stocks=6)
        result = analyze_labels_category_distribution(data, 202302, 202302, plot=False)
        # 6 stocks, alternating labels: 3 with label=0, 3 with label=1
        assert result['label_distribution']['label_1']['count'] == 3
        assert result['label_distribution']['label_0']['count'] == 3
        assert result['num_datapoints'] == 6

    def test_all_zeros_no_crash(self):
        data = _make_target_data(quarters=[202302], n_stocks=4)
        data['label'] = 0  # all zeros
        result = analyze_labels_category_distribution(data, 202302, 202302, plot=False)
        assert result['label_distribution']['label_1']['count'] == 0


# =============================================================================
# analyze_labels_stock_level
# =============================================================================

class TestStockLevelAnalysis:

    def test_basic_output_structure(self):
        data = _make_target_data()
        result = analyze_labels_stock_level(data, 202302, 202308, plot=False)
        assert 'quarter_range' in result
        assert 'num_quarters' in result
        assert 'unique_stocks' in result
        assert 'concentration' in result

    def test_empty_quarter_range(self):
        data = _make_target_data(quarters=[202302])
        result = analyze_labels_stock_level(data, 202502, 202508, plot=False)
        assert result == {}

    def test_unique_stock_count(self):
        data = _make_target_data(quarters=[202302, 202305], n_stocks=4)
        result = analyze_labels_stock_level(data, 202302, 202305, plot=False)
        assert result['unique_stocks']['total'] == 4

    def test_single_quarter_persistence(self):
        """With only one quarter, no transitions are possible."""
        data = _make_target_data(quarters=[202302], n_stocks=4)
        result = analyze_labels_stock_level(data, 202302, 202302, plot=False)
        # Should still have valid output, persistence may be empty
        assert result['num_quarters'] == 1


# =============================================================================
# analyze_labels_temporal
# =============================================================================

class TestTemporalAnalysis:

    def test_basic_output_structure(self):
        data = _make_target_data()
        result = analyze_labels_temporal(data, 202302, 202308)
        assert 'quarter_range' in result
        assert 'quarters' in result
        assert 'temporal_data' in result

    def test_empty_quarter_range(self):
        data = _make_target_data(quarters=[202302])
        result = analyze_labels_temporal(data, 202502, 202508)
        assert result == {}

    def test_quarters_match_data(self):
        data = _make_target_data(quarters=[202302, 202305, 202308])
        result = analyze_labels_temporal(data, 202302, 202308)
        assert result['quarters'] == [202302, 202305, 202308]
