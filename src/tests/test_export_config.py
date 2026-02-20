"""
Tests for export_config.py.

Covers _expand_trial_params (pure-logic function that can be tested
without an actual Optuna study or database).
"""

import pytest

from export_config import _expand_trial_params


# =============================================================================
# _expand_trial_params
# =============================================================================

class TestExpandTrialParams:

    def _make_tuning_config(self, **extra_search):
        """Create minimal tuning config with optional extra search space entries."""
        search_space = {
            'tp_mode': {'type': 'categorical', 'choices': ['flat', 'tiered', 'atr']},
            **extra_search,
        }
        return {'search_space': search_space}

    def test_no_idx_keys_returns_copy(self):
        params = {'tp_mode': 'flat', 'tp_pct': 0.05}
        config = self._make_tuning_config()
        result = _expand_trial_params(params, config)
        assert result == params
        # Ensure it's a copy, not the same object
        assert result is not params

    def test_list_valued_categorical_expanded(self):
        config = self._make_tuning_config(
            category_counts={
                'type': 'categorical',
                'choices': [[8, 7, 5], [15, 10, 5], [20, 15, 10]],
            }
        )
        params = {'tp_mode': 'flat', 'category_counts_idx': 1}
        result = _expand_trial_params(params, config)
        assert result['category_counts'] == [15, 10, 5]
        assert 'category_counts_idx' not in result

    def test_idx_out_of_range_kept(self):
        config = self._make_tuning_config(
            category_counts={
                'type': 'categorical',
                'choices': [[8, 7, 5]],
            }
        )
        params = {'category_counts_idx': 99}  # out of range
        result = _expand_trial_params(params, config)
        # idx should still be removed even if not expanded
        assert 'category_counts_idx' not in result

    def test_non_list_categorical_not_expanded(self):
        """Non-list categoricals (e.g., string choices) should not be touched."""
        config = self._make_tuning_config()
        params = {'tp_mode': 'flat'}
        result = _expand_trial_params(params, config)
        assert result == {'tp_mode': 'flat'}

    def test_mode_specific_blocks_searched(self):
        """Parameters in mode-specific blocks (tiered_tpsl, etc.) should also expand."""
        config = {
            'search_space': {},
            'tiered_tpsl': {
                'tiered_tp_thresholds': {
                    'type': 'categorical',
                    'choices': [[0.05, 0.08, 0.12], [0.10, 0.15, 0.20]],
                }
            },
        }
        params = {'tiered_tp_thresholds_idx': 0}
        result = _expand_trial_params(params, config)
        assert result['tiered_tp_thresholds'] == [0.05, 0.08, 0.12]
        assert 'tiered_tp_thresholds_idx' not in result
