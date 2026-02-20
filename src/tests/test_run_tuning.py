"""
Tests for run_tuning.py.

Covers DataCache, _parse_objective_config, and get_sampler — pure-logic
functions that can be tested without actual data files or running Optuna studies.
"""

import pytest
import optuna

from run_tuning import DataCache, _parse_objective_config, get_sampler


# =============================================================================
# DataCache
# =============================================================================

class TestDataCache:

    def test_initial_state(self):
        dc = DataCache()
        assert dc.loaded is False
        assert dc.input_data is None
        assert dc.price_data is None
        assert dc.index_data is None

    def test_two_instances_are_independent(self):
        """After removing the singleton pattern, each instance should be independent."""
        dc1 = DataCache()
        dc2 = DataCache()
        assert dc1 is not dc2


# =============================================================================
# _parse_objective_config
# =============================================================================

class TestParseObjectiveConfig:

    def test_default_when_empty(self):
        result = _parse_objective_config({})
        assert result['is_multi_objective'] is False
        assert result['objective_names'] == ['calmar']
        assert result['directions'] == ['maximize']

    def test_single_objective(self):
        result = _parse_objective_config({'objective': 'cagr', 'direction': 'maximize'})
        assert result['is_multi_objective'] is False
        assert result['objective_names'] == ['cagr']

    def test_multi_objective(self):
        result = _parse_objective_config({
            'objectives': ['cagr', 'mdd'],
            'directions': ['maximize', 'minimize'],
        })
        assert result['is_multi_objective'] is True
        assert result['objective_names'] == ['cagr', 'mdd']
        assert result['directions'] == ['maximize', 'minimize']
        assert len(result['failure_penalties']) == 2

    def test_both_single_and_multi_raises(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            _parse_objective_config({'objective': 'cagr', 'objectives': ['cagr', 'mdd']})

    def test_multi_objective_mismatched_directions_raises(self):
        with pytest.raises(ValueError, match="directions.*length"):
            _parse_objective_config({
                'objectives': ['cagr', 'mdd'],
                'directions': ['maximize'],  # wrong length
            })

    def test_multi_objective_missing_directions_raises(self):
        with pytest.raises(ValueError, match="directions.*required"):
            _parse_objective_config({
                'objectives': ['cagr', 'mdd'],
            })

    def test_unknown_objective_raises(self):
        with pytest.raises(ValueError, match="Unknown objective"):
            _parse_objective_config({'objective': 'not_a_real_metric'})

    def test_single_objective_less_than_two_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            _parse_objective_config({'objectives': ['cagr']})

    def test_scalar_failure_penalty_broadcast(self):
        result = _parse_objective_config({
            'objectives': ['cagr', 'mdd'],
            'directions': ['maximize', 'minimize'],
            'failure_penalties': -999.0,
        })
        assert result['failure_penalties'] == [-999.0, -999.0]

    def test_calmar_cap_default(self):
        result = _parse_objective_config({})
        from config.defaults import DEFAULT_CALMAR_CAP
        assert result['calmar_cap'] == DEFAULT_CALMAR_CAP

    def test_calmar_cap_override(self):
        result = _parse_objective_config({'objective': 'calmar', 'calmar_cap': 20.0})
        assert result['calmar_cap'] == 20.0


# =============================================================================
# get_sampler
# =============================================================================

class TestGetSampler:

    @pytest.mark.parametrize("name,expected_type", [
        ("TPE", optuna.samplers.TPESampler),
        ("Random", optuna.samplers.RandomSampler),
        ("CmaEs", optuna.samplers.CmaEsSampler),
        ("NSGA-II", optuna.samplers.NSGAIISampler),
        ("NSGA-III", optuna.samplers.NSGAIIISampler),
    ])
    def test_valid_samplers(self, name, expected_type):
        sampler = get_sampler(name)
        assert isinstance(sampler, expected_type)

    def test_unknown_sampler_raises(self):
        with pytest.raises(ValueError, match="Unknown sampler"):
            get_sampler("FooSampler")

    def test_cmaes_multi_obj_raises(self):
        with pytest.raises(ValueError, match="CmaEsSampler"):
            get_sampler("CmaEs", is_multi_obj=True)

    def test_nsga_ii_multi_obj_ok(self):
        sampler = get_sampler("NSGA-II", is_multi_obj=True)
        assert isinstance(sampler, optuna.samplers.NSGAIISampler)
