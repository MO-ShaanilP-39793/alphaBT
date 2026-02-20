"""
Tests for validate_preselected_input().

Focused on the validation logic for preselected portfolios:
column checks, category validation, weight warnings, tiered-mode guards.
"""

import pytest
import warnings
import pandas as pd
from tests.conftest import Q1, Q2, STOCK_NAMES, MCAP_CATEGORIES

from backtest_strategy import validate_preselected_input


# =============================================================================
# HELPERS
# =============================================================================

def _make_preselected(with_category=True, weight=None, stocks=None, quarters=None):
    """Build a minimal preselected DataFrame."""
    stocks = stocks or STOCK_NAMES[:4]
    quarters = quarters or [Q1]
    w = weight or (1.0 / len(stocks))
    rows = []
    for q in quarters:
        for name in stocks:
            row = {"quarter": q, "co_name": name, "stock_weight": w}
            if with_category:
                row["category"] = MCAP_CATEGORIES.get(name, "largecap")
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# HAPPY PATHS
# =============================================================================

class TestValidPreselected:

    def test_valid_input_with_category(self):
        """Happy path: category column present → renamed to 'cat'."""
        df = _make_preselected(with_category=True)
        result, has_cat = validate_preselected_input(
            df, "mcap", "flat", "flat", True, True, False
        )
        assert has_cat is True
        assert "cat" in result.columns
        assert "category" not in result.columns

    def test_valid_input_without_category_flat_mode(self):
        """No category + flat mode → works fine."""
        df = _make_preselected(with_category=False)
        result, has_cat = validate_preselected_input(
            df, "mcap", "flat", "flat", True, True, False
        )
        assert has_cat is False
        assert "cat" not in result.columns


# =============================================================================
# MISSING COLUMNS
# =============================================================================

class TestMissingColumns:

    def test_missing_stock_weight(self):
        """Missing stock_weight → ValueError."""
        df = pd.DataFrame({
            "quarter": [Q1], "co_name": ["STOCK_A"],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_preselected_input(df, "mcap", "flat", "flat", True, True, False)

    def test_missing_quarter(self):
        """Missing quarter → ValueError."""
        df = pd.DataFrame({
            "co_name": ["STOCK_A"], "stock_weight": [1.0],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_preselected_input(df, "mcap", "flat", "flat", True, True, False)

    def test_missing_co_name(self):
        """Missing co_name → ValueError."""
        df = pd.DataFrame({
            "quarter": [Q1], "stock_weight": [1.0],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_preselected_input(df, "mcap", "flat", "flat", True, True, False)


# =============================================================================
# TIERED MODE GUARDS
# =============================================================================

class TestTieredModeGuards:

    def test_tiered_no_category_no_config_tp(self):
        """tp_mode='tiered', no category, no tiered_config → ValueError."""
        df = _make_preselected(with_category=False)
        with pytest.raises(ValueError, match="tiered"):
            validate_preselected_input(
                df, "mcap",
                tp_mode="tiered", sl_mode="flat",
                tp_enabled=True, sl_enabled=True,
                has_tiered_config=False,
            )

    def test_tiered_no_category_no_config_sl(self):
        """sl_mode='tiered', no category, no tiered_config → ValueError."""
        df = _make_preselected(with_category=False)
        with pytest.raises(ValueError, match="tiered"):
            validate_preselected_input(
                df, "mcap",
                tp_mode="flat", sl_mode="tiered",
                tp_enabled=True, sl_enabled=True,
                has_tiered_config=False,
            )

    def test_tiered_no_category_with_config(self):
        """Tiered mode, no category, but has tiered_config → OK."""
        df = _make_preselected(with_category=False)
        result, has_cat = validate_preselected_input(
            df, "mcap",
            tp_mode="tiered", sl_mode="tiered",
            tp_enabled=True, sl_enabled=True,
            has_tiered_config=True,
        )
        assert has_cat is False

    def test_tiered_with_category(self):
        """Tiered mode + category column → works (no need for tiered_config)."""
        df = _make_preselected(with_category=True)
        result, has_cat = validate_preselected_input(
            df, "mcap",
            tp_mode="tiered", sl_mode="tiered",
            tp_enabled=True, sl_enabled=True,
            has_tiered_config=False,
        )
        assert has_cat is True

    def test_tiered_disabled_no_category_ok(self):
        """Tiered mode set, but tp/sl both disabled → no error even without category."""
        df = _make_preselected(with_category=False)
        result, has_cat = validate_preselected_input(
            df, "mcap",
            tp_mode="tiered", sl_mode="tiered",
            tp_enabled=False, sl_enabled=False,
            has_tiered_config=False,
        )
        assert has_cat is False


# =============================================================================
# WARNINGS
# =============================================================================

class TestWarnings:

    def test_weight_sum_warning(self):
        """Weights not summing to 1.0 → UserWarning."""
        df = _make_preselected(with_category=False, weight=0.2)  # 4 stocks × 0.2 = 0.8
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_preselected_input(
                df, "mcap", "flat", "flat", True, True, False,
            )
        weight_warnings = [x for x in w if "stock weights sum" in str(x.message)]
        assert len(weight_warnings) > 0, "Expected a warning about weights not summing to 1.0"

    def test_no_weight_warning_when_correct(self):
        """Weights summing to 1.0 → no warning."""
        df = _make_preselected(with_category=False, weight=0.25)  # 4 stocks × 0.25 = 1.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_preselected_input(
                df, "mcap", "flat", "flat", True, True, False,
            )
        weight_warnings = [x for x in w if "stock weights sum" in str(x.message)]
        assert len(weight_warnings) == 0

    def test_invalid_category_values_warning(self):
        """Category values not matching scheme → UserWarning."""
        df = pd.DataFrame({
            "quarter": [Q1, Q1],
            "co_name": ["STOCK_A", "STOCK_B"],
            "stock_weight": [0.5, 0.5],
            "category": ["invalid_cat", "also_invalid"],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_preselected_input(
                df, "mcap", "flat", "flat", True, True, False,
            )
        cat_warnings = [x for x in w if "not matching" in str(x.message)]
        assert len(cat_warnings) > 0, "Expected a warning about invalid categories"
