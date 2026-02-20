"""
Tests for config.schema — Pydantic BacktestConfig validation.

Covers:
  - Valid config roundtrip (dict → BacktestConfig → .to_dict())
  - Typo detection (extra='forbid')
  - Type coercion / rejection on quarters
  - Cross-validation (tiered_config required when mode is 'tiered')
  - Defaults populated from config.defaults
  - Nested model construction from plain dicts
  - Quarter month validation
  - Pivot level validation
"""

import pytest
from pydantic import ValidationError

from config.schema import (
    AtrConfig,
    BacktestConfig,
    FlatConfig,
    IndexExitConfig,
    PivotConfig,
    RegimeFilterConfig,
    TieredConfig,
    TopKConfig,
    VolAdjustmentConfig,
)
from config.defaults import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_ATR_SL_MULTIPLIER,
    DEFAULT_ATR_TP_MULTIPLIER,
    DEFAULT_CATEGORY_SCHEME,
    DEFAULT_ENTRY_PRICE_WINDOW,
    DEFAULT_FLAT_SL_PCT,
    DEFAULT_FLAT_TP_PCT,
    DEFAULT_GENERATE_REPORT,
    DEFAULT_RUN_STOCK_SELECTION,
    DEFAULT_SELECTION_TYPE,
    DEFAULT_SL_MODE,
    DEFAULT_TP_MODE,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_REQUIRED = {
    "input_data_path": "../data/input.csv",
    "price_data_path": "../data/prices.parquet",
    "index_data_path": "../data/index.csv",
}


def _minimal(**overrides) -> dict:
    """Return the minimal valid config dict with optional overrides."""
    d = {**MINIMAL_REQUIRED, **overrides}
    return d


# ---------------------------------------------------------------------------
# Basic construction & defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    """BacktestConfig should fill defaults from config.defaults when only
    required paths are provided."""

    def test_minimal_config_has_correct_defaults(self):
        cfg = BacktestConfig(**_minimal())
        assert cfg.tp_mode == DEFAULT_TP_MODE
        assert cfg.sl_mode == DEFAULT_SL_MODE
        assert cfg.category_scheme == DEFAULT_CATEGORY_SCHEME
        assert cfg.entry_price_window == DEFAULT_ENTRY_PRICE_WINDOW
        assert cfg.run_stock_selection == DEFAULT_RUN_STOCK_SELECTION
        assert cfg.selection_type == DEFAULT_SELECTION_TYPE
        assert cfg.generate_report == DEFAULT_GENERATE_REPORT
        assert cfg.first_quarter is None
        assert cfg.last_quarter is None
        assert cfg.tiered_config is None
        assert cfg.atr_config is None
        assert cfg.pivot_config is None
        assert cfg.flat_config is None
        assert cfg.index_exit is None
        assert cfg.top_k_config is None

    def test_dimension_defaults_to_category_scheme(self):
        cfg = BacktestConfig(**_minimal(category_scheme="mcap"))
        assert cfg.selection_dimension == "mcap"
        assert cfg.weighting_dimension == "mcap"

    def test_explicit_dimensions_preserved(self):
        cfg = BacktestConfig(**_minimal(
            category_scheme="volatility",
            selection_dimension="mcap",
            weighting_dimension="mcap",
        ))
        assert cfg.selection_dimension == "mcap"
        assert cfg.weighting_dimension == "mcap"


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    """YAML-like dict → BacktestConfig → .to_dict() preserves values."""

    def test_to_dict_roundtrip(self):
        raw = _minimal(
            tp_mode="pivot",
            sl_mode="pivot",
            pivot_config={"lookback_days": 60, "tp_level": "R2", "sl_level": "S2"},
            first_quarter=202402,
            last_quarter=202411,
        )
        cfg = BacktestConfig(**raw)
        d = cfg.to_dict()

        assert d["tp_mode"] == "pivot"
        assert d["pivot_config"]["lookback_days"] == 60
        assert d["first_quarter"] == 202402
        assert d["last_quarter"] == 202411

    def test_model_dump_contains_all_fields(self):
        cfg = BacktestConfig(**_minimal())
        d = cfg.to_dict()
        # Spot-check a few keys that must exist
        for key in ("input_data_path", "tp_mode", "sl_mode", "entry_price_window",
                     "category_scheme", "run_stock_selection", "generate_report"):
            assert key in d


# ---------------------------------------------------------------------------
# Typo detection (extra='forbid')
# ---------------------------------------------------------------------------


class TestTypoDetection:
    def test_extra_field_raises(self):
        with pytest.raises(ValidationError, match="tp_modd"):
            BacktestConfig(**_minimal(tp_modd="flat"))

    def test_unknown_nested_field_raises(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BacktestConfig(**_minimal(
                atr_config={"period": 14, "bogus_field": 999}
            ))


# ---------------------------------------------------------------------------
# Type coercion / rejection
# ---------------------------------------------------------------------------


class TestTypeValidation:
    def test_quarter_string_coerces_to_int(self):
        cfg = BacktestConfig(**_minimal(first_quarter="202402"))
        assert cfg.first_quarter == 202402
        assert isinstance(cfg.first_quarter, int)

    def test_invalid_quarter_month_raises(self):
        with pytest.raises(ValidationError, match="month"):
            BacktestConfig(**_minimal(first_quarter=202403))

    def test_entry_price_window_zero_raises(self):
        with pytest.raises(ValidationError, match="entry_price_window"):
            BacktestConfig(**_minimal(entry_price_window=0))

    def test_entry_price_window_negative_raises(self):
        with pytest.raises(ValidationError, match="entry_price_window"):
            BacktestConfig(**_minimal(entry_price_window=-1))


# ---------------------------------------------------------------------------
# Cross-validation: tiered_config required when mode is 'tiered'
# ---------------------------------------------------------------------------


class TestTieredCrossValidation:
    def test_tiered_tp_without_config_raises(self):
        with pytest.raises(ValidationError, match="tiered_config"):
            BacktestConfig(**_minimal(tp_mode="tiered", tp_enabled=True))

    def test_tiered_sl_without_config_raises(self):
        with pytest.raises(ValidationError, match="tiered_config"):
            BacktestConfig(**_minimal(sl_mode="tiered", sl_enabled=True))

    def test_tiered_mode_with_config_ok(self):
        cfg = BacktestConfig(**_minimal(
            tp_mode="tiered",
            sl_mode="tiered",
            tiered_config={
                "tp_pct": {"high_volatility": 0.08},
                "sl_pct": {"high_volatility": 0.04},
            },
        ))
        assert cfg.tiered_config is not None
        assert cfg.tiered_config.tp_pct == {"high_volatility": 0.08}

    def test_tiered_mode_disabled_no_config_ok(self):
        """tiered mode set but both TP and SL disabled → no error."""
        cfg = BacktestConfig(**_minimal(
            tp_mode="tiered",
            sl_mode="tiered",
            tp_enabled=False,
            sl_enabled=False,
        ))
        assert cfg.tiered_config is None


# ---------------------------------------------------------------------------
# Nested model construction from plain dicts
# ---------------------------------------------------------------------------


class TestNestedModels:
    def test_atr_config_from_dict_fills_defaults(self):
        cfg = BacktestConfig(**_minimal(atr_config={"period": 20}))
        assert cfg.atr_config is not None
        assert cfg.atr_config.period == 20
        assert cfg.atr_config.tp_multiplier == DEFAULT_ATR_TP_MULTIPLIER
        assert cfg.atr_config.sl_multiplier == DEFAULT_ATR_SL_MULTIPLIER

    def test_flat_config_defaults(self):
        cfg = BacktestConfig(**_minimal(flat_config={}))
        assert cfg.flat_config.tp_pct == DEFAULT_FLAT_TP_PCT
        assert cfg.flat_config.sl_pct == DEFAULT_FLAT_SL_PCT

    def test_pivot_config_full(self):
        cfg = BacktestConfig(**_minimal(pivot_config={
            "lookback_days": 90,
            "tp_level": "R3",
            "sl_level": "S3",
        }))
        assert cfg.pivot_config.lookback_days == 90
        assert cfg.pivot_config.tp_level == "R3"

    def test_index_exit_nested(self):
        cfg = BacktestConfig(**_minimal(index_exit={
            "regime_filter": {"enabled": True, "ma_period": 50},
            "vol_adjustment": {"enabled": True, "lookback": 30},
        }))
        assert cfg.index_exit.regime_filter.enabled is True
        assert cfg.index_exit.regime_filter.ma_period == 50
        assert cfg.index_exit.vol_adjustment.lookback == 30

    def test_top_k_config_from_dict(self):
        cfg = BacktestConfig(**_minimal(top_k_config={"k": 15}))
        assert cfg.top_k_config.k == 15
        assert cfg.top_k_config.weighting_scheme == "equal"


# ---------------------------------------------------------------------------
# Pivot level validation
# ---------------------------------------------------------------------------


class TestPivotValidation:
    def test_invalid_tp_level_raises(self):
        with pytest.raises(ValidationError, match="tp_level"):
            PivotConfig(tp_level="R4")

    def test_invalid_sl_level_raises(self):
        with pytest.raises(ValidationError, match="sl_level"):
            PivotConfig(sl_level="S4")

    def test_valid_levels_accepted(self):
        for tp in ("R1", "R2", "R3"):
            for sl in ("S1", "S2", "S3"):
                p = PivotConfig(tp_level=tp, sl_level=sl)
                assert p.tp_level == tp
                assert p.sl_level == sl


# ---------------------------------------------------------------------------
# Sub-model unit tests
# ---------------------------------------------------------------------------


class TestSubModels:
    def test_top_k_extra_forbid(self):
        with pytest.raises(ValidationError):
            TopKConfig(k=10, bad="x")

    def test_tiered_default_fallback(self):
        t = TieredConfig()
        assert t.default_tp_pct == 0.05
        assert t.default_sl_pct == 0.05

    def test_atr_defaults(self):
        a = AtrConfig()
        assert a.period == DEFAULT_ATR_PERIOD
        assert a.tp_multiplier == DEFAULT_ATR_TP_MULTIPLIER
        assert a.sl_multiplier == DEFAULT_ATR_SL_MULTIPLIER

    def test_regime_filter_defaults(self):
        r = RegimeFilterConfig()
        assert r.enabled is False

    def test_vol_adjustment_defaults(self):
        v = VolAdjustmentConfig()
        assert v.enabled is False

    def test_index_exit_defaults(self):
        ie = IndexExitConfig()
        assert ie.regime_filter.enabled is False
        assert ie.vol_adjustment.enabled is False

    def test_flat_extra_forbid(self):
        with pytest.raises(ValidationError):
            FlatConfig(tp_pct=0.1, wrong="x")
