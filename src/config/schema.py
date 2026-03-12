"""
Typed Configuration Schema via Pydantic

Provides validated, typed models for the backtest configuration.
Catches typos at load time (extra='forbid'), enforces types, and
supplies defaults from config.defaults so that consumers can use
direct attribute access instead of .get('key', DEFAULT).

Usage:
    from config.schema import BacktestConfig

    cfg = BacktestConfig(**yaml.safe_load(open('strategy_config.yaml')))
    print(cfg.tp_mode)           # attribute access with IDE autocomplete
    d = cfg.to_dict()            # plain dict for backward compat
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from config.defaults import (
    # Selection
    DEFAULT_CATEGORY_COUNTS,
    DEFAULT_CATEGORY_SCHEME,
    DEFAULT_CATEGORY_WEIGHTS,
    DEFAULT_MIN_PROB_THRESHOLD,
    DEFAULT_SORT_BY,
    DEFAULT_SELECTION_TYPE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_WEIGHTING,
    DEFAULT_TPSL_CATEGORY_DIMENSION,
    DEFAULT_WEIGHTING_SCHEME,
    # TP / SL
    DEFAULT_TP_ENABLED,
    DEFAULT_SL_ENABLED,
    DEFAULT_TP_MODE,
    DEFAULT_SL_MODE,
    DEFAULT_TPSL_FALLBACK_PCT,
    # ATR
    DEFAULT_ATR_PERIOD,
    DEFAULT_ATR_TP_MULTIPLIER,
    DEFAULT_ATR_SL_MULTIPLIER,
    # Pivot
    DEFAULT_PIVOT_LOOKBACK_DAYS,
    DEFAULT_PIVOT_TP_LEVEL,
    DEFAULT_PIVOT_SL_LEVEL,
    # Flat
    DEFAULT_FLAT_TP_PCT,
    DEFAULT_FLAT_SL_PCT,
    # Index exit / regime
    DEFAULT_REGIME_FILTER_ENABLED,
    DEFAULT_REGIME_MA_PERIOD,
    DEFAULT_REGIME_EXIT_THRESHOLD,
    DEFAULT_VOL_ADJUSTMENT_ENABLED,
    DEFAULT_VOL_LOOKBACK,
    DEFAULT_HIGH_VOL_THRESHOLD,
    DEFAULT_LOW_VOL_THRESHOLD,
    DEFAULT_HIGH_VOL_MULTIPLIER,
    DEFAULT_LOW_VOL_MULTIPLIER,
    # Entry
    DEFAULT_ENTRY_PRICE_WINDOW,
    # Misc
    DEFAULT_RUN_STOCK_SELECTION,
    DEFAULT_GENERATE_REPORT,
)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class TopKConfig(BaseModel):
    """Configuration for top-k stock selection."""
    model_config = ConfigDict(extra="forbid")

    k: int = DEFAULT_TOP_K
    weighting_scheme: str = DEFAULT_TOP_K_WEIGHTING
    category_caps: Optional[dict[str, int]] = None  # e.g., {'smallcap': 10}


class TieredConfig(BaseModel):
    """Category-specific TP/SL percentages (tiered mode)."""
    model_config = ConfigDict(extra="forbid")

    tp_pct: dict[str, float] = {}
    sl_pct: dict[str, float] = {}
    default_tp_pct: float = DEFAULT_TPSL_FALLBACK_PCT
    default_sl_pct: float = DEFAULT_TPSL_FALLBACK_PCT


class AtrQuarterOverride(BaseModel):
    """Per-quarter ATR parameter overrides.  All fields are optional;
    only the fields present will replace the base AtrConfig values."""
    model_config = ConfigDict(extra="forbid")

    period: Optional[int] = None
    tp_multiplier: Optional[float] = None
    sl_multiplier: Optional[float] = None


class AtrConfig(BaseModel):
    """ATR-based TP/SL configuration."""
    model_config = ConfigDict(extra="forbid")

    period: int = DEFAULT_ATR_PERIOD
    tp_multiplier: float = DEFAULT_ATR_TP_MULTIPLIER
    sl_multiplier: float = DEFAULT_ATR_SL_MULTIPLIER
    quarter_overrides: Optional[dict[int, AtrQuarterOverride]] = None

    @field_validator("quarter_overrides", mode="before")
    @classmethod
    def _coerce_and_validate_quarter_keys(cls, v):
        if v is None:
            return v
        validated: dict[int, AtrQuarterOverride] = {}
        for key, val in v.items():
            key = int(key)
            month = key % 100
            if month not in {2, 5, 8, 11}:
                raise ValueError(
                    f"quarter_overrides key {key} has month {month:02d}; "
                    f"expected one of 02, 05, 08, 11"
                )
            validated[key] = val
        return validated


class PivotConfig(BaseModel):
    """Pivot-point based TP/SL configuration."""
    model_config = ConfigDict(extra="forbid")

    lookback_days: int = DEFAULT_PIVOT_LOOKBACK_DAYS
    tp_level: str = DEFAULT_PIVOT_TP_LEVEL
    sl_level: str = DEFAULT_PIVOT_SL_LEVEL

    @field_validator("tp_level")
    @classmethod
    def _validate_tp_level(cls, v: str) -> str:
        allowed = {"R1", "R1_R2", "R2", "R2_R3", "R3"}
        if v not in allowed:
            raise ValueError(f"tp_level must be one of {allowed}, got '{v}'")
        return v

    @field_validator("sl_level")
    @classmethod
    def _validate_sl_level(cls, v: str) -> str:
        allowed = {"S1", "S1_S2", "S2", "S2_S3", "S3"}
        if v not in allowed:
            raise ValueError(f"sl_level must be one of {allowed}, got '{v}'")
        return v


class FlatConfig(BaseModel):
    """Flat (constant %) TP/SL configuration."""
    model_config = ConfigDict(extra="forbid")

    tp_pct: float = DEFAULT_FLAT_TP_PCT
    sl_pct: float = DEFAULT_FLAT_SL_PCT


class RegimeFilterConfig(BaseModel):
    """Market regime filter for index-guided exits."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = DEFAULT_REGIME_FILTER_ENABLED
    ma_period: int = DEFAULT_REGIME_MA_PERIOD
    exit_threshold: float = DEFAULT_REGIME_EXIT_THRESHOLD


class VolAdjustmentConfig(BaseModel):
    """Volatility-based TP/SL adjustment configuration."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = DEFAULT_VOL_ADJUSTMENT_ENABLED
    lookback: int = DEFAULT_VOL_LOOKBACK
    high_vol_threshold: float = DEFAULT_HIGH_VOL_THRESHOLD
    low_vol_threshold: float = DEFAULT_LOW_VOL_THRESHOLD
    high_vol_multiplier: float = DEFAULT_HIGH_VOL_MULTIPLIER
    low_vol_multiplier: float = DEFAULT_LOW_VOL_MULTIPLIER


class IndexExitConfig(BaseModel):
    """Index-guided exit configuration (regime filter + vol adjustment)."""
    model_config = ConfigDict(extra="forbid")

    regime_filter: RegimeFilterConfig = RegimeFilterConfig()
    vol_adjustment: VolAdjustmentConfig = VolAdjustmentConfig()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class BacktestConfig(BaseModel):
    """
    Validated backtest configuration.

    Mirrors the strategy YAML structure.  All fields have sensible defaults
    except the three mandatory data paths.  ``extra='forbid'`` catches typos.
    """
    model_config = ConfigDict(extra="forbid")

    # -- Paths (required) ----------------------------------------------------
    input_data_path: str
    price_data_path: str
    index_data_path: str

    # -- Quarters (optional filtering) ---------------------------------------
    first_quarter: Optional[int] = None
    last_quarter: Optional[int] = None

    @field_validator("first_quarter", "last_quarter", mode="before")
    @classmethod
    def _coerce_quarter(cls, v):
        if v is None:
            return v
        v = int(v)
        month = v % 100
        if month not in {2, 5, 8, 11}:
            raise ValueError(
                f"Quarter {v} has month {month:02d}; expected one of 02, 05, 08, 11"
            )
        return v

    # -- Dimensions ----------------------------------------------------------
    category_scheme: str = DEFAULT_CATEGORY_SCHEME
    selection_dimension: Optional[str] = None
    weighting_dimension: Optional[str] = None
    tpsl_category_dimension: str = DEFAULT_TPSL_CATEGORY_DIMENSION

    # -- Selection -----------------------------------------------------------
    run_stock_selection: bool = DEFAULT_RUN_STOCK_SELECTION
    selection_type: str = DEFAULT_SELECTION_TYPE
    category_counts: list[int] = list(DEFAULT_CATEGORY_COUNTS)
    category_weights: list[float] = list(DEFAULT_CATEGORY_WEIGHTS)
    sort_by: str = DEFAULT_SORT_BY
    min_prob_threshold: Optional[float] = DEFAULT_MIN_PROB_THRESHOLD
    top_k_config: Optional[TopKConfig] = None
    category_based_selection_weighting_scheme: str = DEFAULT_WEIGHTING_SCHEME

    # -- TP / SL -------------------------------------------------------------
    tp_enabled: bool = DEFAULT_TP_ENABLED
    sl_enabled: bool = DEFAULT_SL_ENABLED
    tp_mode: str = DEFAULT_TP_MODE
    sl_mode: str = DEFAULT_SL_MODE
    tiered_config: Optional[TieredConfig] = None
    atr_config: Optional[AtrConfig] = None
    pivot_config: Optional[PivotConfig] = None
    flat_config: Optional[FlatConfig] = None

    # -- Entry ---------------------------------------------------------------
    entry_price_window: int = DEFAULT_ENTRY_PRICE_WINDOW

    # -- Cash / risk-free rate -----------------------------------------------
    cash_appreciation_rate: Optional[float] = None

    # -- Index exit ----------------------------------------------------------
    index_exit: Optional[IndexExitConfig] = None

    # -- Reporting -----------------------------------------------------------
    generate_report: bool = DEFAULT_GENERATE_REPORT
    launch_dashboard: bool = False
    report_sub_periods: Optional[list[list[int]]] = None

    # -- Tuning-only ---------------------------------------------------------
    validate_input_data: Optional[bool] = None

    # -- Validators ----------------------------------------------------------

    @model_validator(mode="after")
    def _default_dimensions(self) -> "BacktestConfig":
        """Default selection/weighting dimensions to category_scheme when None."""
        if self.selection_dimension is None:
            self.selection_dimension = self.category_scheme
        if self.weighting_dimension is None:
            self.weighting_dimension = self.category_scheme
        return self

    @model_validator(mode="after")
    def _require_tiered_config_when_tiered(self) -> "BacktestConfig":
        """If tp_mode or sl_mode is 'tiered', tiered_config must be provided."""
        needs_tiered = (
            (self.tp_enabled and self.tp_mode == "tiered")
            or (self.sl_enabled and self.sl_mode == "tiered")
        )
        if needs_tiered and self.tiered_config is None:
            raise ValueError(
                "tiered_config must be set when tp_mode or sl_mode is 'tiered' "
                "and the corresponding exit is enabled"
            )
        return self

    @field_validator("entry_price_window")
    @classmethod
    def _validate_entry_window(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"entry_price_window must be >= 1, got {v}")
        return v

    # -- Helpers -------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dict (backward compat with code expecting dicts)."""
        return self.model_dump(exclude_none=False)
