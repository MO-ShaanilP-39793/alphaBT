"""
Centralized Default Values & Constants

Every default value / hardcoded constant used across the alphaBT codebase lives
here.  All consumer modules should import from this file instead of defining
inline fallbacks, ensuring a single source of truth.

Usage:
    from config.defaults import INITIAL_CAPITAL, DEFAULT_TP_MODE
    # or
    from config import INITIAL_CAPITAL
"""

# =============================================================================
# CAPITAL & FINANCIAL
# =============================================================================

INITIAL_CAPITAL = 1_000_000_000        # ₹100 Crores
RISK_FREE_RATE = 0.065                 # 6.5 % (Indian G-Sec benchmark)

# =============================================================================
# TIME CONSTANTS
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25                 # Used for CAGR calculations

# =============================================================================
# STOCK SELECTION
# =============================================================================

DEFAULT_SELECTION_TYPE = 'category_based'
DEFAULT_CATEGORY_SCHEME = 'volatility'
DEFAULT_CATEGORY_COUNTS = [10, 10, 10]
DEFAULT_CATEGORY_WEIGHTS = [0.33, 0.33, 0.34]
DEFAULT_SORT_BY = 'probability'
DEFAULT_MIN_PROB_THRESHOLD = None
DEFAULT_WEIGHTING_SCHEME = 'use_category_weights'

# Category name constants (single source of truth)
VOLATILITY_CATEGORIES = ['high_volatility', 'medium_volatility', 'low_volatility']
MCAP_CATEGORIES = ['largecap', 'midcap', 'smallcap']

DIMENSION_CATEGORIES = {
    'volatility': VOLATILITY_CATEGORIES,
    'mcap': MCAP_CATEGORIES,
}


def get_dimension_categories(dimension: str) -> list:
    """Return the ordered list of category names for a given dimension.

    Parameters:
        dimension: 'volatility' or 'mcap'

    Returns:
        List of category name strings

    Raises:
        ValueError: If dimension is not recognised
    """
    try:
        return DIMENSION_CATEGORIES[dimension]
    except KeyError:
        raise ValueError(
            f"Unknown dimension: '{dimension}'. Use 'volatility' or 'mcap'."
        )

# Cross-dimensional selection/weighting
# Controls which dimension TP/SL tiered-mode config uses: 'selection', 'weighting',
# or explicit 'volatility'/'mcap'. Defaults to 'selection' (legacy behavior).
DEFAULT_TPSL_CATEGORY_DIMENSION = 'selection'

# Top-k selection
DEFAULT_TOP_K = 30
DEFAULT_TOP_K_WEIGHTING = 'equal'
DEFAULT_TOP_K_CATEGORY_CAPS = None  # No per-category caps by default

# Risk-adjusted ranking epsilon (avoids division-by-zero)
RISK_ADJUSTED_EPSILON = 1e-8

# =============================================================================
# TRADE ENTRY
# =============================================================================

DEFAULT_ENTRY_PRICE_WINDOW = 3

# =============================================================================
# TP / SL CORE
# =============================================================================

DEFAULT_TP_MODE = 'flat'
DEFAULT_SL_MODE = 'flat'
DEFAULT_TP_ENABLED = True
DEFAULT_SL_ENABLED = True

# Fallback percentage when no category or config entry is found (tiered mode)
DEFAULT_TPSL_FALLBACK_PCT = 0.05       # 5 %

# =============================================================================
# ATR CONFIG
# =============================================================================

DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_TP_MULTIPLIER = 2.0
DEFAULT_ATR_SL_MULTIPLIER = 1.5

# SL floor: never let SL go below this fraction of entry price
ATR_SL_FLOOR_FACTOR = 0.5

# =============================================================================
# PIVOT CONFIG
# =============================================================================

DEFAULT_PIVOT_LOOKBACK_DAYS = 60
DEFAULT_PIVOT_TP_LEVEL = 'R1'
DEFAULT_PIVOT_SL_LEVEL = 'S1'

# Percentage fallbacks when all resistance / support levels are on wrong side
PIVOT_TP_FALLBACK_FACTOR = 1.05        # entry_price × this
PIVOT_SL_FALLBACK_FACTOR = 0.95        # entry_price × this

# =============================================================================
# FLAT CONFIG
# =============================================================================

DEFAULT_FLAT_TP_PCT = 0.05
DEFAULT_FLAT_SL_PCT = 0.05

# =============================================================================
# INDEX EXIT / REGIME FILTER
# =============================================================================

DEFAULT_REGIME_FILTER_ENABLED = False
DEFAULT_REGIME_MA_PERIOD = 20
DEFAULT_REGIME_EXIT_THRESHOLD = -0.02

# Volatility adjustment
DEFAULT_VOL_ADJUSTMENT_ENABLED = False
DEFAULT_VOL_LOOKBACK = 20
DEFAULT_HIGH_VOL_THRESHOLD = 0.25
DEFAULT_LOW_VOL_THRESHOLD = 0.15
DEFAULT_HIGH_VOL_MULTIPLIER = 1.5
DEFAULT_LOW_VOL_MULTIPLIER = 0.8

# =============================================================================
# ANALYTICS / REPORTING
# =============================================================================

VAR_CONFIDENCE_LEVEL = 0.05            # 95 % VaR → quantile(0.05)
DEFAULT_ROLLING_WINDOWS = [21, 63, 126, 252]   # 1M, 3M, 6M, 1Y
DEFAULT_CALENDAR_PERIODS = [1, 3, 6, 12]
DEFAULT_ROLLING_ALPHA_WINDOW = 63      # 3 months
DEFAULT_ROLLING_VOL_WINDOW = 21        # 1 month
DEFAULT_ROLLING_SHARPE_WINDOW = 126    # 6 months

ANNUALIZATION_FACTORS = {
    'daily': 252,
    'monthly': 12,
    'yearly': 1,
}

SCALE_FACTORS = {
    'daily_daily': 1,
    'daily_monthly': 21,
    'daily_yearly': 252,
    'monthly_monthly': 1,
    'monthly_yearly': 12,
    'yearly_yearly': 1,
}

# =============================================================================
# TUNING / OPTUNA
# =============================================================================

DEFAULT_OBJECTIVE = 'calmar'
DEFAULT_DIRECTION = 'maximize'
DEFAULT_FAILURE_PENALTY = -999.0
DEFAULT_FAILURE_PENALTY_MINIMIZE = 999.0   # Auto-derived for 'minimize' directions
DEFAULT_CALMAR_CAP = 10.0
DEFAULT_SAMPLER = 'TPE'
DEFAULT_MULTI_OBJ_SAMPLER = 'NSGA-II'
SAMPLER_SEED = 42

DEFAULT_TIERED_TP_THRESHOLDS = [0.05, 0.05, 0.05]
DEFAULT_TIERED_SL_THRESHOLDS = [0.05, 0.05, 0.05]
DEFAULT_INDEPENDENT_TPSL_MODES = False

# =============================================================================
# PATHS
# =============================================================================

DEFAULT_CONFIG_PATH = 'strategy_config.yaml'
DEFAULT_TUNING_CONFIG_PATH = 'tuning_config.yaml'
DEFAULT_OUTPUT_BASE_DIR = '../backtesting_results'

# =============================================================================
# MISC
# =============================================================================

DEFAULT_RUN_STOCK_SELECTION = True
DEFAULT_GENERATE_REPORT = True
