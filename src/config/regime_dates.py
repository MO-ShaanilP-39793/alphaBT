"""
Regime date loader.

Loads crisis and market regime definitions from ``config/regime_dates.yaml``
and exposes them as module-level dicts that match the legacy
``backtest.regime_dates`` API.

Usage::

    from config.regime_dates import CRISIS_REGIMES, MARKET_REGIMES
"""

import os
import warnings
from datetime import datetime
from typing import Dict, Any

import yaml

_YAML_PATH = os.path.join(os.path.dirname(__file__), 'regime_dates.yaml')


def _load_regime_dates(path: str = _YAML_PATH) -> dict:
    """Load regime dates from the YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def _parse_date(date_str: str) -> datetime:
    """Parse a DD-MM-YYYY date string."""
    return datetime.strptime(date_str, '%d-%m-%Y')


def check_regime_coverage(backtest_end_date: datetime) -> None:
    """Emit a warning if the latest market regime ends before *backtest_end_date*.

    Call this early in the pipeline so users know regime analysis will be
    incomplete for recent dates.
    """
    if not MARKET_REGIMES:
        return

    # Find the latest regime end date
    latest_end = max(
        _parse_date(regime['end'])
        for regime in MARKET_REGIMES.values()
    )

    if backtest_end_date > latest_end:
        warnings.warn(
            f"Market regime data ends at {latest_end:%d-%m-%Y} but backtest "
            f"runs until {backtest_end_date:%d-%m-%Y}. Regime analysis will "
            f"be incomplete for dates after {latest_end:%d-%m-%Y}. "
            f"Update config/regime_dates.yaml to add newer regimes.",
            stacklevel=2,
        )


# Module-level singletons — loaded once on first import
_data = _load_regime_dates()

CRISIS_REGIMES: Dict[str, Dict[str, str]] = _data.get('crisis_regimes', {})
MARKET_REGIMES: Dict[str, Dict[str, str]] = _data.get('market_regimes', {})
