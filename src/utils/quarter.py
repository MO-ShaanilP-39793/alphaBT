"""
Quarter-date mapping utilities.

Canonical source of truth for converting YYYYMM quarter identifiers to
their corresponding entry-start and mandatory-exit dates.

Quarter conventions (entry/exit inclusive):
    202402 → Feb 15 – May 30
    202405 → May 31 – Aug 14
    202408 → Aug 15 – Nov 14
    202411 → Nov 15 – Feb 14 (next year)
"""

import pandas as pd


def get_quarter_dates(quarter):
    """
    Convert a YYYYMM quarter identifier to (entry_start, mandatory_exit) timestamps.

    Parameters:
        quarter: Quarter in YYYYMM format (int or str), e.g., 202402

    Returns:
        tuple[pd.Timestamp, pd.Timestamp]: (entry_search_start, mandatory_exit_date)

    Raises:
        ValueError: If the month component is not one of {2, 5, 8, 11}.
    """
    quarter_str = str(quarter)
    year = int(quarter_str[:4])
    mm = int(quarter_str[4:])

    if mm == 2:   # Feb 15 – May 30
        start = pd.Timestamp(year=year, month=2, day=15)
        end = pd.Timestamp(year=year, month=5, day=30)
    elif mm == 5:  # May 31 – Aug 14
        start = pd.Timestamp(year=year, month=5, day=31)
        end = pd.Timestamp(year=year, month=8, day=14)
    elif mm == 8:  # Aug 15 – Nov 14
        start = pd.Timestamp(year=year, month=8, day=15)
        end = pd.Timestamp(year=year, month=11, day=14)
    elif mm == 11:  # Nov 15 – Feb 14 (next year)
        start = pd.Timestamp(year=year, month=11, day=15)
        end = pd.Timestamp(year=year + 1, month=2, day=14)
    else:
        raise ValueError(
            f"Invalid quarter month: {mm}. Expected one of {{2, 5, 8, 11}}."
        )

    return start, end


def get_entry_start_date(quarter):
    """
    Get just the entry search start date for a quarter.

    Convenience wrapper around ``get_quarter_dates`` that returns only the
    first element (entry_search_start).

    Parameters:
        quarter: Quarter in YYYYMM format (int or str), e.g., 202402

    Returns:
        pd.Timestamp: The entry search start date.
    """
    start, _end = get_quarter_dates(quarter)
    return start
