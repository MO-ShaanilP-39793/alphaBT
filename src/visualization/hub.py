"""
alphaBT — Dashboard Hub

Browse all run directories in a single page and click into any run's
full interactive dashboard.

Launch:
    streamlit run src/visualization/hub.py

By default scans ./deployed_dash for run directories.  Override with:
    streamlit run src/visualization/hub.py -- --dashboards-dir path/to/runs
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from visualization.data_loader import probe_run_dir
from visualization.dashboard import render_dashboard

# ── Page config (must be first Streamlit command) ─────────────────────────

st.set_page_config(
    page_title="alphaBT Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Arg parsing ───────────────────────────────────────────────────────────

_DEFAULT_DASHBOARDS_DIR = os.path.join(os.getcwd(), "deployed_dash")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="alphaBT Dashboard Hub")
    parser.add_argument(
        "--dashboards-dir",
        default=_DEFAULT_DASHBOARDS_DIR,
        help="Root directory to scan for run sub-directories (default: ./deployed_dash)",
    )
    args, _ = parser.parse_known_args()
    return args


# ── Directory scanning ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=30)
def _scan_runs(dashboards_dir: str) -> list[dict]:
    """Return metadata dicts for every valid run directory found."""
    dashboards_path = Path(dashboards_dir).resolve()
    if not dashboards_path.is_dir():
        return []

    results = []
    for entry in sorted(dashboards_path.iterdir()):
        if entry.is_dir():
            meta = probe_run_dir(str(entry))
            if meta is not None:
                results.append(meta)

    results.sort(key=lambda m: m["timestamp"], reverse=True)
    return results


# ── Hub page ──────────────────────────────────────────────────────────────

_TYPE_LABELS = {"backtest": "Backtest", "portfolio": "Portfolio"}


def _render_hub(dashboards_dir: str) -> None:
    st.title("alphaBT Dashboard Hub")
    st.caption(f"Scanning: `{dashboards_dir}`")

    runs = _scan_runs(dashboards_dir)

    if not runs:
        st.info(
            "No valid run directories found.  "
            "Place backtest or portfolio output folders inside "
            f"`{dashboards_dir}` and refresh.",
        )
        return

    st.markdown(f"**{len(runs)}** run{'s' if len(runs) != 1 else ''} available")
    st.markdown("---")

    cols_per_row = 3
    for row_start in range(0, len(runs), cols_per_row):
        row_runs = runs[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, meta in zip(cols, row_runs):
            with col:
                run_type_label = _TYPE_LABELS.get(meta["run_type"], meta["run_type"])

                try:
                    ts = datetime.fromisoformat(meta["timestamp"])
                    ts_display = ts.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    ts_display = meta["timestamp"]

                strategy_display = meta["strategy"] or "N/A"

                st.markdown(
                    f"### {meta['name']}\n"
                    f"**Type:** {run_type_label}  \n"
                    f"**Strategy:** {strategy_display}  \n"
                    f"**Modified:** {ts_display}"
                )

                st.link_button(
                    "Open Dashboard",
                    f"?run={meta['name']}",
                    use_container_width=True,
                )


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    dashboards_dir = str(Path(args.dashboards_dir).resolve())

    params = st.query_params
    selected_run = params.get("run")

    if selected_run:
        run_dir = os.path.join(dashboards_dir, selected_run)
        if not os.path.isdir(run_dir):
            st.error(f"Run directory not found: `{run_dir}`")
            st.link_button("Back to Hub", "?")
            return

        st.link_button("Back to Hub", "?")
        render_dashboard(run_dir)
    else:
        _render_hub(dashboards_dir)


main()
