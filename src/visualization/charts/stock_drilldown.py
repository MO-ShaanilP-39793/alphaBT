"""Per-stock candlestick drill-down with TP/SL/entry/exit annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_BUFFER_DAYS_BEFORE = 5
_BUFFER_DAYS_AFTER = 10


def _resolve_tp_sl_prices(trade_row: pd.Series):
    """Return (tp_price, sl_price) preferring direct columns over pct derivation.

    get_portfolio --with-levels stores the actual TP/SL prices (more accurate
    for pivot/ATR modes).  The backtest report only has pct, so we fall back
    to ``entry * (1 +/- pct)`` when the price columns are absent or NaN.
    """
    entry_price = trade_row.get("entry_price")

    # Try direct price columns first
    tp_price = trade_row.get("tp_price")
    sl_price = trade_row.get("sl_price")

    if pd.notna(tp_price):
        tp_price = float(tp_price)
    else:
        tp_pct = trade_row.get("tp_pct_used", np.nan)
        tp_price = (entry_price * (1 + tp_pct)) if (pd.notna(tp_pct) and pd.notna(entry_price)) else None

    if pd.notna(sl_price):
        sl_price = float(sl_price)
    else:
        sl_pct = trade_row.get("sl_pct_used", np.nan)
        sl_price = (entry_price * (1 - sl_pct)) if (pd.notna(sl_pct) and pd.notna(entry_price)) else None

    return tp_price, sl_price


def stock_candlestick_chart(
    co_name: str,
    trade_row: pd.Series,
    price_data: pd.DataFrame,
    quarter_start: pd.Timestamp,
    quarter_end: pd.Timestamp,
) -> go.Figure:
    """Candlestick chart for a single stock with entry/TP/SL/exit annotations.

    Parameters
    ----------
    co_name : str
        Stock name.
    trade_row : Series
        Row from trade_results.  Accepts both backtest output
        (tp_pct_used/sl_pct_used) and get_portfolio output
        (tp_price/sl_price columns).
    price_data : DataFrame
        Full price data (date, co_name, open, high, low, close).
    quarter_start, quarter_end : Timestamps
        The quarter's nominal start and end dates.
    """
    entry_price = trade_row.get("entry_price")
    entry_date = pd.Timestamp(trade_row.get("entry_date"))
    exit_date = trade_row.get("exit_date")
    exit_price = trade_row.get("exit_price")
    tp_triggered = bool(trade_row.get("TP_triggered", False))
    sl_triggered = bool(trade_row.get("SL_triggered", False))

    tp_price, sl_price = _resolve_tp_sl_prices(trade_row)

    # Compute display-only pct (for annotation label)
    tp_pct_display = ((tp_price / entry_price) - 1) if (tp_price and pd.notna(entry_price) and entry_price > 0) else None
    sl_pct_display = (1 - (sl_price / entry_price)) if (sl_price and pd.notna(entry_price) and entry_price > 0) else None

    if pd.notna(exit_date):
        exit_date = pd.Timestamp(exit_date)

    # Filter price data for this stock with buffer
    stock_prices = price_data[price_data["co_name"] == co_name].copy()
    stock_prices = stock_prices.sort_values("date")

    range_start = quarter_start - pd.Timedelta(days=_BUFFER_DAYS_BEFORE + 5)
    range_end = quarter_end + pd.Timedelta(days=_BUFFER_DAYS_AFTER + 5)
    if pd.notna(exit_date):
        range_end = max(range_end, exit_date + pd.Timedelta(days=_BUFFER_DAYS_AFTER + 5))

    stock_prices = stock_prices[
        (stock_prices["date"] >= range_start) & (stock_prices["date"] <= range_end)
    ]

    if stock_prices.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No price data for {co_name}", height=480)
        return fig

    # --- Build candlestick ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.8, 0.2],
        subplot_titles=(co_name, "Volume"),
    )

    fig.add_trace(
        go.Candlestick(
            x=stock_prices["date"],
            open=stock_prices["open"],
            high=stock_prices["high"],
            low=stock_prices["low"],
            close=stock_prices["close"],
            name="OHLC",
            increasing_line_color="#2ca02c",
            decreasing_line_color="#d62728",
        ),
        row=1, col=1,
    )

    if "volume" in stock_prices.columns:
        fig.add_trace(
            go.Bar(
                x=stock_prices["date"], y=stock_prices["volume"],
                name="Volume", marker_color="rgba(100,100,100,0.3)",
                showlegend=False,
            ),
            row=2, col=1,
        )

    # --- Entry price line ---
    if pd.notna(entry_price):
        fig.add_hline(
            y=entry_price, line_dash="dash", line_color="#1f77b4",
            annotation_text=f"Entry ₹{entry_price:,.2f}",
            annotation_position="top left",
            row=1, col=1,
        )

    # --- TP level ---
    if tp_price is not None:
        pct_label = f" ({tp_pct_display*100:.1f}%)" if tp_pct_display is not None else ""
        fig.add_hline(
            y=tp_price, line_dash="dot", line_color="#2ca02c",
            annotation_text=f"TP ₹{tp_price:,.2f}{pct_label}",
            annotation_position="top right",
            row=1, col=1,
        )
        if pd.notna(entry_price):
            fig.add_hrect(
                y0=entry_price, y1=tp_price,
                fillcolor="rgba(44,160,44,0.07)", line_width=0,
                row=1, col=1,
            )

    # --- SL level ---
    if sl_price is not None:
        pct_label = f" ({sl_pct_display*100:.1f}%)" if sl_pct_display is not None else ""
        fig.add_hline(
            y=sl_price, line_dash="dot", line_color="#d62728",
            annotation_text=f"SL ₹{sl_price:,.2f}{pct_label}",
            annotation_position="bottom right",
            row=1, col=1,
        )
        if pd.notna(entry_price):
            fig.add_hrect(
                y0=sl_price, y1=entry_price,
                fillcolor="rgba(214,39,40,0.07)", line_width=0,
                row=1, col=1,
            )

    # --- Entry date marker ---
    if pd.notna(entry_date):
        fig.add_vline(
            x=entry_date.timestamp() * 1000,
            line_dash="dash", line_color="#1f77b4", opacity=0.6,
            row=1, col=1,
        )
        fig.add_annotation(
            x=entry_date, y=entry_price, text="ENTRY",
            showarrow=True, arrowhead=2, arrowcolor="#1f77b4",
            font=dict(color="#1f77b4", size=11),
            row=1, col=1,
        )

    # --- Exit date marker ---
    if pd.notna(exit_date) and pd.notna(exit_price):
        exit_colour = "#2ca02c" if tp_triggered else "#d62728" if sl_triggered else "#7f7f7f"
        exit_label = "TP HIT" if tp_triggered else "SL HIT" if sl_triggered else "TIME EXIT"

        fig.add_vline(
            x=exit_date.timestamp() * 1000,
            line_dash="dash", line_color=exit_colour, opacity=0.6,
            row=1, col=1,
        )
        fig.add_annotation(
            x=exit_date, y=exit_price,
            text=f"{exit_label}<br>₹{exit_price:,.2f}",
            showarrow=True, arrowhead=2, arrowcolor=exit_colour,
            font=dict(color=exit_colour, size=11),
            row=1, col=1,
        )

    # --- Quarter boundary shading ---
    fig.add_vrect(
        x0=quarter_start, x1=quarter_end,
        fillcolor="rgba(0,0,0,0.03)", line_width=0,
        row=1, col=1,
    )

    fig.update_layout(
        height=540,
        margin=dict(l=60, r=20, t=40, b=30),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)

    return fig


def stock_info_card(trade_row: pd.Series) -> dict:
    """Return a dict of key info for display beside the candlestick chart."""
    entry_price = trade_row.get("entry_price")
    tp_price, sl_price = _resolve_tp_sl_prices(trade_row)

    stock_return = trade_row.get("stock_return")

    exit_type = "TP" if trade_row.get("TP_triggered") else (
        "SL" if trade_row.get("SL_triggered") else (
            "Regime" if trade_row.get("regime_exit") else (
                "Open" if pd.isna(trade_row.get("exit_date")) else "Time"
            )
        )
    )

    sector = trade_row.get("sector")
    info = {
        "Stock": trade_row.get("co_name", "—"),
    }
    if pd.notna(sector) and sector:
        info["Sector"] = sector
    info["Category"] = trade_row.get("cat", "—")
    info.update({
        "Entry Date": _fmt_date(trade_row.get("entry_date")),
        "Entry Price": f"₹{entry_price:,.2f}" if pd.notna(entry_price) else "—",
        "TP Level": f"₹{tp_price:,.2f}" if tp_price else "—",
        "SL Level": f"₹{sl_price:,.2f}" if sl_price else "—",
        "Exit Date": _fmt_date(trade_row.get("exit_date")),
        "Exit Price": f"₹{trade_row.get('exit_price'):,.2f}" if pd.notna(trade_row.get("exit_price")) else "—",
        "Exit Type": exit_type,
        "Return": f"{stock_return * 100:.2f}%" if pd.notna(stock_return) else "—",
        "Holding": f"{int(trade_row.get('holding_period', 0))} days" if pd.notna(trade_row.get("holding_period")) else "—",
        "Weight": f"{trade_row.get('stock_weight', 0):.4f}" if pd.notna(trade_row.get("stock_weight")) else "—",
    })

    return info


def _fmt_date(d) -> str:
    if pd.isna(d) or d is None:
        return "—"
    return pd.Timestamp(d).strftime("%Y-%m-%d")
