"""
Shared formatting utilities for chart axis labels and value display.
"""


def crores_formatter(x, pos):
    """Matplotlib tick formatter that displays values in crores (1 Cr = 1e7).

    Usage::

        from utils.formatting import crores_formatter
        ax.yaxis.set_major_formatter(plt.FuncFormatter(crores_formatter))
    """
    return f'{x / 1e7:.1f} Cr'
