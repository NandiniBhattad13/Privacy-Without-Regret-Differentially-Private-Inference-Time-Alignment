"""
Shared matplotlib configuration for publication-quality figures.

Usage in any experiment script:

    from plot_utils import configure_matplotlib
    configure_matplotlib()          # grid on (default)
    configure_matplotlib(grid=False)  # grid off (e.g. for bar charts)
"""

import matplotlib as mpl


def configure_matplotlib(grid: bool = True) -> None:
    """Apply publication-quality rcParams.

    Args:
        grid: Whether to enable the background grid. Default True.
              Pass False for bar-chart figures where a grid would be distracting.
    """
    mpl.rcParams.update({
        # ---- Font sizes ----
        "font.size":          14,
        "axes.titlesize":     16,
        "axes.labelsize":     15,
        "xtick.labelsize":    13,
        "ytick.labelsize":    13,
        "legend.fontsize":    11,
        "figure.titlesize":   18,

        # ---- Serif font (LaTeX-paper compatible) ----
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset":   "cm",

        # ---- Line / marker weights ----
        "axes.linewidth":     1.2,
        "lines.linewidth":    2.5,
        "lines.markersize":   8,

        # ---- Grid ----
        "axes.grid":          grid,
        "grid.alpha":         0.4,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.6,

        # ---- Ticks ----
        "xtick.major.size":   5,
        "ytick.major.size":   5,
        "xtick.major.width":  1.0,
        "ytick.major.width":  1.0,
        "xtick.direction":    "in",
        "ytick.direction":    "in",

        # ---- Save defaults ----
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype":       42,   # embed fonts as TrueType (required by most venues)
        "ps.fonttype":        42,

        # ---- Legend ----
        "legend.frameon":     True,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "black",
        "legend.fancybox":    False,
    })
