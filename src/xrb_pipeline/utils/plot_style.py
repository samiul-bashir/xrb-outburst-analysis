"""
Optional shared plotting style: dark navy / off-white / muted
astronomical-blue palette, for anyone who wants pipeline figures to
match the README/report visual identity. Not applied automatically by
any pipeline module -- call `apply()` once at the top of a notebook or
script session to opt in.

    from xrb_pipeline.utils.plot_style import apply, PALETTE
    apply()

Existing plotting functions elsewhere in this package set their own
colors per-series (BH/NS red/blue, band colors for ZTF g/r/i, etc.) --
this module only restyles the shared chrome (figure/axes background,
grid, text, spines) so those semantic colors still read clearly on top
of it.
"""
from __future__ import annotations

PALETTE = {
    "navy": "#0B1622",
    "off_white": "#F4F1EA",
    "blue": "#5B8FB9",
    "amber": "#E0A458",
}


def apply(dark: bool = True) -> None:
    """Apply the project palette to matplotlib's rcParams for the rest
    of the session. Safe to call multiple times.

    Parameters
    ----------
    dark : bool
        True (default): dark-navy figure/axes background, off-white
        text/spines -- matches the README's dark theme. False: off-white
        background with navy text -- for figures meant to sit on a
        light page (e.g. exported into a light-themed document).
    """
    import matplotlib.pyplot as plt

    navy, off_white, blue = PALETTE["navy"], PALETTE["off_white"], PALETTE["blue"]
    bg, fg = (navy, off_white) if dark else (off_white, navy)

    plt.rcParams.update(
        {
            "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
            "axes.edgecolor": fg, "axes.labelcolor": fg,
            "xtick.color": fg, "ytick.color": fg, "text.color": fg,
            "axes.titlecolor": fg,
            "grid.color": blue, "grid.alpha": 0.25, "grid.linewidth": 0.5,
            "axes.grid": True,
            "axes.prop_cycle": plt.cycler(
                color=[blue, "#E63946", PALETTE["amber"], "#457B9D", "#2D6A4F", "#9D8DF1"]
            ),
            "legend.facecolor": bg, "legend.edgecolor": fg, "legend.labelcolor": fg,
        }
    )


def reset() -> None:
    """Restore matplotlib defaults."""
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)
