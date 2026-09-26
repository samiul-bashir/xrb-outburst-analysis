"""
FRED + Tophat fitting (report section 5.2): a flat plateau of width
``width`` inserted between the rise and decay exponentials. Setting
``width=0`` recovers the pure FRED; comparing chi2_red between the two
models tells you whether the plateau is statistically warranted.

Includes the bounds-clamping fix (from outburst_fred_tophat_fix1): near
the edges of a light curve the actual data window can be shorter than
the configured before/after-peak span, which without clamping trips
curve_fit's "initial guess is outside of provided bounds".
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from xrb_pipeline.detection.simple_detector import get_fit_window
from xrb_pipeline.utils.models import chi2_reduced, fred_tophat


def fit_fred_tophat(name, peak_index, outburst_results, config, width_guess: float = 10.0):
    """Fit FRED+tophat to one detected peak. Returns a dict with
    params/chi2/data, or None on failure.
    """
    result = outburst_results[name]
    window = get_fit_window(result, peak_index, config)
    if window is None:
        print(f"[ERROR] '{name}' peak_index={peak_index} out of range.")
        return None
    t_window, flux_window, peak_mjd = window
    if len(t_window) < 5:
        print(f"[ERROR] Not enough points ({len(t_window)}) in fit window.")
        return None

    t_min, t_max = t_window.min(), t_window.max()
    win_span = t_max - t_min

    # Clamp width guess and t_start guess into the actual data window --
    # a fixed config-derived guess can land outside data-derived bounds
    # near the edges of the light curve.
    width_guess = min(width_guess, win_span * 0.5)
    width_guess = max(width_guess, win_span * 1e-3)

    a0 = flux_window.max()
    if a0 <= 0:
        print("  [ERROR] Peak window has non-positive max flux, cannot fit.")
        return None

    t_start0 = peak_mjd - width_guess / 2
    t_start0 = min(max(t_start0, t_min), t_max)

    p0 = [a0, t_start0, width_guess, 3.0, 15.0]
    lower = [0, t_min, 0, 0.1, 0.1]
    upper = [a0 * 3, t_max, win_span, 50, 200]

    eps = 1e-6
    p0 = [min(max(g, lo + eps), up - eps) for g, lo, up in zip(p0, lower, upper)]

    try:
        params, _ = curve_fit(fred_tophat, t_window, flux_window, p0=p0, bounds=(lower, upper), maxfev=10000)
    except RuntimeError as e:
        print(f"[ERROR] curve_fit failed: {e}")
        return None

    flux_fit = fred_tophat(t_window, *params)
    chi2, dof, chi2_red = chi2_reduced(flux_window, flux_fit, n_params=5)
    return {
        "params": params, "chi2": chi2, "dof": dof, "chi2_red": chi2_red,
        "t": t_window, "flux": flux_window, "peak_mjd": peak_mjd,
    }


def fit_fred_tophat_all(name, outburst_results, config):
    """Fit every dominant peak of a source with FRED+tophat."""
    n_peaks = len(outburst_results[name]["dominant_peaks"])
    fits = []
    for i in range(n_peaks):
        fit = fit_fred_tophat(name, i, outburst_results, config)
        if fit is not None:
            fits.append(fit)
    print(f"Fit {len(fits)}/{n_peaks} peaks for {name}.")
    for fit in fits:
        a, t_start, width, tau_rise, tau_decay = fit["params"]
        print(
            f"  peak MJD {fit['peak_mjd']:.1f}: A={a:.4f} t_start={t_start:.2f} "
            f"width={width:.2f} tau_rise={tau_rise:.2f} tau_decay={tau_decay:.2f} "
            f"chi2_red={fit['chi2_red']:.3f}"
        )
    return fits


def fit_fred_tophat_for_all_sources(outburst_results, config):
    return {name: fit_fred_tophat_all(name, outburst_results, config) for name in outburst_results}


def save_fits_to_disk(all_source_fits: dict, outdir: str = "outburst_fits") -> pd.DataFrame:
    """Save each fitted outburst window to disk as
    ``<source>_<index>.csv`` (mjd, flux, flux_fit), plus a
    ``fit_summary.csv`` with fitted params + chi2 for every peak.
    """
    os.makedirs(outdir, exist_ok=True)
    summary_rows = []

    for name, fits in all_source_fits.items():
        safe_name = name.replace(" ", "_").replace("/", "-")
        for i, fit in enumerate(fits):
            t, flux = fit["t"], fit["flux"]
            flux_fit = fred_tophat(t, *fit["params"])
            fname = f"{safe_name}_{i}.csv"
            pd.DataFrame({"mjd": t, "flux": flux, "flux_fit": flux_fit}).to_csv(
                os.path.join(outdir, fname), index=False
            )
            a, t_start, width, tau_rise, tau_decay = fit["params"]
            summary_rows.append(
                {
                    "source": name, "index": i, "file": fname, "peak_mjd": fit["peak_mjd"],
                    "A": a, "t_start": t_start, "width": width, "tau_rise": tau_rise, "tau_decay": tau_decay,
                    "chi2": fit["chi2"], "dof": fit["dof"], "chi2_red": fit["chi2_red"],
                }
            )
        print(f"Saved {len(fits)} outburst file(s) for {name}.")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(outdir, "fit_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary of {len(summary_df)} peaks written to {summary_path}")
    return summary_df


def plot_fit(fit, title=""):
    """Quick diagnostic plot: data + fitted FRED+tophat curve + chi2_red."""
    import matplotlib.pyplot as plt

    t_smooth = np.linspace(fit["t"].min(), fit["t"].max(), 300)
    flux_smooth = fred_tophat(t_smooth, *fit["params"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fit["t"], fit["flux"], "o", ms=3, color="steelblue", alpha=0.6, label="data")
    ax.plot(t_smooth, flux_smooth, "-", color="red", lw=1.8, label="FRED+tophat fit")
    ax.set_title(f"{title}  Peak MJD {fit['peak_mjd']:.1f}  chi2_r={fit['chi2_red']:.3f}")
    ax.set_xlabel("MJD"); ax.set_ylabel("Flux")
    ax.legend()
    plt.tight_layout()
    plt.show()
