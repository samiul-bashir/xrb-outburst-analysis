"""
Pure FRED (Fast Rise Exponential Decay) fitting (report section 5.1).

Fits xrb_pipeline.utils.models.simple_fred to each dominant peak found
by detection/simple_detector.py. Reports reduced chi-squared for every
fit so quality can be compared across sources/peaks at a glance.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from xrb_pipeline.detection.simple_detector import get_fit_window
from xrb_pipeline.utils.models import chi2_reduced, simple_fred


def fit_fred(name, peak_index, outburst_results, config):
    """Fit a single FRED profile to one detected peak. Returns a dict
    with params/chi2/data, or None on failure.
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

    best_idx = np.argmax(flux_window)
    a_guess, t_peak_guess = flux_window[best_idx], t_window[best_idx]
    p0 = [a_guess, t_peak_guess, 3.0, 15.0]
    lower = [0, t_window.min() - 0.01, 0.1, 0.1]
    upper = [a_guess * 3, t_window.max() + 0.01, 50, 200]

    try:
        params, _ = curve_fit(simple_fred, t_window, flux_window, p0=p0, bounds=(lower, upper), maxfev=10000)
    except RuntimeError as e:
        print(f"[ERROR] curve_fit failed: {e}")
        return None

    flux_fit = simple_fred(t_window, *params)
    chi2, dof, chi2_red = chi2_reduced(flux_window, flux_fit, n_params=4)
    return {
        "params": params, "chi2": chi2, "dof": dof, "chi2_red": chi2_red,
        "t": t_window, "flux": flux_window, "peak_mjd": peak_mjd,
    }


def fit_fred_all(name, outburst_results, config):
    """Fit every dominant peak of a source with a single FRED. Returns
    the list of successful fit dicts.
    """
    n_peaks = len(outburst_results[name]["dominant_peaks"])
    fits = []
    for i in range(n_peaks):
        fit = fit_fred(name, i, outburst_results, config)
        if fit is not None:
            fits.append(fit)
    print(f"Fit {len(fits)}/{n_peaks} peaks for {name}.")
    return fits


def fit_fred_for_all_sources(outburst_results, config):
    """Run fit_fred_all for every source with detected outbursts."""
    return {name: fit_fred_all(name, outburst_results, config) for name in outburst_results}


def plot_fit(fit, title=""):
    """Quick diagnostic plot: data + fitted FRED curve + chi2_red."""
    import matplotlib.pyplot as plt

    t_smooth = np.linspace(fit["t"].min(), fit["t"].max(), 300)
    flux_smooth = simple_fred(t_smooth, *fit["params"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fit["t"], fit["flux"], "o", ms=3, color="steelblue", alpha=0.6, label="data")
    ax.plot(t_smooth, flux_smooth, "-", color="red", lw=1.8, label="FRED fit")
    ax.set_title(f"{title}  Peak MJD {fit['peak_mjd']:.1f}  chi2_r={fit['chi2_red']:.2f}")
    ax.set_xlabel("MJD"); ax.set_ylabel("Flux")
    ax.legend()
    plt.tight_layout()
    plt.show()
