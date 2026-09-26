"""
Asymmetric burst-profile fitting -- an alternative to the FRED family.

Independent rise/decay amplitudes plus a free background level, adapted
from a LAXPC Type-I X-ray burst fitting prototype (see
docs/legacy_notebooks.md) and applied here to MAXI outburst
peaks. Not part of the KSP-07 report's final results, but kept as a
documented alternative model -- useful when a source's rise and decay
have genuinely different characteristic amplitudes rather than sharing
one peak value (see xrb_pipeline.utils.models.asymmetric_burst_profile).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from xrb_pipeline.detection.simple_detector import get_fit_window
from xrb_pipeline.utils.models import asymmetric_burst_profile, chi2_reduced


def fit_burst_profile(name, peak_index, outburst_results, config):
    """Fit the asymmetric burst profile to one detected peak."""
    result = outburst_results[name]
    window = get_fit_window(result, peak_index, config)
    if window is None:
        print(f"[ERROR] '{name}' peak_index={peak_index} out of range.")
        return None
    t_window, flux_window, peak_mjd = window
    if len(t_window) < 6:
        print(f"[ERROR] Not enough points ({len(t_window)}) in fit window.")
        return None

    best_idx = np.argmax(flux_window)
    amp_guess = flux_window[best_idx]
    bg_guess = np.percentile(flux_window, 10)
    p0 = [t_window[best_idx], 3.0, 15.0, amp_guess, amp_guess, bg_guess]
    lower = [t_window.min(), 0.1, 0.1, 0, 0, -np.inf]
    upper = [t_window.max(), 50, 200, amp_guess * 3, amp_guess * 3, np.inf]

    try:
        params, _ = curve_fit(
            asymmetric_burst_profile, t_window, flux_window, p0=p0, bounds=(lower, upper), maxfev=10000
        )
    except RuntimeError as e:
        print(f"[ERROR] curve_fit failed: {e}")
        return None

    flux_fit = asymmetric_burst_profile(t_window, *params)
    chi2, dof, chi2_red = chi2_reduced(flux_window, flux_fit, n_params=6)
    return {
        "params": params, "chi2": chi2, "dof": dof, "chi2_red": chi2_red,
        "t": t_window, "flux": flux_window, "peak_mjd": peak_mjd,
    }


def fit_burst_profile_all(name, outburst_results, config):
    """Fit every dominant peak of a source with the asymmetric burst
    profile.
    """
    n_peaks = len(outburst_results[name]["dominant_peaks"])
    fits = []
    for i in range(n_peaks):
        fit = fit_burst_profile(name, i, outburst_results, config)
        if fit is not None:
            fits.append(fit)
    print(f"Fit {len(fits)}/{n_peaks} peaks for {name}.")
    return fits


def fit_burst_profile_for_all_sources(outburst_results, config):
    return {name: fit_burst_profile_all(name, outburst_results, config) for name in outburst_results}
