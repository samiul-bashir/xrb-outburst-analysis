"""
Multi-FRED fitting: a sum of N simple_fred pulses, for outbursts with
multiple sub-peaks that a single FRED (or FRED+tophat) can't capture.
Not covered in the KSP-07 report, but included here as a natural
extension of the FRED family for complex/multi-peaked light curves.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from xrb_pipeline.detection.simple_detector import get_fit_window
from xrb_pipeline.utils.models import chi2_reduced


def multi_fred_model(t, *params):
    """Sum of N FRED pulses, 4 params each: [A, t_peak, tau_rise, tau_decay] x N."""
    t = np.asarray(t, dtype=float)
    n_components = len(params) // 4
    flux = np.zeros_like(t)
    for c in range(n_components):
        a, t_peak, tau_rise, tau_decay = params[4 * c : 4 * c + 4]
        flux += np.where(
            t < t_peak, a * np.exp(-(t_peak - t) / tau_rise), a * np.exp(-(t - t_peak) / tau_decay)
        )
    return flux


def fit_multi_fred(name, peak_index, outburst_results, config, n_components: int = 2):
    """Fit a sum of ``n_components`` FRED pulses to one detected peak's
    window, seeding component peak guesses from find_peaks (falling
    back to evenly spaced positions if fewer candidates are found than
    components requested).
    """
    result = outburst_results[name]
    window = get_fit_window(result, peak_index, config)
    if window is None:
        print(f"[ERROR] '{name}' peak_index={peak_index} out of range.")
        return None
    t_window, flux_window, peak_mjd = window
    if len(t_window) < 5 * n_components:
        print(f"[ERROR] Not enough points for {n_components} components.")
        return None

    candidate_idx, _ = find_peaks(flux_window)
    candidate_idx = np.unique(np.append(candidate_idx, np.argmax(flux_window)))
    candidate_sorted = candidate_idx[np.argsort(flux_window[candidate_idx])[::-1]]
    chosen_idx = list(candidate_sorted[:n_components])
    while len(chosen_idx) < n_components:
        filler = len(t_window) // (n_components + 1) * (len(chosen_idx) + 1)
        chosen_idx.append(filler)

    p0, lower, upper = [], [], []
    for idx in chosen_idx:
        a_guess, t_peak_guess = flux_window[idx], t_window[idx]
        p0.extend([a_guess, t_peak_guess, 3.0, 15.0])
        lower.extend([0, t_window.min() - 0.01, 0.1, 0.1])
        upper.extend([a_guess * 3, t_window.max() + 0.01, 50, 200])

    try:
        params, _ = curve_fit(
            multi_fred_model, t_window, flux_window, p0=p0, bounds=(lower, upper), maxfev=20000
        )
    except RuntimeError as e:
        print(f"[ERROR] curve_fit failed: {e}")
        return None

    flux_fit = multi_fred_model(t_window, *params)
    chi2, dof, chi2_red = chi2_reduced(flux_window, flux_fit, n_params=4 * n_components)
    return {
        "params": params, "chi2": chi2, "dof": dof, "chi2_red": chi2_red,
        "t": t_window, "flux": flux_window, "peak_mjd": peak_mjd, "n_components": n_components,
    }


def fit_multi_fred_all(name, outburst_results, config, n_components: int = 2):
    n_peaks = len(outburst_results[name]["dominant_peaks"])
    fits = []
    for i in range(n_peaks):
        fit = fit_multi_fred(name, i, outburst_results, config, n_components)
        if fit is not None:
            fits.append(fit)
    print(f"Fit {len(fits)}/{n_peaks} peaks for {name}.")
    return fits
