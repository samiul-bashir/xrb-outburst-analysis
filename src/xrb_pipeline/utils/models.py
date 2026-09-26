"""
Outburst light-curve profile models used across the project.

Four variants appear in the source notebooks; all are kept because each
serves a different stage of the analysis (see docs/algorithms.md):

- ``simple_fred``      : two-sided exponential, symmetric-peak formulation.
                         Used by the Pure-FRED detection+fit pipeline
                         (report section 5.1).
- ``fred_tophat``      : simple_fred with a flat plateau inserted between
                         rise and decay (report section 5.2).
- ``asymmetric_burst_profile`` : independent rise/decay amplitudes plus a
                         free background level; adapted from a LAXPC
                         Type-I burst fitting prototype (see
                         docs/legacy_burst_lc_fitting.md).
- ``norris_fred``      : the smooth (no-kink) Norris-style FRED used
                         throughout the manual inspector / reprocessing
                         phases (report section 8), where ``amplitude`` is
                         the *exact* peak value by construction and
                         ``t_peak`` is derived rather than fitted.
"""
from __future__ import annotations

import numpy as np


def simple_fred(t, A, t_peak, tau_rise, tau_decay):
    """Two-sided exponential FRED, peak-referenced (report eq. 7).

    F(t) = A * exp(-(t_peak - t) / tau_rise)   for t < t_peak
    F(t) = A * exp(-(t - t_peak) / tau_decay)  for t >= t_peak
    """
    t = np.asarray(t, dtype=float)
    return np.where(
        t < t_peak,
        A * np.exp(-(t_peak - t) / tau_rise),
        A * np.exp(-(t - t_peak) / tau_decay),
    )


def fred_tophat(t, A, t_start, width, tau_rise, tau_decay):
    """FRED with a flat top: rise into t_start, plateau until t_start+width,
    then decay (report eq. 9). Setting ``width=0`` recovers a
    start-referenced pure FRED.
    """
    t = np.asarray(t, dtype=float)
    t_end = t_start + width
    flux = np.zeros_like(t)
    rise = t < t_start
    plateau = (t >= t_start) & (t <= t_end)
    decay = t > t_end
    flux[rise] = A * np.exp(-(t_start - t[rise]) / tau_rise)
    flux[plateau] = A
    flux[decay] = A * np.exp(-(t[decay] - t_end) / tau_decay)
    return flux


def asymmetric_burst_profile(t, t_peak, tau_rise, tau_decay, amp_rise, amp_decay, background):
    """Asymmetric exponential burst profile with independent rise/decay
    amplitudes and a free background level. Adapted from the LAXPC
    Type-I X-ray burst prototype (``burst-lc-fitting.ipynb``); applied
    here to MAXI outburst peaks as an alternative to the FRED family.
    """
    t = np.asarray(t, dtype=float)
    exponent = np.where(t > t_peak, -(t - t_peak) / tau_decay, -(t_peak - t) / tau_rise)
    amplitude = np.where(t > t_peak, amp_decay, amp_rise)
    return amplitude * np.exp(exponent) + background


def norris_fred(t, A, t_start, t_rise, t_decay):
    """Norris FRED (mentor's formulation; report eq. 28).

    F(t) = A * exp(2*sqrt(t_rise/t_decay))
             * exp(-t_rise/(t - t_start) - (t - t_start)/t_decay),   t > t_start
    F(t) = 0,                                                        t <= t_start

    Smooth everywhere (no kink at the peak). Peak time is *derived*, not
    fitted: ``t_peak = t_start + sqrt(t_rise * t_decay)``, and
    ``F(t_peak) == A`` exactly, by construction.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    model = np.zeros_like(t)
    active = t > t_start
    dt = t[active] - t_start
    norm = A * np.exp(2.0 * np.sqrt(t_rise / t_decay))
    model[active] = norm * np.exp(-t_rise / dt - dt / t_decay)
    return model


def norris_fred_peak_time(t_start, t_rise, t_decay):
    """t_peak = t_start + sqrt(t_rise * t_decay) -- derived, not fitted."""
    return t_start + np.sqrt(t_rise * t_decay)


def make_norris_fred_smooth(outburst: dict, maxi_conv: float):
    """Build a closure ``fred_smooth_erg(t)`` evaluating the stored Norris
    FRED fit for one outburst, converting the stored count-rate amplitude
    to erg/cm^2/s via ``maxi_conv``.

    ``outburst`` must have keys ``amplitude, t_start, t_rise, t_decay``
    (as saved to the JSON database by the manual inspector, Phase 0).

    Note: earlier notebook versions passed the raw count-rate amplitude
    through unconverted; this was fixed from Phase 3-4 onward and is kept
    fixed here.
    """
    a_erg = outburst["amplitude"] * maxi_conv
    t_start = outburst["t_start"]
    t_rise = outburst["t_rise"]
    t_decay = outburst["t_decay"]

    def fred_smooth_erg(t):
        return norris_fred(t, A=a_erg, t_start=t_start, t_rise=t_rise, t_decay=t_decay)

    return fred_smooth_erg


def chi2_reduced(y_obs, y_model, n_params, epsilon=1e-10):
    """Reduced chi-squared with an epsilon guard against zero-flux bins
    (report eq. 10). Returns (chi2, dof, chi2_red).
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    chi2 = np.sum((y_obs - y_model) ** 2 / (np.abs(y_obs) + epsilon))
    dof = max(len(y_obs) - n_params, 1)
    return chi2, dof, chi2 / dof
