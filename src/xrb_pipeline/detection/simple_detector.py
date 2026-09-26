"""
Simple threshold-based outburst detector (report section 4.1 / 5.1 Stage 2).

A single global sigma-clipped threshold per source, consecutive-day
grouping with a small gap tolerance, and duplicate-peak suppression
within a fixed MJD window. This is the detector shared by the FRED,
FRED+tophat and asymmetric-burst-profile fitting pipelines (see
src/xrb_pipeline/fitting/). For the more sophisticated class-routed
detector used to build the main outburst catalogue and duration
statistics, see detection/tiered_detector.py (report section 4.2).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from xrb_pipeline.utils.fetch import fetch_maxi_lightcurve

DEFAULT_CONFIG = {
    "SIGMA_MULT": 5.0,        # threshold = baseline + SIGMA_MULT * noise_std
    "MIN_DURATION_DAYS": 25,  # drop outburst groups shorter than this
    "GAP_TOLERANCE": 3,       # bridge gaps <= this many days
    "PEAK_WINDOW_MJD": 10,    # suppress duplicate peaks within this window
    "DAYS_BEFORE_PEAK": 20,   # fit window: days before peak
    "DAYS_AFTER_PEAK": 50,    # fit window: days after peak
}


def compute_threshold(flux, sigma_mult):
    """Baseline = median; noise estimated from the bulk (below the 90th
    percentile) so outburst points don't inflate the noise estimate.
    """
    baseline = np.nanmedian(flux)
    bulk_mask = flux < np.nanpercentile(flux, 90)
    noise_std = np.nanstd(flux[bulk_mask])
    threshold = baseline + sigma_mult * noise_std
    return baseline, noise_std, threshold


def find_outburst_groups(mjd, flux, threshold, gap_tol, min_dur):
    """Group consecutive above-threshold days into outburst episodes,
    bridging gaps <= gap_tol days and dropping groups shorter than
    min_dur days. Returns a list of index-lists.
    """
    above = flux > threshold
    groups, current, gap_count = [], [], 0

    for i in range(len(mjd)):
        if above[i]:
            current.append(i)
            gap_count = 0
        elif current:
            gap_count += 1
            if gap_count <= gap_tol:
                current.append(i)
            else:
                groups.append(current)
                current, gap_count = [], 0
    if current:
        groups.append(current)

    return [g for g in groups if (mjd[g[-1]] - mjd[g[0]]) >= min_dur]


def find_dominant_peaks(mjd, flux, group_indices, window_mjd):
    """Within one outburst group, run find_peaks then suppress any peak
    within +/- window_mjd of a taller one. Returns global indices.
    """
    seg_flux = flux[group_indices]
    local_peaks, _ = find_peaks(seg_flux)
    if len(local_peaks) == 0:
        local_peaks = [np.argmax(seg_flux)]

    global_peaks = [group_indices[p] for p in local_peaks]
    global_sorted = sorted(global_peaks, key=lambda i: flux[i], reverse=True)

    kept, suppressed = [], set()
    for pk in global_sorted:
        if pk in suppressed:
            continue
        kept.append(pk)
        for other in global_sorted:
            if other != pk and abs(mjd[other] - mjd[pk]) <= window_mjd:
                suppressed.add(other)
    return kept


def detect_outbursts(maxi_id: str, config: dict = DEFAULT_CONFIG):
    """Run fetch -> threshold -> group -> peak-find for one source.
    Returns a result dict, or None if the source is skipped (no data or
    no outburst survives the duration filter).
    """
    mjd, flux = fetch_maxi_lightcurve(maxi_id)
    if len(mjd) == 0:
        return None

    baseline, noise_std, threshold = compute_threshold(flux, config["SIGMA_MULT"])
    groups = find_outburst_groups(
        mjd, flux, threshold, gap_tol=config["GAP_TOLERANCE"], min_dur=config["MIN_DURATION_DAYS"]
    )
    if not groups:
        return None

    all_dominant_peaks = []
    for g in groups:
        all_dominant_peaks.extend(find_dominant_peaks(mjd, flux, g, window_mjd=config["PEAK_WINDOW_MJD"]))

    return {
        "mjd": mjd, "flux": flux,
        "baseline": baseline, "noise_std": noise_std, "threshold": threshold,
        "groups": groups, "dominant_peaks": all_dominant_peaks,
    }


def get_fit_window(result, peak_index, config):
    """Shared by the fitting modules: slice out the data window around
    one detected peak.
    """
    mjd, flux, peaks = result["mjd"], result["flux"], result["dominant_peaks"]
    if peak_index >= len(peaks):
        return None
    peak_mjd = mjd[peaks[peak_index]]
    mask = (mjd >= peak_mjd - config["DAYS_BEFORE_PEAK"]) & (mjd <= peak_mjd + config["DAYS_AFTER_PEAK"])
    return mjd[mask], flux[mask], peak_mjd


def run_detection_all(source_ids: list[str], config: dict = DEFAULT_CONFIG) -> dict:
    """Run detect_outbursts for every source id, collecting results into
    a dict keyed by source id (consumed by the fitting modules).
    """
    outburst_results = {}
    for maxi_id in source_ids:
        print(f"\n{'=' * 55}\n  {maxi_id}\n{'=' * 55}")
        result = detect_outbursts(maxi_id, config)
        if result is None:
            print("  [INFO] No outbursts detected.")
            continue
        outburst_results[maxi_id] = result
        print(f"  {len(result['groups'])} group(s), {len(result['dominant_peaks'])} peak(s)")
    return outburst_results
