"""
Class-routed automated outburst detector (report section 4.2).

Routes every source to one of four detection configurations by
SIMBAD otype + compact-object class (BH LXB / NS LXB / unknown LXB /
HXB), each with its own baseline method, sigma threshold, duration and
peak-separation requirements. This is the detector behind the main
outburst catalogue (step2_outbursts.csv) and, downstream, the T90/rise/
decay duration statistics (see duration_stats/).

Key differences from detection/simple_detector.py:
    - iterative_baseline(): sigma-clipped, unbiased for sources with
      many or long outbursts (a naive percentile is upward-biased by
      outburst-contaminated data).
    - find_peaks() with a prominence + physically-motivated minimum-
      separation constraint, instead of simple threshold-crossing
      grouping -- prevents a single outburst with a secondary maximum
      from being split into two spurious events.
    - HXB sources use a 300-day rolling-median local baseline (they
      never return to a true quiescent floor) instead of a global one.
    - An HR (hardness-ratio) spectral veto for NS LXBs: a genuine
      outburst must soften (HR drops) near the peak relative to
      quiescence.

Pipeline: python -m xrb_pipeline.detection.tiered_detector \\
              --input step1b_maxi_xrb_classified.csv --outdir data/detection
"""
from __future__ import annotations

import string
import time

import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks, savgol_filter

MAXI_BASE = "https://maxi.riken.jp/star_data"
HEADERS = {"User-Agent": "Mozilla/5.0"}
EPSILON = 1e-6

# Any source whose peak MAXI flux never exceeds this is treated as
# undetected in the MAXI era and skipped entirely.
TIER0_MIN_PEAK_FLUX = 0.04  # ph/s/cm^2 (2-20 keV all-band)

_SAVGOL_WIN = 7   # days -- must be odd; smoothing is for peak-finding only
_SAVGOL_ORDER = 2
_LOW_SIGMA = 1    # boundary walk: stop when flux < baseline + LOW_SIGMA*sigma

# ---------------------------------------------------------------------
# Per-class detection configurations (see docs/algorithms.md Table 6)
# ---------------------------------------------------------------------

BH_LXB_CONFIG = {
    "type": "LXB_BH", "QUIET_PERCENTILE": 15, "SIGMA_MULT": 5.0,
    "MIN_CONTIGUOUS_DAYS": 10, "MIN_DURATION_DAYS": 35, "GAP_TOLERANCE": 5,
    "MIN_PEAK_FLUX": 0.05, "MIN_DISTANCE_DAYS": 120, "HR_VETO": False,
    "MERGE_GAP_DAYS": 80, "MAX_WING_DAYS": 200, "WALK_GAP_TOLERANCE": 5,
}

NS_LXB_CONFIG = {
    "type": "LXB_NS", "QUIET_PERCENTILE": 15, "SIGMA_MULT": 5.0,
    "MIN_CONTIGUOUS_DAYS": 10, "MIN_DURATION_DAYS": 25, "GAP_TOLERANCE": 3,
    "MIN_PEAK_FLUX": 0.08, "MIN_DISTANCE_DAYS": 80, "HR_VETO": True,
    "HR_SMOOTHING_WIN": 20, "HR_NEAR_PEAK_FRAC": 0.5, "HR_SOFTENING": 0.75,
    "HR_MAX_QUIESCENT": 0.55, "MERGE_GAP_DAYS": 40, "MAX_WING_DAYS": 120,
    "WALK_GAP_TOLERANCE": 7,
}

UNK_LXB_CONFIG = {
    "type": "LXB_UNK", "QUIET_PERCENTILE": 15, "SIGMA_MULT": 5.5,
    "MIN_CONTIGUOUS_DAYS": 5, "MIN_DURATION_DAYS": 30, "GAP_TOLERANCE": 3,
    "MIN_PEAK_FLUX": 0.05, "MIN_DISTANCE_DAYS": 80, "HR_VETO": False,
    "MERGE_GAP_DAYS": 50, "MAX_WING_DAYS": 150, "WALK_GAP_TOLERANCE": 10,
}

HXB_CONFIG = {
    "type": "HXB", "ROLLING_WINDOW": 300, "SIGMA_MULT": 5.0,
    "MIN_CONTIGUOUS_DAYS": 10, "MIN_DURATION_DAYS": 25, "MAX_DURATION_DAYS": 200,
    "GAP_TOLERANCE": 3, "MIN_PEAK_FLUX": 0.10, "MIN_DISTANCE_DAYS": 40,
    "HR_VETO": False, "MERGE_GAP_DAYS": 15, "MAX_WING_DAYS": 80,
    "WALK_GAP_TOLERANCE": 5,
}


def ra_dec_to_maxi_id(ra_deg, dec_deg):
    """RA/Dec degrees -> MAXI folder id (JHHMM+-DDd)."""
    ra_hours = ra_deg / 15.0
    hh = int(ra_hours)
    mm = int((ra_hours - hh) * 60)
    sign = "+" if dec_deg >= 0 else "-"
    dd = int(abs(dec_deg))
    d = int((abs(dec_deg) - dd) * 10)
    return f"J{hh:02d}{mm:02d}{sign}{dd:02d}{d}"


def make_source_id(n):
    """0->a, 1->b, ... 25->z, 26->aa ..."""
    letters = string.ascii_lowercase
    return letters[n] if n < 26 else letters[n // 26 - 1] + letters[n % 26]


def fetch_maxi_lightcurve(maxi_id):
    """Fetch MAXI 1-day all-band LC as a DataFrame(mjd, flux_total,
    flux_soft, flux_hard), or None on failure. MAXI column order in
    ``_g_lc_1day_all.dat``: 0:MJD 1:flux_total 2:err 3:flux_soft 4:err
    5:flux_med 6:err 7:flux_hard 8:err.
    """
    url = f"{MAXI_BASE}/{maxi_id}/{maxi_id}_g_lc_1day_all.dat"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
    except Exception as e:
        print(f"    Network error: {e}")
        return None
    if resp.status_code != 200:
        print(f"    HTTP {resp.status_code}")
        return None

    lines = [l for l in resp.text.strip().split("\n") if l.strip() and not l.startswith("!")]
    rows = []
    for l in lines:
        parts = l.split()
        if len(parts) < 8:
            continue
        try:
            rows.append(
                {
                    "mjd": float(parts[0]),
                    "flux_total": float(parts[1]),
                    "flux_soft": float(parts[3]),
                    "flux_hard": float(parts[7]),
                }
            )
        except ValueError:
            continue
    if len(rows) < 10:
        return None
    return pd.DataFrame(rows).sort_values("mjd").reset_index(drop=True)


def tier0_check(lc_df):
    """Returns (passes, peak_flux, noise_rms) against TIER0_MIN_PEAK_FLUX."""
    flux = lc_df["flux_total"].values
    peak = np.nanmax(flux)
    below_med = flux[flux < np.nanmedian(flux)]
    noise_rms = np.nanstd(below_med) if len(below_med) > 0 else np.nan
    return peak >= TIER0_MIN_PEAK_FLUX, peak, noise_rms


def iterative_baseline(flux, percentile=15, sigma_clip=3.0, max_iter=10):
    """Sigma-clip outburst flux iteratively before estimating the quiet
    baseline -- unbiased for sources with many/long outbursts, where a
    raw global percentile sits inside an outburst tail.

    Returns (baseline, sigma_r) where sigma_r = 1.4826 * MAD (robust
    Gaussian-equivalent sigma of the quiet flux).
    """
    mask = np.ones(len(flux), dtype=bool)
    prev_mask = None
    for _ in range(max_iter):
        if mask.sum() == 0:
            break
        med = np.nanmedian(flux[mask])
        mad = np.nanmedian(np.abs(flux[mask] - med))
        sig = max(1.4826 * mad, 1e-6)
        mask = np.abs(flux - med) < sigma_clip * sig
        if prev_mask is not None and np.array_equal(mask, prev_mask):
            break
        prev_mask = mask.copy()

    quiet_flux = flux[mask] if mask.sum() > 10 else flux
    baseline = np.nanpercentile(quiet_flux, percentile)
    mad_f = np.nanmedian(np.abs(quiet_flux - baseline))
    sigma_r = max(1.4826 * mad_f, 1e-6)
    return baseline, sigma_r


def _find_outburst_peaks(flux, mjd, baseline, sigma_r, config, thresh_arr=None):
    """Core peak finder shared by the LXB and HXB detectors: find_peaks
    with a prominence + minimum-separation constraint, then a
    gap-tolerant boundary walk down to a low threshold, then a
    post-detection merge of adjacent groups.
    """
    scalar_thresh = baseline + config["SIGMA_MULT"] * sigma_r
    low_thr_scalar = baseline + _LOW_SIGMA * sigma_r

    cadence = float(np.nanmedian(np.diff(mjd))) or 1.0

    win = _SAVGOL_WIN
    if len(flux) < win:
        win = len(flux) if len(flux) % 2 == 1 else max(3, len(flux) - 1)
    win = max(win, 3)
    smooth = (
        savgol_filter(flux, window_length=win, polyorder=_SAVGOL_ORDER)
        if len(flux) >= win
        else flux.copy()
    )

    min_dist_pts = max(1, int(config.get("MIN_DISTANCE_DAYS", 30) / cadence))
    min_width_pts = max(1, int(config["MIN_CONTIGUOUS_DAYS"] / cadence))

    peaks, props = find_peaks(
        smooth,
        height=scalar_thresh,
        prominence=config["SIGMA_MULT"] * sigma_r,
        distance=min_dist_pts,
        width=min_width_pts,
    )
    if len(peaks) == 0:
        return []

    n = len(flux)
    low_thr_arr = thresh_arr if thresh_arr is not None else np.full(n, low_thr_scalar)

    max_wing_pts = int(config.get("MAX_WING_DAYS", 300) / cadence)

    outbursts = []
    for i, pk in enumerate(peaks):
        left = int(props["left_ips"][i])
        left_cap = max(0, pk - max_wing_pts)
        right_cap = min(n - 1, pk + max_wing_pts)

        while left > left_cap and flux[left] > low_thr_arr[left]:
            left -= 1

        walk_gap = config.get("WALK_GAP_TOLERANCE", 10)
        gap_count = 0
        r = int(props["right_ips"][i])
        best_right = r
        while r < right_cap:
            if flux[r] > low_thr_arr[r]:
                best_right = r
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > walk_gap:
                    break
            r += 1
        right = best_right

        if i + 1 < len(peaks):
            next_pk_idx = peaks[i + 1]
            if right >= next_pk_idx:
                right = next_pk_idx - 1

        duration = mjd[right] - mjd[left]
        if duration < config["MIN_DURATION_DAYS"]:
            continue
        if "MAX_DURATION_DAYS" in config and duration > config["MAX_DURATION_DAYS"]:
            continue
        if flux[pk] < config["MIN_PEAK_FLUX"]:
            continue

        outbursts.append(
            {
                "idx_left": left, "idx_peak": pk, "idx_right": right,
                "t_start": float(mjd[left]), "t_peak": float(mjd[pk]), "t_end": float(mjd[right]),
                "peak_flux": float(flux[pk]), "duration_days": float(duration),
            }
        )

    if len(outbursts) < 2:
        return outbursts

    merge_gap = config.get("MERGE_GAP_DAYS", 999)
    merged = [outbursts[0]]
    for ob in outbursts[1:]:
        prev = merged[-1]
        if ob["t_start"] - prev["t_end"] <= merge_gap:
            prev["t_end"] = ob["t_end"]
            prev["idx_right"] = ob["idx_right"]
            prev["duration_days"] = prev["t_end"] - prev["t_start"]
            if ob["peak_flux"] > prev["peak_flux"]:
                prev["peak_flux"], prev["idx_peak"], prev["t_peak"] = (
                    ob["peak_flux"], ob["idx_peak"], ob["t_peak"],
                )
        else:
            merged.append(ob)
    return merged


def detect_lxb_outbursts(lc_df, maxi_name, config):
    """LXB detector: iterative baseline, find_peaks, optional NS HR veto.
    Returns (confirmed_list, baseline, threshold).
    """
    flux = lc_df["flux_total"].values
    mjd = lc_df["mjd"].values

    baseline, sigma_r = iterative_baseline(flux, percentile=config["QUIET_PERCENTILE"])
    threshold = baseline + config["SIGMA_MULT"] * sigma_r

    raw_groups = _find_outburst_peaks(flux, mjd, baseline, sigma_r, config)
    if not raw_groups:
        return [], baseline, threshold

    hr_available = False
    quiet_hr = np.nan
    lc_work = None

    if config["HR_VETO"]:
        lc_work = lc_df.copy()
        win_hr = config["HR_SMOOTHING_WIN"]
        lc_work["hr"] = lc_work["flux_hard"] / (lc_work["flux_soft"].abs() + EPSILON)
        lc_work["hr"] = (
            lc_work["hr"].rolling(win_hr, center=True, min_periods=win_hr // 3).median().clip(0, 10)
        )
        in_ob = np.zeros(len(lc_work), dtype=bool)
        for g in raw_groups:
            in_ob[g["idx_left"] : g["idx_right"] + 1] = True
        quiet_hr = lc_work.loc[~in_ob, "hr"].median()
        hr_available = (
            not np.isnan(quiet_hr) and 0.01 < quiet_hr < config["HR_MAX_QUIESCENT"] and (~in_ob).sum() > 50
        )

    confirmed = []
    for g in raw_groups:
        g_arr = np.arange(g["idx_left"], g["idx_right"] + 1)
        g_flux = flux[g_arr]
        pk_pos = int(np.argmax(g_flux))
        peak_mjd = mjd[g_arr][pk_pos]
        peak_flx = g_flux[pk_pos]
        measured_hr = np.nan

        if hr_available:
            frac = config["HR_NEAR_PEAK_FRAC"]
            near_peak_ix = g_arr[g_flux >= frac * peak_flx]
            measured_hr = lc_work.loc[near_peak_ix, "hr"].median()
            if not np.isnan(measured_hr):
                hr_thresh = quiet_hr * config["HR_SOFTENING"]
                if measured_hr > hr_thresh:
                    continue  # not softening enough -- hard-state flare, reject

        confirmed.append(
            {
                "maxi_name": maxi_name, "t_start": g["t_start"], "t_end": g["t_end"],
                "t_peak": peak_mjd, "peak_flux": peak_flx, "duration_days": g["duration_days"],
                "baseline": baseline, "threshold": threshold,
                "quiescent_hr": quiet_hr, "peak_hr": measured_hr,
                "hr_veto_used": hr_available, "source_type": config["type"],
            }
        )
    return confirmed, baseline, threshold


def detect_hxb_outbursts(lc_df, maxi_name, config):
    """HXB detector: 300-day rolling-median local baseline (80th-pct
    clipped so bright Type-I bursts don't bias it upward), no HR veto.
    Returns (confirmed_list, threshold_array).
    """
    lc_df = lc_df.copy()
    flux = lc_df["flux_total"].values
    mjd = lc_df["mjd"].values
    win = config["ROLLING_WINDOW"]

    flux_ceil = np.nanpercentile(flux, 80)
    flux_clipped = np.clip(flux, None, flux_ceil)

    lc_df["roll_med"] = pd.Series(flux_clipped).rolling(win, center=True, min_periods=win // 3).median().values
    lc_df["roll_mad"] = (
        (pd.Series(flux_clipped) - lc_df["roll_med"]).abs().rolling(win, center=True, min_periods=win // 3).median().values
    )
    lc_df["sigma_r"] = 1.4826 * lc_df["roll_mad"]
    lc_df["thresh"] = lc_df["roll_med"] + config["SIGMA_MULT"] * lc_df["sigma_r"]

    g_med = np.nanmedian(flux)
    g_mad = np.nanmedian(np.abs(flux - g_med))
    g_sigma = max(1.4826 * g_mad, 1e-6)
    g_thresh = g_med + config["SIGMA_MULT"] * g_sigma
    lc_df["thresh"] = lc_df["thresh"].fillna(g_thresh)
    thresh_arr = lc_df["thresh"].values

    sigma_arr = lc_df["sigma_r"].fillna(g_sigma).values
    roll_med_arr = lc_df["roll_med"].fillna(g_med).values
    low_thr_arr = roll_med_arr + _LOW_SIGMA * sigma_arr

    scalar_baseline = float(lc_df["roll_med"].median())
    raw_groups = _find_outburst_peaks(flux, mjd, scalar_baseline, g_sigma, config, thresh_arr=low_thr_arr)
    if not raw_groups:
        return [], thresh_arr

    confirmed = [
        {
            "maxi_name": maxi_name, "t_start": g["t_start"], "t_end": g["t_end"],
            "t_peak": g["t_peak"], "peak_flux": g["peak_flux"], "duration_days": g["duration_days"],
            "baseline": float(lc_df["roll_med"].median()), "threshold": g_thresh,
            "quiescent_hr": np.nan, "peak_hr": np.nan,
            "hr_veto_used": False, "source_type": config["type"],
        }
        for g in raw_groups
    ]
    return confirmed, thresh_arr


def get_config(simbad_otype, compact_object):
    """Route a source to its detection config by SIMBAD otype + compact
    object. Returns (config_dict, 'lxb'|'hxb').
    """
    if simbad_otype == "HXB":
        return HXB_CONFIG, "hxb"
    if compact_object == "BH":
        return BH_LXB_CONFIG, "lxb"
    if compact_object == "NS":
        return NS_LXB_CONFIG, "lxb"
    return UNK_LXB_CONFIG, "lxb"


def run_pipeline(classified_csv: str, outdir: str = ".") -> pd.DataFrame:
    """Run detection over every source in the classified catalogue
    (output of crossmatch/classify_compact_objects.py). Writes
    step2_outbursts.csv, step2_tier0_failed.csv, step2_skipped_persistent.csv
    to ``outdir``. Returns the outburst DataFrame.
    """
    import os

    os.makedirs(outdir, exist_ok=True)
    xrb_df = pd.read_csv(classified_csv)
    xrb_df["maxi_id"] = xrb_df.apply(lambda r: ra_dec_to_maxi_id(r["maxi_ra"], r["maxi_dec"]), axis=1)
    total = len(xrb_df)
    print(f"Loaded {total} XRB sources from {classified_csv}")

    all_outbursts, tier0_failed, skipped_persistent = [], [], []

    for i, row in xrb_df.iterrows():
        src_id = make_source_id(i)
        name, otype, pf, co, maxi_id = (
            row["maxi_name"], row["simbad_otype"], row["persistent_flag"],
            row["compact_object"], row["maxi_id"],
        )
        print(f"\n[{i + 1}/{total}] {src_id} : {name} ({maxi_id}) [{otype}|{co}|{pf}]")

        if pf == "P":
            print("    SKIP: persistent_flag=P")
            skipped_persistent.append(
                {"maxi_name": name, "maxi_id": maxi_id, "simbad_otype": otype, "compact_object": co}
            )
            continue

        lc_df = fetch_maxi_lightcurve(maxi_id)
        if lc_df is None:
            print("    No LC data -- skipping.")
            time.sleep(0.5)
            continue

        passes_t0, peak_flux, noise_rms = tier0_check(lc_df)
        if not passes_t0:
            print(f"    TIER 0 FAIL: peak {peak_flux:.4f} < {TIER0_MIN_PEAK_FLUX}")
            tier0_failed.append(
                {
                    "maxi_name": name, "maxi_id": maxi_id, "simbad_otype": otype,
                    "compact_object": co, "persistent_flag": pf,
                    "peak_flux": peak_flux, "noise_rms": noise_rms,
                }
            )
            time.sleep(0.3)
            continue

        config, branch = get_config(otype, co)
        if branch == "hxb":
            outbursts, _ = detect_hxb_outbursts(lc_df, name, config)
        else:
            outbursts, baseline, threshold = detect_lxb_outbursts(lc_df, name, config)
            print(f"    Baseline={baseline:.4f}  Threshold={threshold:.4f}")

        print(f"    Confirmed outbursts: {len(outbursts)}")
        for k, ob in enumerate(outbursts):
            ob["source_id"] = src_id
            ob["outburst_id"] = f"{src_id}{k + 1}"
            ob["simbad_otype"] = otype
            ob["compact_object"] = co
            ob["persistent_flag"] = pf
        all_outbursts.extend(outbursts)
        time.sleep(0.3)

    ob_df = pd.DataFrame(all_outbursts)
    pd.DataFrame(tier0_failed).to_csv(f"{outdir}/step2_tier0_failed.csv", index=False)
    pd.DataFrame(skipped_persistent).to_csv(f"{outdir}/step2_skipped_persistent.csv", index=False)
    ob_df.to_csv(f"{outdir}/step2_outbursts.csv", index=False)

    print("\n" + "=" * 60)
    print("  DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Total sources            : {total}")
    print(f"  Skipped (persistent)     : {len(skipped_persistent)}")
    print(f"  Tier 0 failed            : {len(tier0_failed)}")
    print(f"  Total outbursts detected : {len(ob_df)}")
    return ob_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="step1b_maxi_xrb_classified.csv")
    parser.add_argument("--outdir", default="data/detection")
    args = parser.parse_args()
    run_pipeline(args.input, args.outdir)
