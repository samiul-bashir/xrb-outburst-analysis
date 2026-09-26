"""
Manual outburst inspector: Norris-FRED fit (Phase 0), ZTF fetch + Crab
conversion + HR state labels (Phase 1), and hard-state C_source
calibration (Phase 2). Report sections 8.1-8.2 (first half).

This module holds the reusable functions; the interactive per-outburst
workflow (visual window selection, iterative guess tuning) is meant to
be driven from a notebook -- see notebooks/06_reprocessing.ipynb and
docs/xrb_outburst_inspector_guide_phase0.md for the session workflow.

Persists results to the shared JSON outburst database (see
xrb_pipeline.utils.json_db) with the schema documented in
docs/json_db_schema.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from xrb_pipeline.utils.fetch import fetch_maxi_lightcurve_multiband
from xrb_pipeline.utils.models import norris_fred

MAXI_CONV = 2.4e-8 / 3.3  # ct/s/cm^2 -> erg/cm^2/s (Crab normalisation)
BETA_A = 0.50   # X-ray reprocessing prior (van Paradijs & McClintock 1994)
BETA_B = 0.61   # Russell et al. (2006) empirical BHXB slope
HR_HARD = 1.0   # HR > this -> hard state (4-10 / 2-4 keV)
HR_SOFT = 0.5   # HR < this -> soft state
DT_MATCH = 1.0  # max |delta MJD| for ZTF-MAXI pairing [days]


# ------------------------------------------------------------------
# Phase 0: FRED fit on a manually-selected outburst window
# ------------------------------------------------------------------

def fit_outburst_fred(mjd, flux, flux_err, x1: float, x2: float,
                       t_start_guess: float, t_rise_guess: float, t_decay_guess: float):
    """Fit the Norris FRED to MAXI data in [x1, x2] (report Phase 0-E).

    Returns a dict with fitted params, uncertainties, derived peak
    time, T90 (cumulative-fluence, dt-weighted), chi2_red and RMS; or
    ``{"fit_ok": False}`` if the fit doesn't converge or the window has
    too few points.
    """
    mask = (mjd >= x1) & (mjd <= x2)
    t, f, f_e = mjd[mask], flux[mask], flux_err[mask]
    if len(t) < 5:
        print(f"ERROR: only {len(t)} points in MJD {x1}-{x2}. Widen the window.")
        return {"fit_ok": False}

    peak_idx = int(np.argmax(f))
    peak_mjd, peak_flux = float(t[peak_idx]), float(f[peak_idx])

    sigma_fit = np.where(
        np.isfinite(f_e) & (f_e > 0), f_e, np.where(np.abs(f) < 1e-12, 1e-6, np.sqrt(np.abs(f)))
    )

    t_start_hi = min(t_start_guess + 0.1, peak_mjd - 0.5)
    t_start_lo = x1 - 50.0
    p0 = [t_start_guess, t_rise_guess, t_decay_guess, peak_flux]
    lower = [t_start_lo, 0.01, 0.5, 0.0]
    upper = [t_start_hi, (x2 - x1) * 2, (x2 - x1) * 5, peak_flux * 10]
    p0 = [np.clip(p0[i], lower[i] + 1e-6, upper[i] - 1e-6) for i in range(4)]

    def model(t_arr, t_start, t_rise, t_decay, amplitude):
        return norris_fred(t_arr, A=amplitude, t_start=t_start, t_rise=t_rise, t_decay=t_decay)

    try:
        popt, pcov = curve_fit(
            model, t, f, p0=p0, bounds=(lower, upper), sigma=sigma_fit, absolute_sigma=True, maxfev=20000
        )
    except RuntimeError as e:
        print(f"FRED fit did not converge: {e}")
        return {"fit_ok": False}

    perr = np.sqrt(np.diag(pcov))
    t_start_fit, t_rise_fit, t_decay_fit, amplitude_fit = popt
    t_peak_fit = t_start_fit + np.sqrt(t_rise_fit * t_decay_fit)

    # T90 via dt-weighted cumulative fluence
    baseline = np.nanmedian(f)
    net = np.clip(f - baseline, 0, None)
    dt_v = np.diff(t)
    dt_v = np.append(dt_v, dt_v[-1])
    cum = np.cumsum(net * dt_v)
    total = cum[-1]
    t90 = None
    if total > 0:
        t5_mjd = float(np.interp(0.05 * total, cum, t))
        t95_mjd = float(np.interp(0.95 * total, cum, t))
        t90 = t95_mjd - t5_mjd

    f_model = model(t, *popt)
    residuals = f - f_model
    chi2 = np.sum((residuals / sigma_fit) ** 2)
    dof = max(len(t) - 4, 1)
    chi2_red = chi2 / dof
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "fit_ok": True,
        "t_start": round(float(t_start_fit), 2), "t_rise": round(float(t_rise_fit), 2),
        "t_decay": round(float(t_decay_fit), 2), "amplitude": round(float(amplitude_fit), 6),
        "peak_mjd": round(float(t_peak_fit), 2), "T90": round(float(t90), 1) if t90 is not None else None,
        "chi2_red": round(float(chi2_red), 3), "rms": round(float(rms), 6),
        "perr": perr.tolist(), "x1": float(x1), "x2": float(x2),
    }


def save_outburst_fit(db: dict, active_source: str, fit_result: dict, status: str = "fitted") -> dict:
    """Insert or update an outburst entry in the JSON db for
    ``active_source``, keyed by (x1, x2). ``chi2_red`` is diagnostic
    only -- STATUS is always "fitted" if curve_fit converged; pass
    status="rejected" explicitly to override.
    """
    entry = {k: fit_result[k] for k in ("x1", "x2", "t_start", "t_rise", "t_decay", "amplitude",
                                         "peak_mjd", "T90", "chi2_red", "rms")}
    entry["status"] = status

    outbursts = db[active_source].setdefault("outbursts", [])
    for ob in outbursts:
        if ob["x1"] == entry["x1"] and ob["x2"] == entry["x2"]:
            ob.update(entry)
            return db
    outbursts.append(entry)
    return db


# ------------------------------------------------------------------
# Phase 1: ZTF fetch, Crab conversion, HR state labels, FRED smooth model
# ------------------------------------------------------------------

def crop_maxi_to_window(maxi_data: dict, x1: float, x2: float) -> dict:
    """Restrict a multiband MAXI fetch dict to [x1, x2]."""
    win = (maxi_data["mjd"] >= x1) & (maxi_data["mjd"] <= x2)
    return {k: v[win] for k, v in maxi_data.items()}


def assign_hr_states(maxi_window: dict, hr_hard: float = HR_HARD, hr_soft: float = HR_SOFT):
    """Compute HR = flux_4_10/flux_2_4 and label each epoch hard/soft/int."""
    hr = np.full(len(maxi_window["mjd"]), np.nan)
    ok = (maxi_window["flux_2_4"] > 0) & np.isfinite(maxi_window["flux_4_10"])
    hr[ok] = maxi_window["flux_4_10"][ok] / maxi_window["flux_2_4"][ok]
    state = np.where(hr > hr_hard, "hard", np.where(hr < hr_soft, "soft", "int"))
    return hr, state


def fetch_ztf_for_outburst(ztf_id: str, x1: float, x2: float, alerce_client=None) -> pd.DataFrame | None:
    """Fetch ZTF all-band detections and crop to [x1, x2], converting
    AB magnitude to mJy (F_OIR, F_OIR_err). Returns None if nothing
    falls in the window.
    """
    if alerce_client is None:
        from alerce.core import Alerce

        alerce_client = Alerce()
    band_fid = {1: "g", 2: "r", 3: "i"}

    ztf_raw = alerce_client.query_lightcurve(oid=ztf_id, survey="ztf")
    if ztf_raw is None or len(ztf_raw.get("detections", [])) == 0:
        return None

    df = pd.DataFrame(ztf_raw["detections"])
    df["band"] = df["fid"].map(band_fid)
    df = df.dropna(subset=["magpsf", "sigmapsf", "band"])
    df = df[(df["mjd"] >= x1) & (df["mjd"] <= x2)].sort_values("mjd").reset_index(drop=True)
    if len(df) == 0:
        return None

    df["F_OIR"] = 3631.0 * 10.0 ** (-0.4 * df["magpsf"]) * 1000.0
    df["F_OIR_err"] = df["F_OIR"] * df["sigmapsf"] / 1.0857
    return df


# ------------------------------------------------------------------
# Phase 2: hard-state-only C_source calibration
# ------------------------------------------------------------------

def calibrate_c_source(ztf_df: pd.DataFrame, maxi_window: dict, state: np.ndarray, outburst: dict,
                        beta: float, dt_match: float = DT_MATCH, maxi_conv: float = MAXI_CONV):
    """Fit the weighted-mean intercept C_source for one beta (report
    eq. 31-33), pairing each ZTF point with the nearest hard-state MAXI
    epoch within ``dt_match`` days. Only hard-state epochs enter
    calibration -- the OIR-X-ray reprocessing relation is calibrated
    against the hard state specifically (see
    docs/methodology-decisions: soft-state exclusion is physically
    motivated, since the disc emission mechanism changes).

    Returns (C, C_err, N_calib, per_point_dict) -- C/C_err are None if
    N_calib < 3.
    """
    from xrb_pipeline.utils.models import make_norris_fred_smooth

    fred_smooth_erg = make_norris_fred_smooth(outburst, maxi_conv)
    mjd_w = maxi_window["mjd"]

    ln10 = np.log(10.0)
    cal_mjd, cal_foir, cal_fx, cal_soir, cal_sx = [], [], [], [], []

    for _, row in ztf_df.iterrows():
        ztf_t = float(row["mjd"])
        idx = int(np.argmin(np.abs(mjd_w - ztf_t)))
        if abs(mjd_w[idx] - ztf_t) > dt_match or state[idx] != "hard":
            continue
        f_oir, sf_oir = float(row["F_OIR"]), float(row["F_OIR_err"])
        f_x = float(fred_smooth_erg(np.array([ztf_t]))[0])
        sf_x = float(np.abs(maxi_window["flux_total_err"][idx]) * maxi_conv)
        if f_oir <= 0 or f_x <= 0:
            continue
        cal_mjd.append(ztf_t); cal_foir.append(f_oir); cal_fx.append(f_x)
        cal_soir.append(sf_oir); cal_sx.append(sf_x)

    n_calib = len(cal_mjd)
    if n_calib < 3:
        return None, None, n_calib, {}

    f_oir_a, f_x_a = np.array(cal_foir), np.array(cal_fx)
    soir_a, sx_a = np.array(cal_soir), np.array(cal_sx)
    log_foir, log_fx = np.log10(f_oir_a), np.log10(f_x_a)

    c_i = log_foir - beta * log_fx
    sig_c = np.sqrt((soir_a / (f_oir_a * ln10)) ** 2 + (beta * sx_a / (f_x_a * ln10)) ** 2)
    sig_c = np.where(sig_c < 1e-12, 1e-12, sig_c)
    w = 1.0 / sig_c ** 2
    c_wm = np.sum(w * c_i) / np.sum(w)
    c_err = 1.0 / np.sqrt(np.sum(w))

    return float(c_wm), float(c_err), n_calib, {"mjd": cal_mjd, "C_i": c_i, "sigma_C_i": sig_c}


def run_phase2_calibration(ztf_df: pd.DataFrame, maxi_window: dict, state: np.ndarray, outburst: dict):
    """Run both beta anchors (BETA_A, BETA_B) plus tau_lag. Returns a
    dict ready to merge into the outburst's JSON entry.
    """
    c_a, c_a_err, n_calib, _ = calibrate_c_source(ztf_df, maxi_window, state, outburst, BETA_A)
    c_b, c_b_err, _, _ = calibrate_c_source(ztf_df, maxi_window, state, outburst, BETA_B)

    tau_lag = None
    if n_calib >= 3 and len(ztf_df) > 0:
        ztf_peak_mjd = float(ztf_df.loc[ztf_df["F_OIR"].idxmax(), "mjd"])
        tau_lag = round(ztf_peak_mjd - outburst["peak_mjd"], 2)

    return {
        "N_calib": int(n_calib), "tau_lag": tau_lag,
        "C_A": round(c_a, 4) if c_a is not None else None,
        "C_A_err": round(c_a_err, 4) if c_a_err is not None else None,
        "C_B": round(c_b, 4) if c_b is not None else None,
        "C_B_err": round(c_b_err, 4) if c_b_err is not None else None,
        "RMS_A": None, "RMS_B": None,  # filled in Phase 3 (optical_simulation.py)
    }
