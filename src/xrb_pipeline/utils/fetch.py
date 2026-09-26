"""
Data-fetching helpers for MAXI (X-ray) and ZTF/ALeRCE (optical) light curves.

Consolidated from the fetch cells repeated near-verbatim across most of the
project notebooks (manual inspector, automated detection, FRED pipelines,
hardness inspector, reprocessing phases).
"""
from __future__ import annotations

import numpy as np
import requests

MAXI_BASE = "https://maxi.riken.jp/star_data"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; XRB-pipeline/1.0)"}


def fetch_maxi_lightcurve(maxi_id: str, timeout: int = 15):
    """Fetch the MAXI 1-day all-band light curve for one source.

    Parameters
    ----------
    maxi_id : str
        MAXI J-name, e.g. ``"J1911+005"``.

    Returns
    -------
    (mjd, flux) : tuple[np.ndarray, np.ndarray]
        Empty arrays on failure (bad HTTP status, network error, or too few
        valid rows).
    """
    url = f"{MAXI_BASE}/{maxi_id}/{maxi_id}_g_lc_1day_all.dat"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException as e:
        print(f"  [SKIP] {maxi_id}: request failed: {e}")
        return np.array([]), np.array([])

    if resp.status_code != 200:
        print(f"  [SKIP] {maxi_id}: HTTP {resp.status_code}")
        return np.array([]), np.array([])

    lines = [
        l for l in resp.text.strip().split("\n") if l.strip() and not l.startswith("!")
    ]
    parsed = []
    for l in lines:
        parts = l.split()
        if len(parts) >= 2:
            try:
                parsed.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

    if len(parsed) < 10:
        print(f"  [SKIP] {maxi_id}: too few points ({len(parsed)})")
        return np.array([]), np.array([])

    mjd = np.array([p[0] for p in parsed])
    flux = np.array([p[1] for p in parsed])
    return mjd, flux


def fetch_maxi_lightcurve_multiband(maxi_id: str, timeout: int = 15):
    """Fetch the full 9-column MAXI 1-day light curve (all sub-bands).

    Columns (per the MAXI ``.dat`` format): MJD, 2-20 keV flux+err,
    2-4 keV flux+err, 4-10 keV flux+err, 10-20 keV flux+err.

    Returns a dict of numpy arrays keyed by
    ``mjd, flux_total, flux_total_err, flux_2_4, flux_2_4_err,
    flux_4_10, flux_4_10_err, flux_10_20, flux_10_20_err``,
    or an empty dict on failure.
    """
    url = f"{MAXI_BASE}/{maxi_id}/{maxi_id}_g_lc_1day_all.dat"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException as e:
        print(f"  [SKIP] {maxi_id}: request failed: {e}")
        return {}

    if resp.status_code != 200:
        print(f"  [SKIP] {maxi_id}: HTTP {resp.status_code}")
        return {}

    lines = [
        l for l in resp.text.strip().split("\n") if l.strip() and not l.startswith("!")
    ]
    rows = []
    for l in lines:
        parts = l.split()
        if len(parts) >= 9:
            try:
                rows.append([float(p) for p in parts[:9]])
            except ValueError:
                continue

    if len(rows) < 10:
        print(f"  [SKIP] {maxi_id}: too few points ({len(rows)})")
        return {}

    arr = np.array(rows)
    cols = [
        "mjd",
        "flux_total", "flux_total_err",
        "flux_2_4", "flux_2_4_err",
        "flux_4_10", "flux_4_10_err",
        "flux_10_20", "flux_10_20_err",
    ]
    return {name: arr[:, i] for i, name in enumerate(cols)}


ZTF_FID_TO_BAND = {1: "g", 2: "r", 3: "i"}


def fetch_ztf_all_bands(ztf_id: str, timeout: int = 20):
    """Fetch ZTF photometry (all bands) for a source via the ALeRCE REST API.

    Converts AB magnitudes to mJy flux density (with propagated error)
    at fetch time.

    Parameters
    ----------
    ztf_id : str
        ALeRCE/ZTF object id, e.g. ``"ZTF18accedeu"``.

    Returns
    -------
    ztf_df_all : pandas.DataFrame
        Columns ``[mjd, band, mag, mag_err, F_OIR, F_OIR_err]``, all bands
        pooled and sorted by MJD.
    ztf_by_band : dict[str, pandas.DataFrame]
        Same columns, split per band (``g``, ``r``, ``i``) for per-band
        plotting.

    Raises
    ------
    RuntimeError
        On a non-200 response or an empty detections list.
    """
    import pandas as pd

    url = f"https://api.alerce.online/ztf/v1/objects/{ztf_id}/lightcurve"
    resp = requests.get(url, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(f"Could not fetch ZTF data for {ztf_id}: HTTP {resp.status_code}")

    detections = resp.json().get("detections", [])
    if not detections:
        raise RuntimeError(f"No ZTF detections returned for {ztf_id}")

    rows = []
    for d in detections:
        band = ZTF_FID_TO_BAND.get(d.get("fid"))
        mag = d.get("magpsf")
        mag_err = d.get("sigmapsf")
        mjd = d.get("mjd")
        if band is None or mag is None or mjd is None:
            continue

        f_oir = 3631000.0 * 10 ** (-0.4 * mag)  # mJy, AB system
        f_oir_err = 0.4 * np.log(10) * f_oir * (mag_err or 0.0)

        rows.append(
            {
                "mjd": mjd,
                "band": band,
                "mag": mag,
                "mag_err": mag_err,
                "F_OIR": f_oir,
                "F_OIR_err": f_oir_err,
            }
        )

    ztf_df_all = pd.DataFrame(rows).sort_values("mjd").reset_index(drop=True)
    ztf_by_band = {b: g.reset_index(drop=True) for b, g in ztf_df_all.groupby("band")}
    return ztf_df_all, ztf_by_band


def ab_mag_to_mjy(mag, mag_err=None):
    """Convert AB magnitude to flux density in mJy (and propagate error).

    F_OIR [mJy] = 3631e3 * 10^(-0.4 * mag)
    sigma_F = 0.4 * ln(10) * F * sigma_mag
    """
    f_oir = 3631000.0 * 10 ** (-0.4 * np.asarray(mag))
    if mag_err is None:
        return f_oir
    f_err = 0.4 * np.log(10) * f_oir * np.asarray(mag_err)
    return f_oir, f_err
