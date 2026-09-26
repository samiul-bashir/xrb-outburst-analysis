"""
Free-beta fitting (report section 8.3, Phase 5): solve for beta and
C_source simultaneously via weighted least squares in log-log space,
instead of assuming a fixed literature beta as in Phase 2. Compares the
fitted slope against NS/BH literature anchors (Russell et al. 2006).

Locked-in decisions carried over from Phase 2-4 (do not re-tune here):
hard-state-only calibration, HR_HARD=0.6/HR_SOFT=0.3, DT_MATCH=1.0 day,
the bug-fixed MAXI_CONV constant, and beta reference values 0.50/0.61.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from scipy.optimize import curve_fit

from xrb_pipeline.reprocessing.manual_inspector import BETA_A, BETA_B, MAXI_CONV
from xrb_pipeline.utils.fetch import fetch_maxi_lightcurve

HR_HARD = 0.6
HR_SOFT = 0.3
DT_MATCH = 1.0  # days

# Global NS/BH literature slope anchors (Russell et al. 2006)
BETA_NS_LIT, BETA_NS_LIT_ERR = 0.63, 0.04
BETA_BH_LIT, BETA_BH_LIT_ERR = 0.61, 0.02

# Minimum sample size for a trustworthy 2-parameter (slope+intercept) fit
MIN_N_BETA = 5


def fetch_maxi_multiband(maxi_id: str) -> pd.DataFrame:
    """Full band-resolved MAXI 1-day light curve as a DataFrame with
    columns [mjd, flux_2_20, err_2_20, flux_2_4, err_2_4, flux_4_10,
    err_4_10, flux_10_20, err_10_20], all in ct/s/cm^2.
    """
    url = f"https://maxi.riken.jp/star_data/{maxi_id}/{maxi_id}_g_lc_1day_all.dat"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Could not fetch MAXI data for {maxi_id}: HTTP {resp.status_code}")

    lines = [l for l in resp.text.strip().split("\n") if l.strip() and not l.startswith("!")]
    cols = ["mjd", "flux_2_20", "err_2_20", "flux_2_4", "err_2_4",
            "flux_4_10", "err_4_10", "flux_10_20", "err_10_20"]
    rows = []
    for l in lines:
        parts = l.split()
        if len(parts) < 9:
            continue
        try:
            rows.append([float(p) for p in parts[:9]])
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=cols)


def hardness_ratio_state(hr, hr_hard: float = HR_HARD, hr_soft: float = HR_SOFT) -> str:
    """Label a hardness-ratio value as hard/soft/intermediate."""
    if hr >= hr_hard:
        return "hard"
    if hr <= hr_soft:
        return "soft"
    return "intermediate"


def build_state_labelled_maxi(active_source: str, x1: float, x2: float, pad_left: float = 15,
                               tail_pad: float = 60, maxi_conv: float = MAXI_CONV) -> pd.DataFrame:
    """Fetch MAXI (single-band, for flux_erg) + multiband (for HR), crop
    to [x1-pad_left, x2+tail_pad], and label each epoch hard/soft/
    intermediate/unknown. HR = flux_4_10 / flux_2_4, guarded against a
    non-positive soft band.
    """
    mjd_full, flux_full = fetch_maxi_lightcurve(active_source)
    mask = (mjd_full >= x1 - pad_left) & (mjd_full <= x2 + tail_pad)
    maxi_df = pd.DataFrame({"mjd": mjd_full[mask], "flux_erg": flux_full[mask] * maxi_conv})

    band_df = fetch_maxi_multiband(active_source)
    band_df = band_df[(band_df["mjd"] >= x1 - pad_left) & (band_df["mjd"] <= x2 + tail_pad)]
    maxi_df = maxi_df.merge(band_df[["mjd", "flux_2_4", "flux_4_10"]], on="mjd", how="left")

    soft_ok = maxi_df["flux_2_4"] > 0
    maxi_df["hr"] = np.where(soft_ok, maxi_df["flux_4_10"] / maxi_df["flux_2_4"], np.nan)
    maxi_df["state"] = maxi_df["hr"].apply(lambda hr: hardness_ratio_state(hr) if pd.notnull(hr) else "unknown")
    return maxi_df


def build_calibration_pairs(maxi_df: pd.DataFrame, ztf_df: pd.DataFrame, dt_match: float = DT_MATCH):
    """Pair each ZTF point with the nearest hard-state MAXI point within
    ``dt_match`` days, then drop any pair with non-positive flux before
    the log-log transform (background-subtracted X-ray flux can dip to
    zero/negative at low count rates).

    Returns (x, y, w, sig_c, cal_band) -- x=log10(F_X), y=log10(F_OIR),
    w=1/sigma_y^2 (optical error only), cal_band=ZTF band per pair.
    """
    hard_maxi = maxi_df[maxi_df["state"] == "hard"].reset_index(drop=True)

    cal_foir, cal_fx, cal_soir, cal_band = [], [], [], []
    for _, zrow in ztf_df.iterrows():
        dt = np.abs(hard_maxi["mjd"] - zrow["mjd"])
        if dt.empty:
            continue
        j = dt.idxmin()
        if dt.loc[j] > dt_match:
            continue
        mrow = hard_maxi.loc[j]
        cal_foir.append(zrow["F_OIR"]); cal_fx.append(mrow["flux_erg"])
        cal_soir.append(zrow["F_OIR_err"]); cal_band.append(zrow["band"])

    cal_foir, cal_fx = np.array(cal_foir), np.array(cal_fx)
    cal_soir, cal_band = np.array(cal_soir), np.array(cal_band)

    valid = (cal_fx > 0) & (cal_foir > 0) & (cal_soir > 0) & np.isfinite(cal_fx) & np.isfinite(cal_foir)
    cal_foir, cal_fx, cal_soir, cal_band = cal_foir[valid], cal_fx[valid], cal_soir[valid], cal_band[valid]

    x = np.log10(cal_fx)
    y = np.log10(cal_foir)
    sig_c = cal_soir / (cal_foir * np.log(10))
    w = 1.0 / sig_c ** 2
    return x, y, w, sig_c, cal_band


def fit_beta_free(x, y, w, min_n_beta: int = MIN_N_BETA):
    """Analytic weighted least squares in log-log space (report eq.
    36-39). Returns a dict with beta_free/C_free (and uncertainties) or
    all-None if N < min_n_beta or the x-range is degenerate (zero
    variance -- slope not identifiable).
    """
    n_pairs = len(x)
    if n_pairs < min_n_beta:
        return {"beta_free": None, "sigma_beta_free": None, "C_free": None, "sigma_C_free": None, "N_beta": n_pairs}

    x_bar = np.sum(w * x) / np.sum(w)
    y_bar = np.sum(w * y) / np.sum(w)
    sxx = np.sum(w * (x - x_bar) ** 2)
    sxy = np.sum(w * (x - x_bar) * (y - y_bar))

    if sxx <= 0:
        return {"beta_free": None, "sigma_beta_free": None, "C_free": None, "sigma_C_free": None, "N_beta": n_pairs}

    beta_free = sxy / sxx
    c_free = y_bar - beta_free * x_bar
    sigma_beta_free = np.sqrt(1.0 / sxx)
    sigma_c_free = np.sqrt(np.sum(w * x ** 2) / (np.sum(w) * sxx))

    return {
        "beta_free": float(beta_free), "sigma_beta_free": float(sigma_beta_free),
        "C_free": float(c_free), "sigma_C_free": float(sigma_c_free), "N_beta": n_pairs,
    }


def fit_beta_free_scipy_crosscheck(x, y, sig_c):
    """Numerical cross-check of fit_beta_free via curve_fit; should
    reproduce the analytic result almost exactly. Returns
    (beta, beta_err, C, C_err).
    """
    def linear_model(x_arr, beta, c):
        return c + beta * x_arr

    popt, pcov = curve_fit(linear_model, x, y, sigma=sig_c, absolute_sigma=True, p0=[0.6, 15.0])
    beta, c = popt
    beta_err, c_err = np.sqrt(np.diag(pcov))
    return float(beta), float(beta_err), float(c), float(c_err)


def save_beta_free_to_json(db: dict, active_source: str, outburst_idx: int, fit: dict) -> dict:
    """Write beta_free/sigma_beta_free/C_free/sigma_C_free/N_beta into
    the outburst's JSON entry. No-op (fixed-beta results untouched) if
    fit["beta_free"] is None.
    """
    if fit["beta_free"] is None:
        return db
    ob = db[active_source]["outbursts"][outburst_idx]
    ob["beta_free"] = fit["beta_free"]
    ob["sigma_beta_free"] = fit["sigma_beta_free"]
    ob["C_free"] = fit["C_free"]
    ob["sigma_C_free"] = fit["sigma_C_free"]
    ob["N_beta"] = fit["N_beta"]
    return db


def literature_anchor(compact_object: str):
    """Return (beta_lit, beta_lit_err) for 'NS' or 'BH'; compact_object
    is authoritative -- never infer NS/BH from anything else.
    """
    return (BETA_NS_LIT, BETA_NS_LIT_ERR) if compact_object == "NS" else (BETA_BH_LIT, BETA_BH_LIT_ERR)


def plot_loglog_fit(x, y, sig_c, cal_band, fit: dict, c_a: float, c_b: float, title: str = "",
                     savepath: str | None = None):
    """Log-log F_OIR vs F_X scatter with the free-fit line plus the
    fixed-beta reference lines from Phase 2 (report Phase-5 Cell 5-H).
    """
    import matplotlib.pyplot as plt

    if fit["beta_free"] is None:
        print("Skipped -- no free-beta fit available for this outburst.")
        return

    band_colors = {"g": "green", "r": "darkorange", "i": "red"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for band, col in band_colors.items():
        m = cal_band == band
        if m.any():
            ax.errorbar(x[m], y[m], yerr=sig_c[m], fmt="o", color=col, ms=5, alpha=0.8, lw=0.5,
                        capsize=2, label=f"ZTF {band}")

    x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
    ax.plot(x_line, fit["C_free"] + fit["beta_free"] * x_line, "-", color="black", lw=2,
            label=f"Free fit: beta={fit['beta_free']:.2f}+/-{fit['sigma_beta_free']:.2f}")
    ax.plot(x_line, c_a + BETA_A * x_line, "--", color="gray", lw=1.5, label=f"Fixed beta={BETA_A}")
    ax.plot(x_line, c_b + BETA_B * x_line, ":", color="gray", lw=1.5, label=f"Fixed beta={BETA_B}")

    ax.set_xlabel("log10 F_X [erg cm^-2 s^-1]"); ax.set_ylabel("log10 F_OIR [mJy]")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(ls="--", alpha=0.4)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def cross_source_beta_summary(db: dict) -> pd.DataFrame:
    """Flatten every outburst with a saved beta_free across the whole
    database into a summary DataFrame (report Phase-5 Cell 5-K).
    """
    rows = []
    for maxi_id, src in db.items():
        for i, ob in enumerate(src.get("outbursts", [])):
            if "beta_free" not in ob:
                continue
            rows.append(
                {
                    "maxi_id": maxi_id, "name": src["display_name"], "type": src["compact_object"],
                    "ob_idx": i, "beta_free": ob["beta_free"], "sigma_beta_free": ob["sigma_beta_free"],
                    "N_beta": ob["N_beta"],
                }
            )
    return pd.DataFrame(rows)
