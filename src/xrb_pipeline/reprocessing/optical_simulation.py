"""
Phase 3 (simulate optical from the X-ray FRED model + dual-axis plot)
and Phase 4 (cross-source summary) -- report section 8.2 second half.

Phase 3 evaluates the stored Norris FRED fit at the ZTF epochs (lag-
shifted by tau_lag), scales it by the calibrated C_source for each beta
run, and compares against the observed ZTF photometry via RMS residual.
Phase 4 flattens every calibrated outburst across the whole JSON
database into one summary table for cross-source comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xrb_pipeline.reprocessing.manual_inspector import BETA_A, BETA_B, MAXI_CONV
from xrb_pipeline.utils.fetch import fetch_maxi_lightcurve
from xrb_pipeline.utils.models import make_norris_fred_smooth


def simulate_optical(outburst: dict, ztf_df: pd.DataFrame, maxi_conv: float = MAXI_CONV) -> dict:
    """Evaluate the stored FRED fit at each ZTF epoch (lag-shifted),
    scale by C_A/C_B, and compute the RMS residual against observed
    F_OIR for both beta runs (report eq. 34-35).

    Returns a dict with F_sim_A, F_sim_B (arrays, mJy) and RMS_A, RMS_B
    (scalars, mJy).
    """
    fred_smooth_erg = make_norris_fred_smooth(outburst, maxi_conv)
    tau_lag = outburst["tau_lag"]

    f_x_at_ztf = fred_smooth_erg(ztf_df["mjd"].values - tau_lag)
    f_sim_a = 10 ** outburst["C_A"] * f_x_at_ztf ** BETA_A
    f_sim_b = 10 ** outburst["C_B"] * f_x_at_ztf ** BETA_B

    resid_a = ztf_df["F_OIR"].values - f_sim_a
    resid_b = ztf_df["F_OIR"].values - f_sim_b
    rms_a = float(np.sqrt(np.mean(resid_a ** 2)))
    rms_b = float(np.sqrt(np.mean(resid_b ** 2)))

    return {"F_sim_A": f_sim_a, "F_sim_B": f_sim_b, "RMS_A": rms_a, "RMS_B": rms_b}


def save_rms_to_json(db: dict, active_source: str, outburst_idx: int, rms_a: float, rms_b: float) -> dict:
    ob = db[active_source]["outbursts"][outburst_idx]
    ob["RMS_A"] = round(rms_a, 5)
    ob["RMS_B"] = round(rms_b, 5)
    return db


def plot_dual_axis(outburst: dict, mjd_w, f_x_raw, ztf_by_band: dict, x1: float, x2: float,
                    pad_left: float = 15, tail_pad: float = 60, title: str = "",
                    maxi_conv: float = MAXI_CONV, savepath: str | None = None):
    """Dual-axis outburst plot: MAXI + FRED model on the left axis,
    ZTF (per band) + reprocessing model (both beta runs) on the right
    axis (report Figure 10). ``tau_lag`` shifts the model curve only.
    """
    import matplotlib.pyplot as plt

    fred_smooth_erg = make_norris_fred_smooth(outburst, maxi_conv)
    tau_lag = outburst["tau_lag"]
    band_colors = {"g": "green", "r": "darkorange", "i": "red"}

    t_dense = np.linspace(x1 - pad_left, x2 + tail_pad, 1000)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlim(x1 - pad_left, x2 + tail_pad)
    ax1.scatter(mjd_w, f_x_raw * 1e7, color="steelblue", s=15, alpha=0.6, label="MAXI (X-ray)")
    ax1.plot(t_dense, fred_smooth_erg(t_dense) * 1e7, color="navy", lw=2, label="X-ray model")
    ax1.set_xlabel("MJD")
    ax1.set_ylabel(r"X-ray flux (1e-7 erg cm$^{-2}$ s$^{-1}$)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    for band, df_b in ztf_by_band.items():
        if len(df_b) == 0:
            continue
        ax2.errorbar(df_b["mjd"], df_b["F_OIR"], yerr=df_b["F_OIR_err"], fmt="s",
                      color=band_colors.get(band, "gray"), ms=5, alpha=0.8, label=f"ZTF {band}-band")

    ax2.plot(t_dense, 10 ** outburst["C_A"] * fred_smooth_erg(t_dense - tau_lag) ** BETA_A,
              "--", color="darkred", lw=2, label=f"Reprocessing (beta={BETA_A})")
    ax2.plot(t_dense, 10 ** outburst["C_B"] * fred_smooth_erg(t_dense - tau_lag) ** BETA_B,
              ":", color="firebrick", lw=2, label=f"Reprocessing (beta={BETA_B})")
    ax2.set_ylabel("Optical flux (mJy)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


# ------------------------------------------------------------------
# Phase 4: cross-source summary
# ------------------------------------------------------------------

def flatten_calibrated_outbursts(db: dict) -> pd.DataFrame:
    """Flatten every fitted + calibrated outburst across the JSON
    database into one summary DataFrame.
    """
    rows = []
    for maxi_id, src in db.items():
        for i, ob in enumerate(src.get("outbursts", [])):
            if ob.get("status") != "fitted" or "C_A" not in ob:
                continue
            rows.append(
                {
                    "maxi_id": maxi_id, "name": src["display_name"], "type": src["compact_object"],
                    "lmxb": src["lmxb"], "ob_idx": i, "x1": ob["x1"], "x2": ob["x2"],
                    "peak_mjd": ob.get("peak_mjd"), "T90": ob.get("T90"), "N_calib": ob.get("N_calib"),
                    "tau_lag": ob.get("tau_lag"), "C_A": ob["C_A"], "C_A_err": ob["C_A_err"],
                    "C_B": ob["C_B"], "C_B_err": ob["C_B_err"], "RMS_A": ob.get("RMS_A"),
                    "RMS_B": ob.get("RMS_B"), "chi2_red": ob.get("chi2_red"),
                }
            )
    return pd.DataFrame(rows)


def plot_c_source_bars(summary_df: pd.DataFrame, savepath: str | None = None):
    """C_source bar chart, one panel per beta run, coloured by compact
    object (report Figure-adjacent summary; not itself a report figure).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    type_color = {"NS": "steelblue", "BH": "crimson"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (col, err_col, label) in zip(
        axes, [("C_A", "C_A_err", f"Run A beta={BETA_A}"), ("C_B", "C_B_err", f"Run B beta={BETA_B}")]
    ):
        colors = summary_df["type"].map(type_color).fillna("gray")
        ax.bar(range(len(summary_df)), summary_df[col], yerr=summary_df[err_col], color=colors,
               alpha=0.8, capsize=4, error_kw={"lw": 1.2})
        ax.set_xticks(range(len(summary_df)))
        ax.set_xticklabels([f"{r.name}\n[{r.ob_idx}]" for _, r in summary_df.iterrows()],
                            rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("C_source"); ax.set_title(label)
        ax.grid(axis="y", ls="--", alpha=0.4)
    axes[0].legend(handles=[Patch(color="steelblue", label="NS"), Patch(color="crimson", label="BH")])
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def plot_rms_comparison(summary_df: pd.DataFrame, savepath: str | None = None):
    """RMS_A vs RMS_B scatter -- which beta fits better per outburst?"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7, 6))
    better_b = summary_df["RMS_B"] < summary_df["RMS_A"]
    colors = np.where(better_b, "firebrick", "navy")
    ax.scatter(summary_df["RMS_A"], summary_df["RMS_B"], c=colors, s=60, edgecolor="k", linewidth=0.5)
    hi = max(summary_df["RMS_A"].max(), summary_df["RMS_B"].max()) * 1.1
    ax.plot([0, hi], [0, hi], "k--", lw=1)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel(f"RMS_A (beta={BETA_A}) [mJy]"); ax.set_ylabel(f"RMS_B (beta={BETA_B}) [mJy]")
    ax.legend(handles=[Patch(color="firebrick", label=f"beta={BETA_B} wins"),
                        Patch(color="navy", label=f"beta={BETA_A} wins")])
    ax.grid(ls="--", alpha=0.4)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"beta={BETA_B} fits better in {int(better_b.sum())}/{len(summary_df)} outbursts")
