"""
Hardness-Intensity Diagrams, X-ray and optical (report section 7).

X-ray hardness = flux(4-10 keV) / flux(2-4 keV); intensity = flux_total
(2-20 keV). Optical "color" = F(band1) - F(band2) (mJy, converted from
AB magnitude); optical intensity = F(band1) + F(band2). ZTF bands
aren't observed simultaneously, so points are grouped to nightly medians
per band before pairing; the g-r pair is preferred (best cadence),
falling back to r-i when g is too sparse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xrb_pipeline.utils.fetch import fetch_maxi_lightcurve_multiband, fetch_ztf_all_bands

EPSILON = 1e-6
MIN_NIGHTS_FOR_PAIR = 3
PRIMARY_PAIR = ("g", "r")
FALLBACK_PAIR = ("r", "i")


def compute_xray_hardness(maxi_df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``hr_x`` column: flux_4_10 / flux_2_4 (guarded against a
    zero/negative soft band).
    """
    out = maxi_df.copy()
    out["hr_x"] = out["flux_4_10"] / (out["flux_2_4"].abs() + EPSILON)
    return out


def nightly_binned(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-band ZTF DataFrame to nightly-mean (mjd, F_OIR)."""
    d = df.copy()
    d["night"] = np.floor(d["mjd"]).astype(int)
    return d.groupby("night").agg(mjd=("mjd", "mean"), F_OIR=("F_OIR", "mean")).reset_index()


def matched_band_pair(ztf_by_band: dict, band1: str, band2: str, t_start=None, t_end=None,
                       min_nights: int = MIN_NIGHTS_FOR_PAIR):
    """Nightly-bin and inner-join two ZTF bands on night, optionally
    restricted to [t_start, t_end]. Returns the merged DataFrame or
    None if fewer than ``min_nights`` nights match.
    """
    if band1 not in ztf_by_band or band2 not in ztf_by_band:
        return None
    b1, b2 = nightly_binned(ztf_by_band[band1]), nightly_binned(ztf_by_band[band2])
    if t_start is not None and t_end is not None:
        b1 = b1[(b1["mjd"] >= t_start) & (b1["mjd"] <= t_end)]
        b2 = b2[(b2["mjd"] >= t_start) & (b2["mjd"] <= t_end)]
    merged = b1.merge(b2, on="night", suffixes=(f"_{band1}", f"_{band2}"))
    if len(merged) < min_nights:
        return None
    merged["mjd"] = merged[[f"mjd_{band1}", f"mjd_{band2}"]].mean(axis=1)
    return merged


def find_usable_band_pair(ztf_by_band: dict, t_start=None, t_end=None, min_nights: int = MIN_NIGHTS_FOR_PAIR):
    """Try PRIMARY_PAIR then FALLBACK_PAIR. Returns (merged_df, band1,
    band2) or (None, None, None) if neither pair has enough matched
    nights.
    """
    for b1, b2 in (PRIMARY_PAIR, FALLBACK_PAIR):
        merged = matched_band_pair(ztf_by_band, b1, b2, t_start, t_end, min_nights)
        if merged is not None:
            return merged, b1, b2
    return None, None, None


def plot_xray_hid(maxi_df: pd.DataFrame, title: str = "", savepath: str | None = None):
    """X-ray HID + hardness-vs-time, points colored by MJD (report
    Figure 8a / 9a/c).
    """
    import matplotlib.pyplot as plt

    df = compute_xray_hardness(maxi_df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc = axes[0].scatter(df["hr_x"], df["flux_total"], c=df["mjd"], cmap="viridis", s=15, alpha=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Hardness (4-10 keV / 2-4 keV) [log]")
    axes[0].set_ylabel("Intensity (flux_total, 2-20 keV)")
    axes[0].set_title(f"{title} -- X-ray HID")
    axes[0].grid(alpha=0.25, lw=0.5, which="both")
    fig.colorbar(sc, ax=axes[0]).set_label("MJD")

    axes[1].plot(df["mjd"], df["hr_x"], color="mediumpurple", marker="o", ms=3, lw=0.8, alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("MJD"); axes[1].set_ylabel("Hardness (4-10 / 2-4 keV) [log]")
    axes[1].set_title(f"{title} -- X-ray Hardness vs Time")
    axes[1].grid(alpha=0.25, lw=0.5, which="both")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def plot_optical_hid(ztf_pair_df: pd.DataFrame, band1: str, band2: str, title: str = "",
                      savepath: str | None = None):
    """Optical HID + color-vs-time for a matched band pair (report
    Figure 8b / 9b/d).
    """
    import matplotlib.pyplot as plt

    color = ztf_pair_df[f"F_OIR_{band1}"] - ztf_pair_df[f"F_OIR_{band2}"]
    intensity = ztf_pair_df[f"F_OIR_{band1}"] + ztf_pair_df[f"F_OIR_{band2}"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(color, intensity, c=ztf_pair_df["mjd"], cmap="plasma", s=30, edgecolor="k", linewidth=0.3)
    axes[0].set_xlabel(f"Color (F_{band1} - F_{band2}, mJy)")
    axes[0].set_ylabel(f"Intensity (F_{band1} + F_{band2}, mJy)")
    axes[0].set_title(f"{title} -- Optical HID ({band1}-{band2})")
    axes[0].grid(alpha=0.25, lw=0.5)
    fig.colorbar(sc, ax=axes[0]).set_label("MJD")

    axes[1].plot(ztf_pair_df["mjd"], color, color="darkorange", marker="o", ms=4, lw=0.8)
    axes[1].set_xlabel("MJD"); axes[1].set_ylabel(f"Color (F_{band1} - F_{band2}, mJy)")
    axes[1].set_title(f"{title} -- Optical Color vs Time")
    axes[1].grid(alpha=0.25, lw=0.5)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def inspect_outburst_hid(maxi_id: str, ztf_id: str | None, t_start: float, t_end: float,
                          source_name: str = "", savedir: str | None = None):
    """Convenience wrapper: fetch MAXI multiband + ZTF (if available),
    restrict to [t_start, t_end], and produce the X-ray HID (always)
    plus the optical HID (if a usable band pair exists).
    """
    maxi_data = fetch_maxi_lightcurve_multiband(maxi_id)
    if not maxi_data:
        print(f"[ERROR] No MAXI data for {maxi_id}")
        return
    maxi_df = pd.DataFrame(maxi_data)
    win = maxi_df[(maxi_df["mjd"] >= t_start) & (maxi_df["mjd"] <= t_end)]
    plot_xray_hid(win, title=source_name, savepath=f"{savedir}/xray_hid.png" if savedir else None)

    if not ztf_id:
        print("[ZTF] no ztf_id given -- skipping optical HID.")
        return
    try:
        _, ztf_by_band = fetch_ztf_all_bands(ztf_id)
    except Exception as e:
        print(f"[ZTF] unavailable for {ztf_id}: {e}")
        return

    merged, b1, b2 = find_usable_band_pair(ztf_by_band, t_start, t_end)
    if merged is None:
        print("[ZTF] no band pair has enough matched nights in this window -- skipping optical HID.")
        return
    plot_optical_hid(merged, b1, b2, title=source_name, savepath=f"{savedir}/optical_hid.png" if savedir else None)
