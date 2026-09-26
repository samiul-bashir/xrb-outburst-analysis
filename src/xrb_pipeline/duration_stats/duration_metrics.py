"""
Outburst duration statistics: T90, T_rise, T_decay (report section 6).

All three are derived from the baseline-subtracted, cumulative-flux
profile of the MAXI light curve within a detected outburst window --
the same cumulative-fluence formalism used for gamma-ray burst
durations. T90 is the interval containing the central 90% of net
fluence; splitting that interval at the flux peak gives T_rise and
T_decay (T_rise + T_decay = T90 by construction).

Consumes step2_outbursts.csv (from detection/tiered_detector.py) plus
the underlying MAXI light curves, and appends t90_days, t_rise_days,
t_decay_days columns.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from xrb_pipeline.detection.tiered_detector import fetch_maxi_lightcurve, ra_dec_to_maxi_id


def compute_t90(mjd, flux, baseline, t_start, t_end):
    """T90 via the cumulative-flux method (report eq. 11-14).

    Returns (t90, t5_mjd, t95_mjd), all NaN if the window has fewer
    than 3 points or zero net (baseline-subtracted) flux.
    """
    mask = (mjd >= t_start) & (mjd <= t_end)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan

    ob_mjd = mjd[mask]
    net_flux = np.clip(flux[mask] - baseline, 0, None)
    if net_flux.sum() == 0:
        return np.nan, np.nan, np.nan

    cumsum = np.cumsum(net_flux)
    total = cumsum[-1]
    t5_idx = min(np.searchsorted(cumsum, 0.05 * total), len(ob_mjd) - 1)
    t95_idx = min(np.searchsorted(cumsum, 0.95 * total), len(ob_mjd) - 1)
    t5_mjd, t95_mjd = ob_mjd[t5_idx], ob_mjd[t95_idx]
    return t95_mjd - t5_mjd, t5_mjd, t95_mjd


def compute_trise_tdecay(mjd, flux, baseline, t_start, t_end):
    """Split the [t5, t95] window at the flux peak into T_rise and
    T_decay (report eq. 15-16). Returns (t_rise, t_decay, t_peak); all
    NaN if the peak doesn't fall inside [t5, t95] (multi-peaked /
    pathological profiles) or the window is degenerate.
    """
    mask = (mjd >= t_start) & (mjd <= t_end)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan

    ob_mjd = mjd[mask]
    net_flux = np.clip(flux[mask] - baseline, 0, None)
    if net_flux.sum() == 0:
        return np.nan, np.nan, np.nan

    cumsum = np.cumsum(net_flux)
    total = cumsum[-1]
    t5_idx = min(np.searchsorted(cumsum, 0.05 * total), len(ob_mjd) - 1)
    t95_idx = min(np.searchsorted(cumsum, 0.95 * total), len(ob_mjd) - 1)
    t5_mjd, t95_mjd = ob_mjd[t5_idx], ob_mjd[t95_idx]

    peak_idx = np.argmax(net_flux)
    t_peak_mjd = ob_mjd[peak_idx]
    if not (t5_mjd <= t_peak_mjd <= t95_mjd):
        return np.nan, np.nan, np.nan

    return t_peak_mjd - t5_mjd, t95_mjd - t_peak_mjd, t_peak_mjd


def compute_duration_stats(outbursts_csv: str, classified_csv: str, outdir: str = ".") -> pd.DataFrame:
    """Compute T90, T_rise, T_decay for every outburst in
    ``outbursts_csv`` (step2_outbursts.csv), caching one light-curve
    fetch per source. Writes and returns the augmented DataFrame.
    """
    ob_df = pd.read_csv(outbursts_csv)
    xrb_df = pd.read_csv(classified_csv)
    xrb_df["maxi_id"] = xrb_df.apply(lambda r: ra_dec_to_maxi_id(r["maxi_ra"], r["maxi_dec"]), axis=1)
    id_map = dict(zip(xrb_df["maxi_name"], xrb_df["maxi_id"]))

    t90_list, t_rise_list, t_decay_list = [], [], []
    lc_cache: dict = {}

    for _, row in ob_df.iterrows():
        maxi_id = id_map.get(row["maxi_name"])
        if maxi_id is None:
            t90_list.append(np.nan); t_rise_list.append(np.nan); t_decay_list.append(np.nan)
            continue
        if maxi_id not in lc_cache:
            lc_cache[maxi_id] = fetch_maxi_lightcurve(maxi_id)
            time.sleep(0.2)
        lc_df = lc_cache[maxi_id]
        if lc_df is None:
            t90_list.append(np.nan); t_rise_list.append(np.nan); t_decay_list.append(np.nan)
            continue

        mjd, flux = lc_df["mjd"].values, lc_df["flux_total"].values
        t90, _, _ = compute_t90(mjd, flux, row["baseline"], row["t_start"], row["t_end"])
        t_rise, t_decay, _ = compute_trise_tdecay(mjd, flux, row["baseline"], row["t_start"], row["t_end"])
        t90_list.append(t90); t_rise_list.append(t_rise); t_decay_list.append(t_decay)

    ob_df["t90_days"] = t90_list
    ob_df["t_rise_days"] = t_rise_list
    ob_df["t_decay_days"] = t_decay_list

    out_path = f"{outdir}/step3_outbursts_t90.csv"
    ob_df.to_csv(out_path, index=False)
    print(f"T90 valid: {ob_df['t90_days'].notna().sum()}/{len(ob_df)}")
    print(f"T_rise/T_decay valid: {ob_df['t_rise_days'].notna().sum()}/{len(ob_df)}")
    print(f"Saved -> {out_path}")
    return ob_df


COLORS = {"BH": "#E63946", "NS": "#457B9D", "unknown": "#6D6875"}


def plot_t90_histogram(ob_df: pd.DataFrame, savepath: str | None = None):
    """Two-panel T90 histogram: BH vs NS, and BH vs NS vs unknown
    (report Figure 6). Logarithmic bins from 5 to 500 days.
    """
    import matplotlib.pyplot as plt

    df = ob_df[ob_df["t90_days"].notna()]
    bins = np.logspace(np.log10(5), np.log10(500), 25)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("XRB Outburst T90 Duration Distribution", fontsize=13, fontweight="bold")

    for ax, cos in [(ax1, ("BH", "NS")), (ax2, ("BH", "NS", "unknown"))]:
        for co in cos:
            vals = df[df["compact_object"] == co]["t90_days"].values
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, alpha=0.6, color=COLORS[co], label=f"{co} (n={len(vals)})",
                    edgecolor="white", linewidth=0.4)
            ax.axvline(np.median(vals), color=COLORS[co], lw=1.5, ls="--", alpha=0.9)
        ax.set_xlabel("T90 (days)"); ax.set_ylabel("Number of outbursts")
        ax.legend(fontsize=9); ax.grid(alpha=0.25, which="both", lw=0.4)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def plot_rise_decay_histograms(ob_df: pd.DataFrame, savepath: str | None = None):
    """2x2 T_rise/T_decay histogram panel (report Figure 7). Logarithmic
    bins from 1 to 300 days.
    """
    import matplotlib.pyplot as plt

    df = ob_df[ob_df["t_rise_days"].notna() & ob_df["t_decay_days"].notna()]
    bins = np.logspace(np.log10(1), np.log10(300), 25)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("XRB Outburst Rise & Decay Time Distribution", fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], "t_rise_days", "T_rise -- BH vs NS", False),
        (axes[0, 1], "t_decay_days", "T_decay -- BH vs NS", False),
        (axes[1, 0], "t_rise_days", "T_rise -- BH vs NS vs Unk", True),
        (axes[1, 1], "t_decay_days", "T_decay -- BH vs NS vs Unk", True),
    ]
    for ax, col, title, inc_unk in panels:
        cos = ("BH", "NS", "unknown") if inc_unk else ("BH", "NS")
        for co in cos:
            vals = df[df["compact_object"] == co][col].values
            if len(vals) == 0:
                continue
            med = np.median(vals)
            ax.hist(vals, bins=bins, alpha=0.6, color=COLORS[co], edgecolor="white", linewidth=0.4,
                    label=f"{co} median={med:.0f}d")
            ax.axvline(med, color=COLORS[co], lw=1.5, ls="--", alpha=0.9)
        ax.set_xlabel("Duration (days)"); ax.set_ylabel("Number of outbursts")
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.25, which="both", lw=0.4)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outbursts", default="step2_outbursts.csv")
    parser.add_argument("--classified", default="step1b_maxi_xrb_classified.csv")
    parser.add_argument("--outdir", default="data/duration_stats")
    args = parser.parse_args()

    df = compute_duration_stats(args.outbursts, args.classified, args.outdir)
    plot_t90_histogram(df, savepath=f"{args.outdir}/t90_histogram.png")
    plot_rise_decay_histograms(df, savepath=f"{args.outdir}/rise_decay_histogram.png")
