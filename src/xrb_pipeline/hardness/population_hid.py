"""
Population-level hardness analysis: HR-vs-time diagnostic plots per
source, and one combined Hardness-Intensity Diagram overlaying every
BH/NS outburst track in the catalogue -- complementary to
`hardness/hid.py`, which inspects one outburst (X-ray + optical) at a
time. This module works purely in X-ray (MAXI sub-bands) across the
whole `step4_master_catalogue.csv`, color-coded by compact object, with
time-direction arrows along each track.

Input: step4_master_catalogue.csv (from duration_stats, extended with
compact_object/source_type/outburst window columns from detection).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

MAXI_BASE = "https://maxi.riken.jp/star_data"
HEADERS = {"User-Agent": "Mozilla/5.0"}
EPSILON = 1e-6

HR_SMOOTH_WIN = 7      # rolling median window (days)
HR_CLIP_MAX = 10.0     # clip runaway HR (noise spikes in faint sources)
CO_KEEP = {"BH", "NS"}  # unknown compact_object sources are skipped

CO_COLORS = {"BH": "#E63946", "NS": "#457B9D", "HXB": "#2D6A4F"}


def fetch_maxi_lc_allbands(maxi_id: str) -> pd.DataFrame | None:
    """Fetch MAXI 1-day LC with all 4 bands + errors, or None on failure."""
    url = f"{MAXI_BASE}/{maxi_id}/{maxi_id}_g_lc_1day_all.dat"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
    except Exception as e:
        print(f"    Network error ({maxi_id}): {e}")
        return None
    if resp.status_code != 200:
        print(f"    HTTP {resp.status_code}  ({maxi_id})")
        return None

    rows = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("!"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            rows.append(
                {
                    "mjd": float(parts[0]), "flux_total": float(parts[1]), "err_total": float(parts[2]),
                    "flux_soft": float(parts[3]), "err_soft": float(parts[4]),
                    "flux_med": float(parts[5]), "err_med": float(parts[6]),
                    "flux_hard": float(parts[7]), "err_hard": float(parts[8]),
                }
            )
        except ValueError:
            continue
    if len(rows) < 10:
        return None
    return pd.DataFrame(rows).sort_values("mjd").reset_index(drop=True)


def compute_hr(lc_df: pd.DataFrame, smooth_win: int = HR_SMOOTH_WIN, clip_max: float = HR_CLIP_MAX) -> pd.DataFrame:
    """Add hr/hr_err/hr_smooth columns. HR = flux_med / flux_soft
    (4-10 keV / 2-4 keV), with propagated fractional error; points where
    either band is non-positive are NaN.
    """
    lc = lc_df.copy()
    valid = (lc["flux_soft"] > 0) & (lc["flux_med"] > 0)

    lc["hr"] = np.where(valid, lc["flux_med"] / lc["flux_soft"], np.nan)
    lc["hr_err"] = np.where(
        valid,
        lc["hr"] * np.sqrt((lc["err_med"] / lc["flux_med"].clip(EPSILON)) ** 2
                            + (lc["err_soft"] / lc["flux_soft"].clip(EPSILON)) ** 2),
        np.nan,
    )
    lc["hr"] = lc["hr"].clip(0, clip_max)
    lc["hr_err"] = lc["hr_err"].clip(0, clip_max)
    lc["hr_smooth"] = lc["hr"].rolling(smooth_win, center=True, min_periods=max(1, smooth_win // 3)).median()
    return lc


def build_hr_catalogue(master_csv: str, co_keep: set = CO_KEEP, delay: float = 0.25):
    """Load the master outburst catalogue, filter to ``co_keep`` compact
    objects, fetch+cache one light curve per unique maxi_id, and append
    per-outburst HR summary stats (hr_median, hr_at_peak, hr_min,
    hr_max, hr_n).

    Returns (ob_df, lc_store) -- ob_df is the augmented outburst table,
    lc_store maps maxi_id -> HR-enriched light-curve DataFrame.
    """
    master = pd.read_csv(master_csv)
    print(f"Master catalogue: {len(master)} outbursts, {master['maxi_name'].nunique()} sources")

    ob_df = master[master["compact_object"].isin(co_keep)].copy().reset_index(drop=True)
    print(f"Kept {len(ob_df)}/{len(master)} outbursts (compact_object in {co_keep})")

    lc_store: dict[str, pd.DataFrame] = {}
    for maxi_id in ob_df["maxi_id"].unique():
        lc = fetch_maxi_lc_allbands(maxi_id)
        time.sleep(delay)
        if lc is None:
            print(f"  [!] FAILED: {maxi_id}")
            continue
        lc_store[maxi_id] = compute_hr(lc)
        print(f"  OK {maxi_id:18s} pts={len(lc):4d} valid_HR={lc_store[maxi_id]['hr'].notna().sum():4d}")

    hr_rows = []
    for _, row in ob_df.iterrows():
        lc = lc_store.get(row["maxi_id"])
        if lc is None:
            hr_rows.append({"hr_median": np.nan, "hr_at_peak": np.nan, "hr_min": np.nan, "hr_max": np.nan, "hr_n": 0})
            continue
        mask = (lc["mjd"] >= row["t_start"]) & (lc["mjd"] <= row["t_end"])
        seg = lc[mask].dropna(subset=["hr"])
        if seg.empty:
            hr_rows.append({"hr_median": np.nan, "hr_at_peak": np.nan, "hr_min": np.nan, "hr_max": np.nan, "hr_n": 0})
            continue
        pk_idx = seg["flux_total"].idxmax()
        hr_rows.append(
            {
                "hr_median": round(seg["hr"].median(), 4), "hr_at_peak": round(seg.loc[pk_idx, "hr"], 4),
                "hr_min": round(seg["hr"].min(), 4), "hr_max": round(seg["hr"].max(), 4),
                "hr_n": int(seg["hr"].notna().sum()),
            }
        )

    ob_df = pd.concat([ob_df, pd.DataFrame(hr_rows, index=ob_df.index)], axis=1)
    print(f"HR valid for {ob_df['hr_median'].notna().sum()}/{len(ob_df)} outbursts")
    return ob_df, lc_store


def plot_hr_vs_time_per_source(ob_df: pd.DataFrame, lc_store: dict, savedir: str | None = None):
    """Two-panel plot per source: flux on top, HR (raw + smoothed) below,
    with detected outburst windows shaded and the NS quiescent-HR veto
    line overlaid where available.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    for name, grp in ob_df.groupby("maxi_name"):
        maxi_id = grp["maxi_id"].iloc[0]
        lc = lc_store.get(maxi_id)
        if lc is None:
            print(f"  [skip] {name} -- no LC")
            continue

        co = grp["compact_object"].iloc[0]
        stype = grp["source_type"].iloc[0]
        col = CO_COLORS.get(co, "#888888")

        fig = plt.figure(figsize=(15, 6), dpi=110)
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.05)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        ax1.plot(lc["mjd"], lc["flux_total"], color="steelblue", lw=0.7, alpha=0.85, zorder=2)
        ax1.axhline(0, color="black", lw=0.4, alpha=0.3)
        ax1.set_ylabel("Flux (ph/s/cm^2)")
        ax1.set_title(f"{name}  [{maxi_id}]  {co}/{stype}  ({len(grp)} outburst(s))", fontweight="bold")
        ax1.grid(alpha=0.2, lw=0.4)

        ax2.errorbar(lc["mjd"], lc["hr"], yerr=lc["hr_err"], fmt=".", color="mediumpurple",
                     ms=1.5, elinewidth=0.4, alpha=0.45, zorder=2)
        ax2.plot(lc["mjd"], lc["hr_smooth"], color=col, lw=1.1, alpha=0.9, zorder=3,
                 label=f"HR ({HR_SMOOTH_WIN}d median)")
        ax2.set_ylabel("HR (4-10)/(2-4)"); ax2.set_xlabel("MJD")
        ax2.set_ylim(-0.1, 6); ax2.axhline(0, color="black", lw=0.4, alpha=0.3)
        ax2.grid(alpha=0.2, lw=0.4)

        for k, (_, ob) in enumerate(grp.iterrows()):
            ax1.axvspan(ob["t_start"], ob["t_end"], alpha=0.15, color="crimson", zorder=1)
            ax1.axvline(ob["t_peak"], color="crimson", lw=0.8, alpha=0.5, zorder=4)
            ax2.axvspan(ob["t_start"], ob["t_end"], alpha=0.15, color="crimson", zorder=1)

            seg = lc[(lc["mjd"] >= ob["t_start"]) & (lc["mjd"] <= ob["t_end"])]
            if not seg.empty:
                pk_i = seg["flux_total"].idxmax()
                hr_val = lc.loc[pk_i, "hr"]
                if not np.isnan(hr_val):
                    ax2.scatter(lc.loc[pk_i, "mjd"], hr_val, color="crimson", s=40, marker="v", zorder=6)
            y_top = ax1.get_ylim()[1]
            ax1.text(ob["t_peak"], y_top * 0.90, f"#{k + 1}", fontsize=7, color="crimson", ha="center", va="top")

        if co == "NS" and "quiescent_hr" in grp.columns:
            q_hr = grp["quiescent_hr"].dropna().median()
            if not np.isnan(q_hr):
                ax2.axhline(q_hr, color="gray", ls="--", lw=0.9, label=f"Quiescent HR={q_hr:.2f}")
                ax2.axhline(q_hr * 0.75, color="red", ls=":", lw=0.9, label=f"Veto line={q_hr * 0.75:.2f}")

        ax2.legend(fontsize=7, loc="upper right", ncol=2)
        plt.setp(ax1.get_xticklabels(), visible=False)
        fig.tight_layout()
        if savedir:
            fig.savefig(f"{savedir}/hr_vs_time_{name.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()


def plot_population_hid(ob_df: pd.DataFrame, lc_store: dict, savepath: str | None = None):
    """One combined HID overlaying every outburst track (X-ray only,
    no optical component). Opacity encodes time (light=early,
    dark=late); arrows show direction of travel. BH tracks are
    expected to sweep right-to-left (hard-to-soft); NS tracks can be
    more complex.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    plotted_labels = set()
    n_tracks = 0

    for _, ob in ob_df.iterrows():
        lc = lc_store.get(ob["maxi_id"])
        if lc is None:
            continue

        co, stype = ob["compact_object"], ob["source_type"]
        col_key = "HXB" if stype == "HXB" else co
        col = CO_COLORS.get(col_key, "#888888")

        mask = (lc["mjd"] >= ob["t_start"]) & (lc["mjd"] <= ob["t_end"])
        seg = lc[mask].dropna(subset=["hr", "flux_total"]).reset_index(drop=True)
        if len(seg) < 3:
            continue

        n = len(seg)
        alphas = np.linspace(0.2, 0.9, n)
        ax.plot(seg["hr"], seg["flux_total"], color=col, lw=0.7, alpha=0.35, zorder=2)

        label = f"{co} ({stype})" if col_key not in plotted_labels else "_nolegend_"
        plotted_labels.add(col_key)
        for i, pt in seg.iterrows():
            ax.scatter(pt["hr"], pt["flux_total"], color=col, s=9, alpha=float(alphas[i]), zorder=3,
                       label=label if i == 0 else "_nolegend_")

        if n >= 5:
            mid = n // 2
            dx = float(seg["hr"].iloc[mid + 1] - seg["hr"].iloc[mid - 1])
            dy = float(seg["flux_total"].iloc[mid + 1] - seg["flux_total"].iloc[mid - 1])
            ax.annotate(
                "", xy=(float(seg["hr"].iloc[mid]) + dx * 0.02, float(seg["flux_total"].iloc[mid]) + dy * 0.02),
                xytext=(float(seg["hr"].iloc[mid]), float(seg["flux_total"].iloc[mid])),
                arrowprops=dict(arrowstyle="->", color=col, lw=0.9), zorder=4,
            )
        n_tracks += 1

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=k)
        for k, c in CO_COLORS.items()
    ]
    ax.legend(handles=handles, title="Type", fontsize=9, title_fontsize=9, loc="upper right")
    ax.set_xlabel("Hardness Ratio (4-10 keV) / (2-4 keV)")
    ax.set_ylabel("Intensity 2-20 keV (ph/s/cm^2)")
    ax.set_title(f"Hardness-Intensity Diagram -- MAXI XRBs ({n_tracks} outburst tracks, BH+NS only)", fontweight="bold")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.grid(alpha=0.18, lw=0.4)
    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"{n_tracks} tracks plotted.")


def print_hr_summary(ob_df: pd.DataFrame):
    """Median HR by compact_object and by source_type, for outbursts
    with a valid hr_median.
    """
    summary = ob_df[ob_df["hr_median"].notna()].reset_index(drop=True)
    print(f"Outbursts with valid HR: {len(summary)}")
    print("\nMedian HR by compact object:")
    for co, grp in summary.groupby("compact_object"):
        print(f"  {co:5s}  hr_median={grp['hr_median'].median():.3f}  "
              f"hr_at_peak={grp['hr_at_peak'].median():.3f}  n={len(grp)}")
    print("\nMedian HR by source_type:")
    for st, grp in summary.groupby("source_type"):
        print(f"  {st:10s}  hr_median={grp['hr_median'].median():.3f}  n={len(grp)}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="step4_master_catalogue.csv")
    parser.add_argument("--outdir", default="data/hardness")
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    ob_df, lc_store = build_hr_catalogue(args.master)
    ob_df.to_csv(f"{args.outdir}/step5_outbursts_hr.csv", index=False)
    plot_hr_vs_time_per_source(ob_df, lc_store, savedir=args.outdir)
    plot_population_hid(ob_df, lc_store, savepath=f"{args.outdir}/step5_HID.png")
    print_hr_summary(ob_df)
