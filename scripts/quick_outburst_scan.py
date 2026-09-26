#!/usr/bin/env python3
"""
Quick outburst scan: a minimal, easy-to-tweak 3-rule detector for
eyeballing one source, with no dependency on the JSON database or the
classified catalogue. Good starting point to fork for a one-off check.

The rule, for a point i to count as an outburst peak:
    1. Its flux is above threshold (SIGMA_MULT * median(flux)).
    2. It is the highest point within [-WINDOW_BACK, +WINDOW_FORWARD] days.
    3. Every point in that same window is also above threshold.

Each rule is its own tiny function below -- change one without
touching the others.

Usage:
    python scripts/quick_outburst_scan.py J1911+005
    python scripts/quick_outburst_scan.py J1911+005 --sigma-mult 4 --window-back 15 --window-forward 25
"""
from __future__ import annotations

import argparse

import numpy as np
import requests

MAXI_BASE = "https://maxi.riken.jp/star_data"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; XRB-pipeline/1.0)"}


def fetch_maxi_lightcurve(maxi_id: str):
    url = f"{MAXI_BASE}/{maxi_id}/{maxi_id}_g_lc_1day_all.dat"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        print(f"[ERROR] HTTP {resp.status_code} for {maxi_id}")
        return np.array([]), np.array([])

    lines = [l for l in resp.text.strip().split("\n") if l.strip() and not l.startswith("!")]
    parsed = []
    for l in lines:
        parts = l.split()
        if len(parts) >= 2:
            try:
                parsed.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

    if len(parsed) < 10:
        print(f"[ERROR] Too few points for {maxi_id} ({len(parsed)})")
        return np.array([]), np.array([])

    return np.array([p[0] for p in parsed]), np.array([p[1] for p in parsed])


def get_threshold(flux, sigma_mult):
    """Threshold = sigma_mult * median(flux). Change the formula here only."""
    return sigma_mult * np.nanmedian(flux)


def get_window_mask(mjd, center_mjd, back_days, forward_days):
    """Boolean mask: True for every point within [center-back, center+forward] days."""
    return (mjd >= center_mjd - back_days) & (mjd <= center_mjd + forward_days)


def is_highest_in_window(flux, mask, i):
    return flux[i] == np.nanmax(flux[mask])


def all_points_above_threshold(flux, mask, threshold):
    return np.all(flux[mask] > threshold)


def detect_outbursts(mjd, flux, sigma_mult=5.0, window_back=20, window_forward=30):
    """Run the 3-rule check on every point. Returns (outburst_indices, threshold)."""
    threshold = get_threshold(flux, sigma_mult)
    outburst_indices = []

    for i in range(len(mjd)):
        if flux[i] <= threshold:
            continue
        mask = get_window_mask(mjd, mjd[i], window_back, window_forward)
        if not is_highest_in_window(flux, mask, i):
            continue
        if not all_points_above_threshold(flux, mask, threshold):
            continue
        outburst_indices.append(i)

    return outburst_indices, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("maxi_id", help="MAXI J-name, e.g. J1911+005")
    parser.add_argument("--sigma-mult", type=float, default=5.0)
    parser.add_argument("--window-back", type=float, default=20.0, help="days")
    parser.add_argument("--window-forward", type=float, default=30.0, help="days")
    parser.add_argument("--save", help="Save the plot to this path")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    mjd, flux = fetch_maxi_lightcurve(args.maxi_id)
    if len(mjd) == 0:
        return

    idx, threshold = detect_outbursts(mjd, flux, args.sigma_mult, args.window_back, args.window_forward)
    print(f"{args.maxi_id}: threshold={threshold:.4f}  outbursts found={len(idx)}")
    print("Outburst MJDs:", [round(mjd[i], 1) for i in idx])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(mjd, flux, color="steelblue", lw=0.8, label="MAXI flux")
    ax.axhline(threshold, color="orange", ls="--", lw=1, label=f"threshold ({args.sigma_mult}x median)")
    if idx:
        ax.scatter(mjd[idx], flux[idx], color="red", s=70, marker="^", zorder=5,
                   label=f"Outburst peak(s) ({len(idx)})")
    ax.set_xlabel("MJD"); ax.set_ylabel("Flux (ct/s/cm^2)")
    ax.set_title(f"{args.maxi_id} -- quick outburst scan")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved -> {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
