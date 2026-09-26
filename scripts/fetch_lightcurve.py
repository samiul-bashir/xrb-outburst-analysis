#!/usr/bin/env python3
"""
Fetch and plot a single MAXI light curve by source ID. No CSV, no
database, no config -- just point it at a MAXI J-name and look.

Usage:
    python scripts/fetch_lightcurve.py J1911+005
    python scripts/fetch_lightcurve.py J1911+005 --save aql_x1.png
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import requests

MAXI_BASE = "https://maxi.riken.jp/star_data"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; XRB-pipeline/1.0)"}


def fetch_maxi_lightcurve(maxi_id: str):
    url = f"{MAXI_BASE}/{maxi_id}/{maxi_id}_g_lc_1day_all.dat"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        print(f"[ERROR] HTTP {resp.status_code} for {maxi_id} -- check the source ID.")
        sys.exit(1)

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
        sys.exit(1)

    mjd = np.array([p[0] for p in parsed])
    flux = np.array([p[1] for p in parsed])
    return mjd, flux


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("maxi_id", help="MAXI J-name, e.g. J1911+005")
    parser.add_argument("--save", help="Save the plot to this path instead of (or as well as) showing it")
    parser.add_argument("--no-show", action="store_true", help="Don't open an interactive window")
    args = parser.parse_args()

    mjd, flux = fetch_maxi_lightcurve(args.maxi_id)
    print(f"{args.maxi_id}: {len(mjd)} points, MJD {mjd.min():.1f} -> {mjd.max():.1f}, "
          f"flux {flux.min():.4f} -> {flux.max():.4f} ct/s/cm^2")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(mjd, flux, color="steelblue", lw=0.8)
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux (ct/s/cm^2)")
    ax.set_title(f"{args.maxi_id} -- MAXI light curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved -> {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
