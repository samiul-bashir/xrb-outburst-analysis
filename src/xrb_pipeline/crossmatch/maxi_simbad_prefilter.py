"""
Step 1 (report section 3.2, upstream of classify_compact_objects.py):
cross-match the full MAXI GSC source list against SIMBAD, keeping only
X-ray-binary object types. This is the prefilter that produces
step1_maxi_xrb_sources.csv, the input classify_compact_objects.py reads.

Unlike the per-source `star_data` light-curve endpoint used everywhere
else in this pipeline, this queries MAXI's all-sky *source list* page
(an HTML table), not a single source's data.

Usage:
    python -m xrb_pipeline.crossmatch.maxi_simbad_prefilter --output step1_maxi_xrb_sources.csv
"""
from __future__ import annotations

import time

import pandas as pd
import requests
import astropy.units as u
from astropy.coordinates import SkyCoord

MAXI_SOURCE_LIST_URL = "http://maxi.riken.jp/sourcelist.html"
SEARCH_RADIUS = 2.0 * u.arcmin  # generous: MAXI position accuracy is a few arcmin for faint sources
XRB_TYPES = ["XB", "LXB", "HXB"]


def fetch_maxi_source_list(url: str = MAXI_SOURCE_LIST_URL, timeout: int = 30) -> pd.DataFrame:
    """Download and parse MAXI's all-sky source list HTML table. Each
    row: source name, and "RA, Dec" in decimal degrees, comma-separated.
    Returns a DataFrame [maxi_name, maxi_ra, maxi_dec].
    """
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Could not download MAXI source list: HTTP {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")

    sources = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        source_name = cells[0].get_text(strip=True)
        coord_text = cells[1].get_text(strip=True)
        if not source_name or coord_text in ("", "--"):
            continue
        try:
            ra_str, dec_str = coord_text.split(",")
            ra, dec = float(ra_str.strip()), float(dec_str.strip())
        except (ValueError, IndexError):
            continue
        sources.append({"maxi_name": source_name, "maxi_ra": ra, "maxi_dec": dec})

    return pd.DataFrame(sources)


def simbad_xrb_prefilter(maxi_df: pd.DataFrame, search_radius=SEARCH_RADIUS,
                          xrb_types=XRB_TYPES, delay: float = 0.3) -> pd.DataFrame:
    """For each MAXI source, cone-search SIMBAD within ``search_radius``
    and keep it only if a returned object's otype is in ``xrb_types``
    (first XRB-typed match wins). Returns a DataFrame
    [maxi_name, maxi_ra, maxi_dec, simbad_name, simbad_otype, simbad_ra,
    simbad_dec] -- simbad_ra/dec are SIMBAD's sexagesimal strings.
    """
    from astroquery.simbad import Simbad

    custom_simbad = Simbad()
    custom_simbad.add_votable_fields("otype")

    matched = []
    total = len(maxi_df)
    for i, row in maxi_df.iterrows():
        if i % 20 == 0:
            print(f"  [{i + 1}/{total}] Querying: {row['maxi_name']}")

        coord = SkyCoord(ra=row["maxi_ra"] * u.degree, dec=row["maxi_dec"] * u.degree, frame="icrs")
        try:
            result = custom_simbad.query_region(coord, radius=search_radius)
        except Exception as e:
            print(f"  WARNING: SIMBAD query failed for {row['maxi_name']} -> {e}")
            time.sleep(2.0)
            continue

        if result is None:
            time.sleep(delay)
            continue

        for simbad_row in result:
            if simbad_row["otype"] in xrb_types:
                matched.append(
                    {
                        "maxi_name": row["maxi_name"], "maxi_ra": row["maxi_ra"], "maxi_dec": row["maxi_dec"],
                        "simbad_name": simbad_row["main_id"], "simbad_otype": simbad_row["otype"],
                        "simbad_ra": simbad_row["ra"], "simbad_dec": simbad_row["dec"],
                    }
                )
                break  # first XRB-typed match is enough for this source

        time.sleep(delay)

    return pd.DataFrame(matched)


def apply_manual_corrections(xrb_df: pd.DataFrame, additions: list[dict] | None = None,
                              removals: list[str] | None = None) -> pd.DataFrame:
    """SIMBAD sometimes types a known XRB as generic 'X' (missed by the
    otype filter above), and brand-new transients may not have an otype
    yet at all. Use this to patch the catalogue by hand before saving.

    ``additions``: list of dicts with the same columns as ``xrb_df``.
    ``removals``: list of ``maxi_name`` values to drop (e.g. known bad
    matches, or persistent sources you want excluded up front).
    """
    if additions:
        xrb_df = pd.concat([xrb_df, pd.DataFrame(additions)], ignore_index=True)
    if removals:
        xrb_df = xrb_df[~xrb_df["maxi_name"].isin(removals)].reset_index(drop=True)
    return xrb_df


def run_pipeline(output_csv: str = "step1_maxi_xrb_sources.csv") -> pd.DataFrame:
    print("Downloading MAXI source list...")
    maxi_df = fetch_maxi_source_list()
    print(f"Total MAXI sources parsed: {len(maxi_df)}")

    print(f"Starting SIMBAD queries for {len(maxi_df)} MAXI sources "
          f"(this takes a while -- polite delay between queries)...")
    xrb_df = simbad_xrb_prefilter(maxi_df)
    print(f"\nXRBs confirmed by SIMBAD: {len(xrb_df)}")
    print(xrb_df["simbad_otype"].value_counts())

    xrb_df.to_csv(output_csv, index=False)
    print(f"\nSaved -> {output_csv}  ({len(xrb_df)} rows)")
    return xrb_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="step1_maxi_xrb_sources.csv")
    args = parser.parse_args()
    run_pipeline(args.output)
