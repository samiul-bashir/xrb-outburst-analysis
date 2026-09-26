"""
BAT -> MAXI -> SIMBAD -> ZTF cross-matching pipeline (report section 3.1).

Identifies confirmed X-ray binaries simultaneously detected by Swift/BAT
(14-195 keV), monitored by MAXI (2-20 keV), and with an optical
counterpart in ZTF (via the ALeRCE broker). Four sequential steps:

    1. BAT -> MAXI      : does a MAXI light-curve page exist for this
                           BAT-derived sky position?
    2. MAXI -> SIMBAD    : resolve precise coordinates + object type.
    3. SIMBAD -> ZTF     : 5-arcsec cone search via ALeRCE for an optical
                           counterpart.
    4. XRB-type filter   : keep only SIMBAD otype LXB or HXB with a ZTF
                            match.

Run as a script (each step writes/reads an intermediate CSV so the
pipeline can be resumed) or import the individual functions.

    python -m xrb_pipeline.crossmatch.bat_maxi_simbad_ztf \\
        --bat-fits BAT_catalog.fits --outdir data/crossmatch
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

MAXI_BASE = "https://maxi.riken.jp/star_data"
MAXI_DELAY = 0.3  # seconds between requests (be polite to the server)
MAXI_TIMEOUT = 5
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )
}
ZTF_CONE_RADIUS_ARCSEC = 5.0


def ra_dec_to_maxi_id(ra_deg: float, dec_deg: float) -> str:
    """Convert decimal RA/Dec to a MAXI J-name: J + HHMM + sign + DDd
    (hours/minutes and degrees/tenths, all truncated not rounded).
    """
    ra_hours_total = ra_deg / 15.0
    hh = int(ra_hours_total)
    mm = int((ra_hours_total - hh) * 60)
    ra_str = f"{hh:02d}{mm:02d}"

    sign = "+" if dec_deg >= 0 else "-"
    abs_dec = abs(dec_deg)
    dd = int(abs_dec)
    d = int((abs_dec - dd) * 10)
    dec_str = f"{dd:02d}{d}"

    return f"J{ra_str}{sign}{dec_str}"


def load_bat_catalog(fits_path: str) -> pd.DataFrame:
    """Load the Swift/BAT hard X-ray catalog FITS table into a DataFrame,
    decoding any byte-string columns.
    """
    bat_table = Table.read(fits_path, hdu=1)
    df = bat_table.to_pandas()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8").strip()
                if isinstance(x, bytes)
                else (x.strip() if isinstance(x, str) else x)
            )
    return df


def step1_bat_to_maxi(bat_df: pd.DataFrame) -> pd.DataFrame:
    """For every BAT source, derive its MAXI J-name and check (via HTTP
    200) whether MAXI has a light-curve page for it. Returns the matched
    subset with columns [BAT_Catalogue_Name, RA_BAT, DEC_BAT, MAXI_Name].
    """
    maxi_matches = []
    for i, row in bat_df.iterrows():
        maxi_id = ra_dec_to_maxi_id(row["RA_OBJ"], row["DEC_OBJ"])
        maxi_url = f"{MAXI_BASE}/{maxi_id}/{maxi_id}.html"
        try:
            resp = requests.get(maxi_url, headers=HEADERS, timeout=MAXI_TIMEOUT)
            matched = resp.status_code == 200
        except Exception as exc:
            print(f"  [WARNING] MAXI request failed for {row['NAME']}: {exc}")
            matched = False
        maxi_matches.append(matched)
        if (i + 1) % 50 == 0:
            print(f"Checking MAXI: {i + 1}/{len(bat_df)}...")
        time.sleep(MAXI_DELAY)

    bat_df = bat_df.copy()
    bat_df["MAXI_match"] = maxi_matches
    matched_df = bat_df[bat_df["MAXI_match"]].copy()
    matched_df["MAXI_NAME"] = matched_df.apply(
        lambda r: ra_dec_to_maxi_id(r["RA_OBJ"], r["DEC_OBJ"]), axis=1
    )
    out = matched_df[["NAME", "RA_OBJ", "DEC_OBJ", "MAXI_NAME"]].copy()
    out.columns = ["BAT_Catalogue_Name", "RA_BAT", "DEC_BAT", "MAXI_Name"]
    print(f"MAXI check complete. Matches: {len(out)}/{len(bat_df)}")
    return out


def step2_maxi_to_simbad(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve each BAT source name against SIMBAD for a precise optical
    identifier, object type, and coordinates. SIMBAD coordinates replace
    the coarser BAT position in all downstream steps.
    """
    from astroquery.simbad import Simbad

    custom_simbad = Simbad()
    custom_simbad.add_votable_fields("otype")

    ids, otypes, ras, decs = [], [], [], []
    for index, row in matched_df.iterrows():
        name = row["BAT_Catalogue_Name"]
        try:
            result = custom_simbad.query_object(name)
            if result is not None and len(result) > 0:
                cols = result.colnames
                main_id = result["main_id"][0] if "main_id" in cols else result.get("MAIN_ID", [None])[0]
                otype = result["otype"][0] if "otype" in cols else result.get("OTYPE", [None])[0]
                ra = float(result["ra"][0]) if "ra" in cols else float(result["RA"][0])
                dec = float(result["dec"][0]) if "dec" in cols else float(result["DEC"][0])
                ids.append(str(main_id).strip() if main_id is not None else None)
                otypes.append(str(otype).strip() if otype is not None else None)
                ras.append(ra)
                decs.append(dec)
            else:
                ids.append(None); otypes.append(None); ras.append(None); decs.append(None)
        except Exception as error:
            print(f"  [SKIP] {name}: {error}")
            ids.append(None); otypes.append(None); ras.append(None); decs.append(None)

        if (index + 1) % 30 == 0:
            print(f"  Progress: {index + 1}/{len(matched_df)}...")
        time.sleep(0.1)

    out = matched_df.copy()
    out["SIMBAD_ID"] = ids
    out["SIMBAD_Otype"] = otypes
    out["SIMBAD_RA"] = ras
    out["SIMBAD_DEC"] = decs
    return out


def step3_simbad_to_ztf(df: pd.DataFrame, radius_arcsec: float = ZTF_CONE_RADIUS_ARCSEC) -> pd.DataFrame:
    """Cone-search ZTF (via ALeRCE) around each SIMBAD position; keep the
    nearest match within ``radius_arcsec``. Drops rows with no SIMBAD
    coordinates first.
    """
    from alerce.core import Alerce

    client = Alerce()
    valid_df = df.dropna(subset=["SIMBAD_RA", "SIMBAD_DEC"]).copy()
    print(f"Found {len(valid_df)} sources with valid coordinates to target on ZTF.")

    ztf_ids, ztf_ras, ztf_decs, seps_out = [], [], [], []
    for index, row in valid_df.iterrows():
        ra_s, dec_s = row["SIMBAD_RA"], row["SIMBAD_DEC"]
        center = SkyCoord(ra=ra_s, dec=dec_s, unit=(u.deg, u.deg), frame="icrs")
        try:
            matches = client.query_objects(survey="ztf", ra=ra_s, dec=dec_s, radius=radius_arcsec)
            if matches is not None and len(matches) > 0:
                ztf_coords = SkyCoord(ra=matches["meanra"].values, dec=matches["meandec"].values, unit="deg")
                seps = center.separation(ztf_coords).arcsec
                best_idx = int(np.argmin(seps))
                best = matches.iloc[best_idx]
                ztf_ids.append(best.get("oid")); ztf_ras.append(float(best["meanra"]))
                ztf_decs.append(float(best["meandec"])); seps_out.append(float(seps[best_idx]))
            else:
                ztf_ids.append(None); ztf_ras.append(None); ztf_decs.append(None); seps_out.append(None)
        except Exception as e:
            print(f"  [ERROR] ZTF cross-match failed for {row['BAT_Catalogue_Name']}: {e}")
            ztf_ids.append(None); ztf_ras.append(None); ztf_decs.append(None); seps_out.append(None)

        if len(ztf_ids) % 30 == 0:
            print(f"  Progress: {len(ztf_ids)}/{len(valid_df)} processed...")
        time.sleep(0.1)

    valid_df["ZTF_Object_ID"] = ztf_ids
    valid_df["ZTF_RA"] = ztf_ras
    valid_df["ZTF_DEC"] = ztf_decs
    valid_df["SIMBAD_ZTF_Sep_Arcsec"] = seps_out
    return valid_df


def step4_xrb_type_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only SIMBAD-typed X-ray binaries (LXB or HXB) with a
    successful ZTF match -- the master catalog for outburst analysis.
    """
    return df[df["SIMBAD_Otype"].isin(["LXB", "HXB"]) & df["ZTF_Object_ID"].notna()].copy()


def run_pipeline(bat_fits_path: str, outdir: str = ".") -> pd.DataFrame:
    """Run the full 4-step pipeline, writing an intermediate CSV after
    each stage so the run can be resumed. Returns the final master
    catalog DataFrame.
    """
    import os

    os.makedirs(outdir, exist_ok=True)
    bat_df = load_bat_catalog(bat_fits_path)

    matched = step1_bat_to_maxi(bat_df)
    matched.to_csv(f"{outdir}/BAT_MAXI_matched_sources.csv", index=False)

    with_simbad = step2_maxi_to_simbad(matched)
    with_simbad.to_csv(f"{outdir}/BAT_MAXI_SIMBAD_coords.csv", index=False)

    with_ztf = step3_simbad_to_ztf(with_simbad)
    with_ztf.to_csv(f"{outdir}/BAT_MAXI_SIMBAD_ZTF_match.csv", index=False)

    xrb_df = step4_xrb_type_filter(with_ztf)
    xrb_df.to_csv(f"{outdir}/XRB_BAT_MAXI_SIMBAD_ZTF_final.csv", index=False)

    print(f"\nFinal XRB catalog: {len(xrb_df)} sources -> {outdir}/XRB_BAT_MAXI_SIMBAD_ZTF_final.csv")
    return xrb_df


def print_funnel_summary(bat_fits_path: str, outdir: str = "."):
    """Print the Table-2-style cross-match funnel counts, reading the
    intermediate CSVs already written by ``run_pipeline``.
    """
    bat_df = load_bat_catalog(bat_fits_path)
    maxi_df = pd.read_csv(f"{outdir}/BAT_MAXI_matched_sources.csv")
    simbad_df = pd.read_csv(f"{outdir}/BAT_MAXI_SIMBAD_coords.csv")
    ztf_df = pd.read_csv(f"{outdir}/BAT_MAXI_SIMBAD_ZTF_match.csv")
    xrb_df = pd.read_csv(f"{outdir}/XRB_BAT_MAXI_SIMBAD_ZTF_final.csv")

    n_simbad = simbad_df[["SIMBAD_RA", "SIMBAD_DEC"]].notna().all(axis=1).sum()
    n_ztf = ztf_df["ZTF_Object_ID"].notna().sum()

    print("=" * 45)
    print(f"{'Step':<6} {'Stage':<35} {'Sources':>7}")
    print("=" * 45)
    print(f"{'0':<6} {'BAT input catalog':<35} {len(bat_df):>7}")
    print(f"{'1':<6} {'BAT n MAXI (HTTP 200 match)':<35} {len(maxi_df):>7}")
    print(f"{'2':<6} {'+ SIMBAD resolution':<35} {n_simbad:>7}")
    print(f"{'3':<6} {'+ ZTF counterpart (5 arcsec)':<35} {n_ztf:>7}")
    print(f"{'4':<6} {'+ XRB filter (LXB or HXB)':<35} {len(xrb_df):>7}")
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bat-fits", required=True, help="Path to Swift/BAT catalog FITS file")
    parser.add_argument("--outdir", default="data/crossmatch", help="Output directory for intermediate CSVs")
    args = parser.parse_args()
    run_pipeline(args.bat_fits, args.outdir)
    print_funnel_summary(args.bat_fits, args.outdir)
