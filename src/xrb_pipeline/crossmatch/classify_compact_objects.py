"""
Classify MAXI XRB candidates as BH/NS and persistent/transient, using
SIMBAD (upstream pre-filter, not included here), BlackCAT, and the Liu et
al. LMXB/HMXB VizieR catalogs (report section 3.2).

Input : step1_maxi_xrb_sources.csv
        7 columns: maxi_name, maxi_ra, maxi_dec, simbad_name,
        simbad_otype, simbad_ra, simbad_dec
        (produced by an upstream SIMBAD-otype pre-filter over the full
        MAXI GSC source list -- not part of this repo; see docs/README
        "Data" section for how to regenerate it.)

Output: step1b_maxi_xrb_classified.csv, with four new columns:
        compact_object   -- BH / NS / unknown
        persistent_flag  -- P / T / unknown
        liu_name         -- matched name from Liu et al. catalog
        blackcat_name    -- matched name from BlackCAT

Classification priority:
    compact_object : BlackCAT -> BH  >  Liu (NS pulsation/burst) -> NS  >  unknown
    persistent_flag: Liu P/T flag    >  known-persistent override  >  BH->T heuristic  >  unknown

Cross-match radius: 5 arcmin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

SEARCH_RADIUS = 5.0 * u.arcmin

# Sources always active -- no outburst episodes. Applied after Liu, so always wins.
KNOWN_PERSISTENT = [
    "Her X-1", "Sco X-1", "Cyg X-2", "Cyg X-1", "Crab",
    "GX 5-1", "GX 349+2", "GX 9+1", "GX 17+2", "GX 9+9",
    "4U 1820-30", "Ser X-1", "4U 1636-53", "4U 1735-44",
    "EXO 0748-676", "GX 13+1", "GX 340+0", "Cir X-1",
    "LMC X-4", "SMC X-1", "4U 1822-37", "X 1822-371",
    "XB 1254-690", "4U 1254-69", "2S 0921-63", "V395 Car",
    "Cyg X-3", "SS 433",
]

# BlackCAT (Corral-Santana et al. 2016 + updates to ~2024).
# Format: (canonical_name, ra_deg, dec_deg, [alternate_names])
BLACKCAT_SOURCES = [
    ("A 0620-00", 95.685, -0.346, ["1A 0620-00", "V616 Mon", "A0620-00", "Nova Mon 75"]),
    ("GS 1124-683", 181.744, -68.678, ["Nova Mus 91", "GU Mus", "GRS 1124-683", "GS1124-683"]),
    ("GRO J0422+32", 65.729, 32.907, ["V518 Per", "GRO0422+32", "Nova Per 92"]),
    ("4U 1543-47", 236.779, -47.671, ["IL Lup", "4U1543-47"]),
    ("XTE J1550-564", 237.692, -56.476, ["V381 Nor", "XTEJ1550-564"]),
    ("XTE J1650-500", 252.825, -49.957, ["V2107 Oph", "XTEJ1650-500"]),
    ("GRO J1655-40", 253.479, -39.845, ["V1033 Sco", "Nova Sco 94", "GROJ1655-40"]),
    ("GX 339-4", 255.706, -48.789, ["V821 Ara", "GX339-4"]),
    ("H 1705-250", 257.152, -25.097, ["Nova Oph 77", "V2291 Oph", "H1705-250"]),
    ("GS 2000+25", 300.587, 25.240, ["QZ Vul", "GS2000+25"]),
    ("V404 Cyg", 306.016, 33.867, ["GS 2023+338", "GS2023+338", "V404Cyg", "GS 2023"]),
    ("GRS 1716-249", 259.964, -25.017, ["GRO J1719-24", "Nova Oph 93", "GRS1716-249"]),
    ("XTE J1859+226", 284.776, 22.701, ["V406 Vul", "XTEJ1859+226"]),
    ("SAX J1819.3-2525", 274.830, -25.423, ["V4641 Sgr", "SAXJ1819.3-2525", "XTE J1819.3-2525", "V4641Sgr"]),
    ("XTE J1908+094", 287.245, 9.419, ["XTEJ1908+094"]),
    ("H 1743-322", 266.575, -32.236, ["IGR J17464-3213", "1H 1743-322", "H1743-322"]),
    ("MAXI J1659-152", 254.763, -15.260, ["Swift J1659.2-1538", "MAXIJ1659-152"]),
    ("Swift J1753.5-0127", 268.367, -1.453, ["SWIFT J1753.5-0127", "SwiftJ1753.5-0127"]),
    ("MAXI J1836-194", 279.016, -19.320, ["Swift J1836.0-1925", "MAXIJ1836-194"]),
    ("MAXI J1820+070", 275.091, 7.185, ["ASASSN-18ey", "MAXIJ1820+070", "Swift J1820.6+0553"]),
    ("MAXI J1348-630", 207.190, -63.004, ["MAXIJ1348-630"]),
    ("MAXI J1535-571", 233.827, -57.181, ["MAXIJ1535-571", "Swift J1535.3-5726"]),
    ("XTE J1752-223", 268.062, -22.361, ["XTEJ1752-223"]),
    ("GRS 1915+105", 288.798, 10.946, ["V1487 Aql", "GRS1915+105"]),
    ("GRO J1719-24", 259.964, -25.017, ["GRO1719-24", "Nova Oph 93"]),
    ("4U 1630-47", 248.511, -47.395, ["4U1630-47", "4U 1630-472"]),
    ("GS 1354-64", 209.175, -64.738, ["BW Cir", "GS1354-64"]),
    ("Swift J1910.2-0546", 287.540, -5.795, ["MAXI J1910-057", "MAXIJ1910-057", "SwiftJ1910.2-0546"]),
    ("MAXI J1631-479", 248.015, -47.960, ["Swift J1631.9-4752", "MAXIJ1631-479"]),
    ("GRS 1739-278", 265.923, -27.825, ["GRS1739-278"]),
    ("EXO 1846-031", 282.245, -3.076, ["EXO1846-031"]),
    ("MAXI J0556-332", 89.228, -33.225, ["MAXIJ0556-332", "Swift J0556.4-3320"]),
    ("GRS 1009-45", 152.951, -45.646, ["MM Vel", "GRS1009-45", "Nova Vel 93"]),
    ("XTE J1719-291", 259.770, -29.194, ["XTEJ1719-291"]),
    ("XTE J2012+381", 303.157, 38.194, ["XTEJ2012+381"]),
    ("Swift J1539.2-6227", 234.800, -62.461, ["SwiftJ1539.2-6227"]),
    ("XTE J1652-453", 253.129, -45.367, ["XTEJ1652-453"]),
    ("MAXI J1543-564", 235.924, -56.481, ["MAXIJ1543-564"]),
    ("IGR J17098-3628", 257.451, -36.477, ["IGRJ17098-3628"]),
    ("XTE J1817-330", 274.261, -33.036, ["XTEJ1817-330"]),
    ("IGR J17177-3656", 259.436, -36.943, ["IGRJ17177-3656"]),
    ("Swift J1745-26", 266.295, -26.403, ["Swift J174510.8-262411", "SwiftJ1745-26"]),
    ("MAXI J1807-132", 271.836, -13.232, ["MAXIJ1807-132"]),
    ("MAXI J1803-298", 270.850, -29.840, ["MAXIJ1803-298", "Swift J1803.4-2938"]),
    ("Swift J1728.9-3613", 262.246, -36.220, ["SwiftJ1728.9-3613"]),
    ("MAXI J1727-203", 261.992, -20.336, ["MAXIJ1727-203", "Swift J1727.8-2035"]),
    ("Swift J1658.2-4242", 254.565, -42.699, ["SwiftJ1658.2-4242"]),
    ("4U 1957+11", 300.067, 11.712, ["V1408 Aql", "4U1957+11"]),
    ("MAXI J1848-015", 282.077, -1.583, ["MAXIJ1848-015"]),
    ("AT2019wey", 111.856, 62.322, ["AT2019wey", "Swift J0427.3+6223"]),
    # Persistent confirmed BHs (also in BlackCAT):
    ("Cyg X-1", 299.590, 35.202, ["HDE 226868", "CygX-1", "Cygnus X-1", "HDE226868"]),
    ("LMC X-1", 84.912, -69.743, ["LMCX-1"]),
    ("LMC X-3", 84.735, -64.085, ["LMCX-3"]),
    ("GRS 1758-258", 270.308, -25.735, ["GRS1758-258", "1E 1755-338"]),
    ("1E 1740.7-2942", 265.978, -29.745, ["1E1740.7-2942", "Great Annihilator", "1E 1740"]),
    ("M33 X-7", 23.758, 30.541, ["M33X-7"]),
    ("IC 10 X-1", 5.100, 59.592, ["IC10X-1"]),
]

_PERSISTENT_BH_CANONS = {
    "Cyg X-1", "LMC X-1", "LMC X-3",
    "GRS 1758-258", "1E 1740.7-2942", "M33 X-7", "IC 10 X-1",
}


def _norm(s: str) -> str:
    """Normalise a source name: uppercase, strip spaces/dashes/plus/dots."""
    return str(s).upper().replace(" ", "").replace("-", "").replace("+", "").replace(".", "")


def _build_blackcat_lookup():
    name_map, ra_list, dec_list, canon_list = {}, [], [], []
    for canon, ra, dec, aliases in BLACKCAT_SOURCES:
        for nm in [canon] + aliases:
            name_map[_norm(nm)] = canon
        ra_list.append(ra); dec_list.append(dec); canon_list.append(canon)
    sc = SkyCoord(ra=ra_list * u.deg, dec=dec_list * u.deg)
    return name_map, sc, canon_list


_bc_name_map, _bc_sc, _bc_canon_list = _build_blackcat_lookup()


def match_blackcat(maxi_name: str, simbad_name: str, ra_deg: float, dec_deg: float) -> str | None:
    """Return the BlackCAT canonical name if matched, else None.
    Order: exact name -> substring name (len>6) -> 5-arcmin position.
    """
    for nm in [maxi_name, simbad_name]:
        key = _norm(nm)
        if key in _bc_name_map:
            return _bc_name_map[key]

    for nm in [maxi_name, simbad_name]:
        key = _norm(nm)
        for bc_key, bc_canon in _bc_name_map.items():
            if len(bc_key) > 6 and (bc_key in key or (len(key) > 6 and key in bc_key)):
                return bc_canon

    try:
        if not (np.isnan(float(ra_deg)) or np.isnan(float(dec_deg))):
            src = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg)
            seps = src.separation(_bc_sc)
            idx = int(np.argmin(seps))
            if seps[idx] < SEARCH_RADIUS:
                return _bc_canon_list[idx]
    except Exception:
        pass
    return None


def _get_col(table, candidates: list) -> str | None:
    for cand in candidates:
        for col in table.colnames:
            if col.strip().upper() == cand.upper():
                return col
    return None


def _extract_coords(table):
    """Return (SkyCoord, original_index_array) from a VizieR table, trying
    decimal-degree columns first then sexagesimal.
    """
    for ra_cands, dec_cands in [
        (["_RAJ2000", "_RA.ICRS", "RADEG", "RA_DEG"], ["_DEJ2000", "_DE.ICRS", "DEDEG", "DE_DEG"]),
        (["RAJ2000", "RA2000", "RA"], ["DEJ2000", "DE2000", "DEC"]),
    ]:
        rc, dc = _get_col(table, ra_cands), _get_col(table, dec_cands)
        if rc and dc:
            try:
                ra, dec = np.array(table[rc], dtype=float), np.array(table[dc], dtype=float)
                mask = ~np.isnan(ra) & ~np.isnan(dec)
                if np.any(mask):
                    return SkyCoord(ra=ra[mask] * u.deg, dec=dec[mask] * u.deg), np.where(mask)[0]
            except (ValueError, TypeError):
                try:
                    ra_s, dec_s = np.array(table[rc], dtype=str), np.array(table[dc], dtype=str)
                    valid = (ra_s != "") & (ra_s != "nan") & (dec_s != "") & (dec_s != "nan")
                    return (
                        SkyCoord(ra=ra_s[valid], dec=dec_s[valid], unit=(u.hourangle, u.deg)),
                        np.where(valid)[0],
                    )
                except Exception:
                    pass
    return None, None


def _parse_persistent_flag(table, row) -> str:
    type_col = _get_col(table, ["Type", "Var", "Class", "Vartype", "Status"])
    if not type_col:
        return "unknown"
    val = str(row[type_col]).strip().upper().replace("?", "").replace(" ", "")
    if not val or val in ("", "--", "NONE", "NAN"):
        return "unknown"
    if val.startswith("P"):
        return "P"
    if val.startswith("T") or val in ("SXT", "XRT", "TRANSIENT", "BET", "BE"):
        return "T"
    return "unknown"


def _parse_compact_object(table, row) -> str:
    pul_col = _get_col(table, ["Puls", "Ppulse", "Ppuls", "Pulse", "Pul", "Pp"])
    if pul_col:
        try:
            pval = float(row[pul_col])
            if not np.isnan(pval) and pval > 0:
                return "NS"
        except (ValueError, TypeError):
            raw = str(row[pul_col]).strip()
            if raw and raw not in ("", "--", "nan", "None"):
                return "NS"

    co_col = _get_col(table, ["Cp", "CO", "Compact", "Cobj", "Type2", "Rem"])
    if co_col:
        val = str(row[co_col]).strip().upper()
        if "BH" in val:
            return "BH"
        if "NS" in val or "PULSAR" in val or "NEUTRON" in val:
            return "NS"
    return "unknown"


def _get_source_name(table, row) -> str:
    name_col = _get_col(table, ["Name", "Xray", "Source", "SRC", "XRAY", "NAME"])
    return str(row[name_col]).strip() if name_col else ""


def fetch_liu_catalogs():
    """Fetch the Liu et al. LMXB (J/A+A/469/807) and HMXB (J/A+A/455/1165)
    VizieR catalogs. Returns (lmxb_table, lmxb_sc, lmxb_idx, hmxb_table,
    hmxb_sc, hmxb_idx); any of these may be None on a failed fetch.
    """
    from astroquery.vizier import Vizier

    Vizier.ROW_LIMIT = -1
    lmxb_table = lmxb_sc = lmxb_idx = None
    hmxb_table = hmxb_sc = hmxb_idx = None

    try:
        res = Vizier(columns=["**"]).get_catalogs("J/A+A/469/807")
        lmxb_table = res[0]
        lmxb_sc, lmxb_idx = _extract_coords(lmxb_table)
        print(f"LMXB catalog: {len(lmxb_sc) if lmxb_sc is not None else 0} sources loaded")
    except Exception as e:
        print(f"LMXB fetch FAILED: {e}")

    try:
        res = Vizier(columns=["**"]).get_catalogs("J/A+A/455/1165")
        hmxb_table = res[0]
        hmxb_sc, hmxb_idx = _extract_coords(hmxb_table)
        print(f"HMXB catalog: {len(hmxb_sc) if hmxb_sc is not None else 0} sources loaded")
    except Exception as e:
        print(f"HMXB fetch FAILED: {e}")

    return lmxb_table, lmxb_sc, lmxb_idx, hmxb_table, hmxb_sc, hmxb_idx


def query_liu(ra_deg, dec_deg, lmxb_table, lmxb_sc, lmxb_idx, hmxb_table, hmxb_sc, hmxb_idx):
    """Cross-match one position against Liu LMXB then HMXB within 5 arcmin.
    Returns (persistent_flag, compact_object, liu_name, catalog_label).
    """
    src = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg)

    if lmxb_sc is not None and len(lmxb_sc) > 0:
        seps = src.separation(lmxb_sc)
        best = int(np.argmin(seps))
        if seps[best] < SEARCH_RADIUS:
            row = lmxb_table[lmxb_idx[best]]
            return (
                _parse_persistent_flag(lmxb_table, row),
                _parse_compact_object(lmxb_table, row),
                _get_source_name(lmxb_table, row),
                "LMXB",
            )

    if hmxb_sc is not None and len(hmxb_sc) > 0:
        seps = src.separation(hmxb_sc)
        best = int(np.argmin(seps))
        if seps[best] < SEARCH_RADIUS:
            row = hmxb_table[hmxb_idx[best]]
            p_flag = _parse_persistent_flag(hmxb_table, row)
            co = _parse_compact_object(hmxb_table, row)
            if co == "NS" and p_flag == "unknown":  # NS pulsars are persistent
                p_flag = "P"
            return p_flag, co, _get_source_name(hmxb_table, row), "HMXB"

    return "unknown", "unknown", "", "NONE"


def classify_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full 4-step classification priority over every row of
    ``df`` (needs columns maxi_name, maxi_ra, maxi_dec, simbad_name).
    Returns a copy of ``df`` with the four classification columns added.
    """
    lmxb_table, lmxb_sc, lmxb_idx, hmxb_table, hmxb_sc, hmxb_idx = fetch_liu_catalogs()
    norm_persistent_set = {_norm(p) for p in KNOWN_PERSISTENT}

    records = []
    for _, row in df.iterrows():
        maxi_name = str(row["maxi_name"])
        simbad_name = str(row.get("simbad_name", ""))
        ra, dec = float(row["maxi_ra"]), float(row["maxi_dec"])

        persistent_flag, compact_object = "unknown", "unknown"
        liu_name_out, blackcat_out = "", ""

        # Step 1: BlackCAT -> BH
        bc_match = match_blackcat(maxi_name, simbad_name, ra, dec)
        if bc_match:
            compact_object = "BH"
            blackcat_out = bc_match

        # Step 2: Liu et al. -> persistent_flag + compact_object
        liu_pflag, liu_co, liu_name_val, _ = query_liu(
            ra, dec, lmxb_table, lmxb_sc, lmxb_idx, hmxb_table, hmxb_sc, hmxb_idx
        )
        if liu_name_val:
            liu_name_out = liu_name_val
        if compact_object == "unknown" and liu_co in ("BH", "NS"):
            compact_object = liu_co
        if liu_pflag in ("P", "T"):
            persistent_flag = liu_pflag

        # Step 3: known-persistent override
        mn, sn = _norm(maxi_name), _norm(simbad_name)
        for kp in norm_persistent_set:
            if kp in mn or kp in sn:
                persistent_flag = "P"
                break

        # Step 4: BH heuristic
        if persistent_flag == "unknown" and compact_object == "BH":
            persistent_flag = "P" if bc_match in _PERSISTENT_BH_CANONS else "T"

        records.append(
            {
                "persistent_flag": persistent_flag,
                "compact_object": compact_object,
                "liu_name": liu_name_out,
                "blackcat_name": blackcat_out,
            }
        )

    res_df = pd.DataFrame(records)
    out = df.copy()
    for col in ["persistent_flag", "compact_object", "liu_name", "blackcat_name"]:
        out[col] = res_df[col].values
    return out


def print_summary(df: pd.DataFrame) -> None:
    print(f"\nTotal sources: {len(df)}")
    print("\npersistent_flag breakdown:")
    for v in ("P", "T", "unknown"):
        print(f"  {v:<10}: {(df['persistent_flag'] == v).sum():>4}")
    print("\ncompact_object breakdown:")
    for v in ("BH", "NS", "unknown"):
        print(f"  {v:<10}: {(df['compact_object'] == v).sum():>4}")
    print("\nCross-match stats:")
    print(f"  Liu matches   : {(df['liu_name'] != '').sum():>4}")
    print(f"  BlackCAT hits : {(df['blackcat_name'] != '').sum():>4}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="step1_maxi_xrb_sources.csv")
    parser.add_argument("--output", default="step1b_maxi_xrb_classified.csv")
    args = parser.parse_args()

    df_in = pd.read_csv(args.input)
    print(f"Loaded {len(df_in)} sources from {args.input}")
    df_out = classify_sources(df_in)
    df_out.to_csv(args.output, index=False)
    print_summary(df_out)
    print(f"\nSaved -> {args.output}")
