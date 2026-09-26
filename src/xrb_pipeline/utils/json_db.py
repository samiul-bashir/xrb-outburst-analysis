"""
Persistent JSON outburst database, shared by the manual inspector and every
downstream reprocessing phase (see docs/algorithms.md and the schema in
docs/json_db_schema.md).

The database is keyed by MAXI source ID (e.g. ``"J1911+005"``); each entry
carries source metadata plus a list of per-outburst fit/calibration
records. Never edit the JSON file by hand -- always go through
``load_json`` / ``save_json``.
"""
from __future__ import annotations

import json
import os


def load_json(json_path: str, source_catalogue: dict | None = None) -> dict:
    """Load the outburst database, seeding or merging from
    ``source_catalogue`` (a dict of MAXI-ID -> source metadata) on first
    run or when new sources have been added since the file was created.
    """
    source_catalogue = source_catalogue or {}
    if os.path.exists(json_path):
        with open(json_path) as fh:
            data = json.load(fh)
        for mid, cat in source_catalogue.items():
            if mid not in data:
                data[mid] = {k: v for k, v in cat.items()}
                print(f"  Merged new source: {cat.get('display_name', mid)}")
        print(f"Loaded  {json_path}")
    else:
        data = {k: {kk: vv for kk, vv in v.items()} for k, v in source_catalogue.items()}
        print(f"No JSON found -- seeded from source_catalogue -> {json_path}")
    return data


def save_json(data: dict, json_path: str) -> None:
    with open(json_path, "w") as fh:
        json.dump(data, fh, indent=4)
    print(f"Saved -> {json_path}")


def summarize(data: dict) -> None:
    """Print a one-line-per-source summary of database contents."""
    print(f"  {'MAXI ID':<15s} {'Name':<22s} {'Type':<5s} {'LMXB':<6s}  #obs  #fitted")
    print("  " + "-" * 62)
    for mid, src in data.items():
        obs = src.get("outbursts", [])
        n_fit = sum(1 for o in obs if o.get("status") == "fitted")
        flag = "OK" if src.get("lmxb") else "HMXB"
        print(
            f"  {mid:<15s} {src.get('display_name', ''):<22s} "
            f"{src.get('compact_object', ''):<5s} {flag:<6s}  "
            f"{len(obs):>3}   {n_fit:>3}"
        )
