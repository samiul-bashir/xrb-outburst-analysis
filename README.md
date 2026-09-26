# Multiwavelength Analysis of X-ray Binary Outbursts

Pipeline for identifying, fitting, and characterising X-ray binary (XRB)
outbursts using MAXI (X-ray) and ZTF (optical) light curves, and
modelling the optical emission as X-ray reprocessing. Built for KSP-07
(Summer 2026); see [`docs/report.pdf`](docs/report.pdf) for the full
write-up and physical background, and [`docs/algorithms.md`](docs/algorithms.md)
for a code-facing explanation of how each stage works.

## What it does

1. **Cross-match** Swift/BAT, MAXI, SIMBAD and ZTF to build a sample of
   confirmed XRBs with optical counterparts, and classify a broader
   MAXI-wide sample as BH/NS and persistent/transient using BlackCAT
   and the Liu et al. catalogs.
2. **Detect outbursts** in MAXI light curves via a class-routed,
   sigma-thresholded peak finder (separate tuning for BH LXBs, NS LXBs,
   and HXBs, with an optional hardness-ratio veto).
3. **Fit** each outburst with a Fast-Rise-Exponential-Decay (FRED)
   profile -- plain, with a flat-top plateau, or an asymmetric
   variant -- and compute duration statistics (T90, rise/decay times).
4. **Track spectral state** via Hardness-Intensity Diagrams in both
   X-ray (MAXI sub-bands) and optical (ZTF color).
5. **Model the optical outburst** as X-ray reprocessing,
   `F_OIR = C_source * F_X^beta`, calibrated per outburst against
   fixed literature betas and, separately, with beta left free.

## Repository layout

```
src/xrb_pipeline/     Importable pipeline modules (see below)
notebooks/            One notebook per pipeline stage, for interactive use
scripts/              Standalone CLI scripts for common one-off tasks
docs/                 Report PDF, algorithm notes, JSON schema, phase-0 guide
data/                 Empty -- see "Data" below
```

`src/xrb_pipeline/` is organised by pipeline stage, matching the report
chapters:

| Module | Report section | What's in it |
|---|---|---|
| `crossmatch/` | 3.1, 3.2 | BAT-MAXI-SIMBAD-ZTF matching; BH/NS + persistent/transient classification |
| `detection/` | 4.1, 4.2 | Simple threshold detector; class-routed tiered detector |
| `fitting/` | 5.1, 5.2 | Pure FRED, FRED+tophat, asymmetric burst profile, multi-FRED (bonus) |
| `duration_stats/` | 6 | T90, T_rise, T_decay |
| `hardness/` | 7 | X-ray and optical Hardness-Intensity Diagrams |
| `reprocessing/` | 8 | Manual inspector (FRED fit + hard-state calibration), optical simulation, free-beta fitting |
| `utils/` | -- | Shared MAXI/ZTF fetch, outburst-profile models, JSON database helpers |

## Installation

```bash
git clone <this-repo-url>
cd xrb-outburst-analysis
pip install -r requirements.txt
pip install -e .   # optional, if you want `import xrb_pipeline` from anywhere
```

Or without an editable install, just add `src/` to your path:

```bash
export PYTHONPATH=src
```

## Usage

Each pipeline stage can be run as a script or imported as a library.

```bash
# 1. Cross-match and classify
python -m xrb_pipeline.crossmatch.bat_maxi_simbad_ztf --bat-fits BAT_catalog.fits --outdir data/crossmatch
python -m xrb_pipeline.crossmatch.classify_compact_objects --input step1_maxi_xrb_sources.csv

# 2. Detect outbursts
python -m xrb_pipeline.detection.tiered_detector --input step1b_maxi_xrb_classified.csv

# 3. Duration statistics
python -m xrb_pipeline.duration_stats.duration_metrics --outbursts step2_outbursts.csv --classified step1b_maxi_xrb_classified.csv
```

For interactive, per-source work (manual outburst inspection, FRED
fitting, HID plots), use the notebooks in `notebooks/` -- these follow
the session workflow described in
[`docs/xrb_outburst_inspector_guide_phase0.md`](docs/xrb_outburst_inspector_guide_phase0.md).

### Scripts

`scripts/` has small standalone tools that don't need the full
pipeline context -- useful starting points to modify for a one-off
task:

- `scripts/fetch_lightcurve.py` -- fetch and plot a single MAXI light
  curve by source ID.
- `scripts/quick_outburst_scan.py` -- run the simple detector on one
  source and print/plot what it finds, without touching the JSON
  database or the classified catalogue.

## Data

This repo ships no data. You'll need:

- A Swift/BAT catalog FITS file (for cross-matching) -- e.g. the
  Swift/BAT 157-month hard X-ray catalog.
- MAXI, SIMBAD, ZTF/ALeRCE, VizieR (Liu et al. LMXB/HMXB) and BlackCAT
  are all queried live over HTTP; no local copies needed beyond the
  BAT FITS file. Be considerate of the MAXI server (the crossmatch
  script already rate-limits itself).
- The MAXI-wide classification pipeline (`classify_compact_objects.py`)
  expects `step1_maxi_xrb_sources.csv`, an upstream SIMBAD-otype
  pre-filter over the full MAXI GSC source list. That pre-filter step
  isn't included here; regenerate it by querying SIMBAD for every MAXI
  GSC source position and keeping `otype in {LXB, HXB}`.

The persistent per-outburst database (`xrb_outbursts.json`) is built
and grown by the notebooks in `notebooks/` as you inspect and fit
sources; see [`docs/json_db_schema.md`](docs/json_db_schema.md).

## Citing / license

MIT licensed -- see [`LICENSE`](LICENSE). If you use this code, please
cite the accompanying report (`docs/report.pdf`): Samiu Bashir,
*"Multiwavelength Analysis of X-ray Binary Outbursts"*, KSP-07, Summer
2026.
