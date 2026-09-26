# Legacy: `burst-lc-fitting.ipynb`

Not included in this repo's `src/` or `notebooks/`, but worth
documenting since two pieces of code elsewhere trace back to it.

This was a prototype notebook operating on **LAXPC event-file data**
(hardcoded local paths like `/Users/.../4U1728-34/.../laxpc-results/`)
-- a different instrument and a different phenomenon (Type-I
thermonuclear X-ray bursts, not accretion outbursts) from the rest of
this project. It fits an asymmetric exponential burst profile
(independent rise/decay amplitudes + free background) to individual
bursts extracted from a LAXPC light curve.

Two things were carried forward from it into the main MAXI-outburst
pipeline:

1. The **asymmetric burst-profile model** itself, applied to MAXI
   outburst peaks instead of LAXPC bursts -- see
   `src/xrb_pipeline/utils/models.asymmetric_burst_profile` and
   `src/xrb_pipeline/fitting/asymmetric_burst_profile.py`.
2. The **cumulative-fluence T90 method**, adapted for outburst
   (rather than burst) timescales -- see
   `src/xrb_pipeline/duration_stats/duration_metrics.py`.

If you have the original LAXPC notebook and want it in the repo for
provenance, a natural place is `notebooks/legacy/burst-lc-fitting.ipynb`
with a short header noting it needs LAXPC-specific data products
(`laxpc-results/` directories) that aren't part of this project's data
pipeline.

# Legacy: earlier crossmatch drafts

Three earlier drafts of the BAT-MAXI-SIMBAD-ZTF crossmatch (report
§3.1) were superseded by `crossmatch/bat_maxi_simbad_ztf.py` (sourced
from the notebook named `M-B-S-Z_outburst_detection_clean.ipynb`) and
aren't included in `src/` or `notebooks/`:

- `BAT_ZTF_crossmatch.ipynb` -- earliest draft, BAT→ZTF direct cone
  search with no MAXI or SIMBAD step.
- `MAXI_BAT_ZTFmatch_Copy.ipynb` -- intermediate draft, adds MAXI and
  ZTF but a different step ordering than the final pipeline.
- `MAXI_BAT_ZTFmatch_SIMBAD_copy_fix.ipynb` -- the messier precursor
  `_clean` was cleaned from. Its crossmatch logic is identical to what's
  in `bat_maxi_simbad_ztf.py`, but it also contains a large amount of
  exploratory plotting code not carried forward:
  - A multi-panel grid combining each source's MAXI (all sub-bands) +
    ZTF light curves into one figure -- this produced the "Multi-band
    X-ray & Optical Lightcurves" grid (report Figure 1, the 8-source
    panel).
  - Several alternate interactive-plotting cells (Plotly range-slider
    versions, combined PNG grids).
  - A separate, simpler duration-filtered outburst detector applied
    only to the small BAT-matched "Gold" sample -- superseded by the
    tiered detector in `detection/tiered_detector.py`, which is used
    for the full catalogue instead.

  If you want to regenerate the Figure 1 grid, the plotting logic is
  worth pulling from this notebook specifically; it wasn't ported here
  since it's presentation code rather than pipeline logic.
