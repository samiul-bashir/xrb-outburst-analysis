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
