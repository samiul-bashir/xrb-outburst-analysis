# Algorithms

How each stage of the pipeline works, in the order data flows through it.
See `docs/report.pdf` (KSP-07, Samiu Bashir) for the full physical
background and results; this document is a code-facing companion.

## 1. Cross-matching (`src/xrb_pipeline/crossmatch/`)

**`bat_maxi_simbad_ztf.py`** -- narrow, high-confidence sample. Four
sequential steps, each gated on the previous:

1. **BAT -> MAXI**: convert each Swift/BAT source's RA/Dec to a MAXI
   J-name (`J` + truncated RA hours/minutes + sign + truncated Dec
   degrees/tenths) and check for an HTTP 200 on that MAXI light-curve
   page.
2. **MAXI -> SIMBAD**: resolve each surviving name against SIMBAD for a
   precise optical position and object type (`otype`). SIMBAD
   coordinates replace the coarser BAT position downstream.
3. **SIMBAD -> ZTF**: 5-arcsec cone search via the ALeRCE broker;
   keep the nearest match.
4. **XRB-type filter**: keep only `otype` in `{LXB, HXB}` with a ZTF
   match.

**`classify_compact_objects.py`** -- broader MAXI-wide classification
(this is what feeds the outburst catalogue). Priority order:

1. **BlackCAT name/position match -> BH.**
2. **Liu et al. LMXB/HMXB VizieR catalogs** (5-arcmin match): a
   non-empty pulse-period column -> NS; otherwise trust an explicit
   compact-object column. Also supplies the persistent/transient flag.
3. **Known-persistent override**: a hardcoded list of ~28 sources
   (Sco X-1, Cyg X-1, Crab, ...) always gets `persistent_flag = P`,
   applied last so it always wins.
4. **BH heuristic**: a BH source not otherwise flagged persistent gets
   `T` unless it's one of the five known persistent BHs (Cyg X-1,
   LMC X-1/X-3, GRS 1758-258, 1E 1740.7-2942).

## 2. Outburst detection (`src/xrb_pipeline/detection/`)

Two detectors exist because they serve different downstream needs.

**`simple_detector.py`** -- one global sigma-clipped threshold
(baseline = median, noise = std of the sub-90th-percentile bulk),
consecutive-day grouping with a small gap tolerance, and duplicate-peak
suppression. Used by the FRED-family fitting pipelines, where a quick,
uniform detector is sufficient to hand each model a well-defined peak
window.

**`tiered_detector.py`** -- the detector behind the main outburst
catalogue and duration statistics. Routes each source by SIMBAD otype +
compact-object class into one of four configs (BH LXB / NS LXB /
unknown LXB / HXB), each with its own baseline method, sigma
threshold, minimum duration/separation, and merge tolerance:

- **Iterative baseline** (LXB): sigma-clip outburst flux out of the
  quiet-flux estimate iteratively, so a source with many/long outbursts
  doesn't bias its own baseline upward through contamination.
- **Peak finding**: `scipy.signal.find_peaks` with a prominence
  constraint (each peak must clear its local surroundings by
  `SIGMA_MULT * sigma_r`) and a minimum separation in days tuned per
  class (BH: 120 d, NS: 80 d) -- this is what prevents a single
  multi-peaked outburst from being split into spurious separate events.
- **Boundary walk**: from each peak's half-width, walk outward while
  flux stays above a low (1-sigma) threshold, with a gap tolerance and
  a hard cap (`MAX_WING_DAYS`) so a slow BH decay can't eat into the
  next outburst.
- **HR spectral veto** (NS LXB only): a genuine accretion outburst
  should soften (HR near peak <= 0.75x the quiescent HR); a candidate
  that doesn't soften is a rejected hard-state flare. Disabled when
  quiescent baseline is too short or itself too hard to be a reliable
  reference.
- **HXB baseline**: a 300-day rolling median on 80th-percentile-clipped
  flux (HXBs never return to a true quiescent floor), so a discrete
  outburst is judged against a smoothly time-varying local background
  rather than one global number.

## 3. Duration statistics (`src/xrb_pipeline/duration_stats/`)

T90, T_rise, T_decay are all derived from the cumulative, baseline-
subtracted fluence within a detected window (same formalism as GRB T90):
find the MJDs bounding the central 90% of net fluence (`t5`, `t95`);
`T90 = t95 - t5`. Splitting at the flux peak within `[t5, t95]` gives
`T_rise = t_peak - t5` and `T_decay = t95 - t_peak` (so `T_rise +
T_decay = T90` by construction). If the peak doesn't fall inside
`[t5, t95]` (multi-peaked/pathological profiles), the split is NaN'd
out rather than guessed at.

## 4. Fitting (`src/xrb_pipeline/fitting/`, models in `utils/models.py`)

Four outburst-profile models, because different stages of the analysis
needed different trade-offs:

- **`simple_fred`**: two-sided exponential referenced to the peak
  (`t_peak` is a free parameter). Used for the Pure-FRED pipeline
  (report 5.1) -- simple, but the symmetric peak-referencing can leave
  a visible kink and a wide range of chi2_red across complex outbursts.
- **`fred_tophat`**: `simple_fred` with a flat plateau of width `width`
  inserted between rise and decay (report 5.2). `width=0` recovers a
  start-referenced pure FRED. Dramatically better chi2_red on outbursts
  with a sustained maximum rather than a sharp spike.
- **`asymmetric_burst_profile`**: independent rise/decay amplitudes
  plus a free background level, adapted from a LAXPC Type-I burst
  prototype (see `docs/legacy_burst_lc_fitting.md`). Not in the KSP-07
  report's final results, kept as a documented alternative.
- **`norris_fred`**: the smooth (no-kink) mentor's-formulation FRED
  used throughout Phases 0-5 (report 8). `t_start`/`t_rise`/`t_decay`
  are fitted; `t_peak = t_start + sqrt(t_rise * t_decay)` is *derived*,
  and `F(t_peak) == amplitude` exactly by construction -- there is no
  kink at the peak, unlike the piecewise `simple_fred`/`fred_tophat`
  models.
- **`multi_fred`** (`fitting/multi_fred.py`, bonus, not in the report):
  a sum of N `simple_fred` pulses for outbursts with genuine multiple
  sub-peaks that no single-pulse model can capture.

All fits report `chi2_reduced` for comparison, but it is *diagnostic
only* -- no fit is ever auto-rejected on chi2_red; complex/multi-peaked
outbursts routinely have chi2_red well above 1 with a single smooth
model, and that's expected (see
`docs/methodology-decisions` if present in your memory notes, or the
report's Section 5 discussion).

## 5. Hardness (`src/xrb_pipeline/hardness/hid.py`)

X-ray hardness = flux(4-10 keV) / flux(2-4 keV); intensity =
flux_total (2-20 keV). The Hardness-Intensity Diagram traces spectral
state transitions through an outburst as a loop rather than two
independent time series. The optical analogue uses a "color" (flux
difference between two ZTF bands, converted from AB magnitude) instead
of a ratio, since ZTF reports magnitudes rather than physical flux
ratios; bands aren't observed simultaneously, so points are first
nightly-binned per band, then matched by night. The g-r pair is
preferred (best cadence); r-i is the fallback when g is too sparse in a
given window.

## 6. Reprocessing (`src/xrb_pipeline/reprocessing/`)

The optical-X-ray reprocessing relation is `F_OIR = C_source * F_X^beta`.

- **`manual_inspector.py`** (Phases 0-2): fit the Norris FRED to a
  manually-selected outburst window; fetch ZTF and label MAXI epochs
  hard/soft/intermediate by HR; calibrate `C_source` for two fixed
  literature betas (`BETA_A=0.50`, the theoretical X-ray-reprocessing
  prior; `BETA_B=0.61`, the empirical Russell et al. 2006 BH LMXB
  slope) using **hard-state epochs only** -- soft-state points are
  excluded because the dominant optical emission mechanism changes
  (thermal disk vs. reprocessing), so mixing states would bias the
  calibration.
- **`optical_simulation.py`** (Phase 3-4): evaluate the stored FRED fit
  at the ZTF epochs (shifted by the fitted `tau_lag`, the offset
  between the ZTF and X-ray peaks), scale by the calibrated
  `C_source`, and compute an RMS residual against observed ZTF flux for
  each beta -- the smaller RMS says which beta describes that specific
  outburst better. Phase 4 flattens every calibrated outburst across
  the whole database for cross-source comparison.
- **`free_beta_fitting.py`** (Phase 5): instead of assuming a fixed
  beta, solve for beta and C_source simultaneously via analytic
  weighted least squares in log-log space (with a scipy cross-check).
  Gated on sample size (`MIN_N_BETA=5`) and on the x-range being
  non-degenerate (a slope isn't identifiable if the X-ray flux barely
  moved across the whole hard-state window). The resulting `beta_free`
  is compared against NS/BH literature anchors (Russell et al. 2006:
  NS 0.63+/-0.04, BH 0.61+/-0.02) to test whether a given outburst is
  consistent with disc reprocessing, jet emission, or something in
  between.

Two count-rate-to-flux constants appear in this codebase and are *not*
interchangeable:

| Constant | Value | Used by |
|---|---|---|
| `MAXI_CONV` (Phase 0-5 / reprocessing) | `2.4e-8 / 3.3` | `reprocessing/*` |
| `MAXI_CONV` (crossmatch/detection notebooks) | `7.27e-9` | report's HID section only |

The reprocessing value is the one that matters for calibration; an
early draft passed the *raw* count-rate amplitude through unconverted
when building `fred_smooth_erg` -- this was fixed from Phase 3-4
onward and stays fixed in `utils/models.make_norris_fred_smooth`.
