# `xrb_outbursts.json` schema

The persistent database used by `reprocessing/manual_inspector.py`,
`optical_simulation.py`, and `free_beta_fitting.py`. Keyed by MAXI
source ID (e.g. `"J1911+005"`). Never edit by hand -- always go through
`xrb_pipeline.utils.json_db.load_json` / `save_json`.

```json
{
    "J1911+005": {
        "display_name":   "Aql X-1",
        "compact_object": "NS",
        "lmxb":           true,
        "ztf_id":         "ZTF18accedeu",
        "ztf_band":       "r",
        "outbursts": [
            {
                "x1": 57580.0,
                "x2": 57750.0,
                "status": "fitted",

                "t_start": 57578.3,
                "t_rise": 2.1,
                "t_decay": 22.4,
                "amplitude": 0.084200,
                "peak_mjd": 57584.6,

                "T90": 51.4,
                "chi2_red": 2.31,
                "rms": 0.000234,

                "N_calib": 12,
                "tau_lag": 1.8,
                "C_A": 15.2341, "C_A_err": 0.0412,
                "C_B": 16.3102, "C_B_err": 0.0498,
                "RMS_A": 0.00314, "RMS_B": 0.00287,

                "beta_free": 0.53, "sigma_beta_free": 0.01,
                "C_free": 15.87, "sigma_C_free": 0.09,
                "N_beta": 42
            }
        ]
    }
}
```

## Source-level fields

| Field | Meaning |
|---|---|
| `display_name` | Human-readable name (e.g. "Aql X-1") |
| `compact_object` | `"NS"` or `"BH"` -- authoritative; never re-inferred downstream |
| `lmxb` | `true` for LMXB, `false` for HMXB (Russell et al. 2006 reprocessing relation was derived for LMXBs; HMXB results should be read with caution) |
| `ztf_id` | ALeRCE/ZTF object id |
| `ztf_band` | Preferred single band for quick-look plots (the reprocessing pipeline itself pools all bands) |

## Per-outburst fields, by pipeline stage

| Stage | Fields | Set by |
|---|---|---|
| Window selection | `x1`, `x2` | Manual inspection (Phase 0) |
| FRED fit (Phase 0) | `status`, `t_start`, `t_rise`, `t_decay`, `amplitude`, `peak_mjd`, `T90`, `chi2_red`, `rms` | `manual_inspector.fit_outburst_fred` + `save_outburst_fit` |
| Calibration (Phase 2) | `N_calib`, `tau_lag`, `C_A`, `C_A_err`, `C_B`, `C_B_err` | `manual_inspector.run_phase2_calibration` |
| Optical sim (Phase 3) | `RMS_A`, `RMS_B` | `optical_simulation.save_rms_to_json` |
| Free-beta (Phase 5) | `beta_free`, `sigma_beta_free`, `C_free`, `sigma_C_free`, `N_beta` | `free_beta_fitting.save_beta_free_to_json` |

## `status` values

| Value | Meaning |
|---|---|
| `"window_only"` | Window spotted visually, FRED not yet run |
| `"fitted"` | FRED converged -- always set if `fit_ok`, regardless of `chi2_red` |
| `"rejected"` | Not a real outburst -- set manually |

`chi2_red` is diagnostic only. Complex/multi-peaked outbursts routinely
have `chi2_red` well above 1 with a single smooth FRED; the fit is
still saved, and `chi2_red` just tells you how well the model
approximates that particular light curve.

Downstream stages gate on the presence of earlier fields, not on
`status` alone: Phase 2 requires `status == "fitted"`; Phase 3/5
require `"C_A" in outburst` (i.e. Phase 2 already ran). A stage that
finds its prerequisite missing should skip that outburst cleanly rather
than guessing at defaults.
