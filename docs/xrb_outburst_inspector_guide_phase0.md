# XRB Outburst Inspector — User Guide

**Notebook:** `xrb_outburst_inspector.ipynb`  
**Phases covered:** 0 (manual inspection + FRED fit) · 1 (ZTF + Crab conversion + states) · 2 (calibration)  
**Next session:** Phase 3 (simulate optical + dual-axis plot) loads from the JSON this notebook produces.

---

## Prerequisites

```bash
pip install numpy scipy pandas matplotlib plotly requests alerce nbformat
```

---

## The JSON — your persistent data store

Everything lives in `xrb_outbursts.json` in the same folder as the notebook.
It is created automatically on first run. **Never edit it by hand** — always use the notebook cells.

### Structure

```json
{
    "J1911+005": {
        "display_name"  : "Aql X-1",
        "compact_object": "NS",
        "lmxb"          : true,
        "ztf_id"        : "ZTF18accedeu",
        "ztf_band"      : "r",
        "outbursts": [
            {
                "x1"       : 57580.0,
                "x2"       : 57750.0,
                "status"   : "fitted",
                "t_start"  : 57578.3,
                "t_rise"   : 2.1,
                "t_decay"  : 22.4,
                "amplitude": 0.084200,
                "peak_mjd" : 57584.6,
                "T90"      : 51.4,
                "chi2_red" : 2.31,
                "rms"      : 0.000234,
                "N_calib"  : 12,
                "tau_lag"  : 1.8,
                "C_A"      : 15.2341,
                "C_A_err"  : 0.0412,
                "C_B"      : 16.3102,
                "C_B_err"  : 0.0498,
                "RMS_A"    : null,
                "RMS_B"    : null
            }
        ]
    }
}
```

### Status field

| Value | Meaning |
|:---|:---|
| `"window_only"` | Window spotted visually, FRED not yet run |
| `"fitted"` | FRED converged — always set if fit_ok, regardless of χ²_red |
| `"rejected"` | Not a real outburst — manually set by you in 0-F |

> **χ²_red is diagnostic only.** Complex outbursts rarely reach χ²_red < 3 with a single FRED.
> The fit is saved regardless; χ²_red just tells you how well the smooth model approximates the data.

---

## The FRED Model

This notebook uses the **Norris FRED** (mentor's formulation):

$$F(t) = A \cdot e^{2\sqrt{t_\mathrm{rise}/t_\mathrm{decay}}} \cdot \exp\!\left(-\frac{t_\mathrm{rise}}{t - t_\mathrm{start}} - \frac{t - t_\mathrm{start}}{t_\mathrm{decay}}\right), \quad t > t_\mathrm{start}$$

| Parameter | Units | Meaning |
|:---|:---:|:---|
| `t_start` | MJD | Onset time — where flux begins rising from zero |
| `t_rise` | days | Rise sharpness — smaller = sharper rise |
| `t_decay` | days | Exponential decay timescale |
| `amplitude` | ct/s/cm² | True peak value (exact by construction) |

**Peak time** (derived, not fitted):  
$$t_\mathrm{peak} = t_\mathrm{start} + \sqrt{t_\mathrm{rise} \times t_\mathrm{decay}}$$

This profile is **smooth everywhere** — no kink at the peak like a piecewise exponential.

---

## Session Workflow

### First-ever run

1. Open `xrb_outburst_inspector.ipynb`
2. Run **Section 0** (all three cells) — imports, catalogue, JSON init
3. JSON file `xrb_outbursts.json` is created

### Every subsequent session

Always start by running **Section 0** to reload the JSON into `db`.

---

## Phase 0 — Finding and Fitting Outbursts

### Per-source setup (do once per source)

**Cell 0-A** — set `ACTIVE_SOURCE` to the MAXI ID you want to work on.

```python
ACTIVE_SOURCE = "J1911+005"   # Aql X-1
```

Available MAXI IDs:

| MAXI ID | Name | Type |
|:---|:---|:---|
| `J1911+005` | Aql X-1 | NS LMXB |
| `J0622-003` | 1A 0620-00 | BH LMXB |
| `J1657+353` | Her X-1 | NS HMXB |
| `J1911+049` | SS 433 | BH HMXB |
| `J1949+302` | KS 1947+300 | NS HMXB |
| `J2058+417` | GRO J2058+42 | NS HMXB |
| `J0607+220` | IGR J06074+2205 | NS HMXB |

**Cell 0-B** — fetches the 9-column MAXI 1-day lightcurve. Run once, stays in memory.

---

### Per-outburst loop (repeat for every outburst in this source)

**Cell 0-C** — interactive Plotly lightcurve.  
- Drag to zoom horizontally  
- The 5σ threshold (red dashed) helps identify real outbursts  
- Already-saved windows appear as coloured bands (green = fitted, blue = window_only)  
- Note the approximate **start and end MJD** of the outburst you want to fit

**Cell 0-D** — set the window and guesses, then run.

```python
X1              = 57580.0   # MJD start — from 0-C
X2              = 57750.0   # MJD end   — from 0-C
T_START_GUESS   = 57578.0   # MJD onset (≈ X1, or where flux first rises)
T_RISE_GUESS    = 2.0       # days — try 1–5 for sharp rises, 5–15 for gradual
T_DECAY_GUESS   = 20.0      # days — eyeball the tail length in 0-C
```

A zoomed preview plot appears immediately — confirm the window looks right before continuing.  
**You never need to scroll back up** — 0-D is always directly below 0-C.

**Cell 0-E** — runs the FRED fit. You will see:
- A fit plot styled like your mentor's example (gray data points, smooth black FRED curve)
- A parameter box with all fitted values and their uncertainties
- A residuals panel below the main fit
- χ²_red and RMS as diagnostic output

If the fit **does not converge**:
1. Scroll up ONE cell to 0-D
2. Try a different `T_START_GUESS` (move it closer to where flux first appears)
3. Try larger `T_RISE_GUESS` for a more gradual rise
4. Re-run 0-D then 0-E

**Cell 0-F** — saves the result. Always sets `status = "fitted"`.  
To mark as rejected: uncomment `STATUS = "rejected"` before running.

**Cell 0-G** *(optional)* — if you spotted a window but don't want to fit it now,
run 0-G to save a `window_only` entry. The window will appear in 0-C overlays next session.

---

### Typical per-source session

```
Run 0-A → Run 0-B → Run 0-C (look around)
  → Edit 0-D → Run 0-D → Run 0-E → Run 0-F   ← outburst 1
  → Edit 0-D → Run 0-D → Run 0-E → Run 0-F   ← outburst 2
  → Edit 0-D → Run 0-D → Run 0-E → Run 0-F   ← outburst 3
  ...
```

---

## Phase 1 — ZTF + Crab Conversion + State Labels

Run after all outbursts for `ACTIVE_SOURCE` are fitted. Processes one outburst at a time.

**Cell 1-A** — select which outburst to process:
```python
OUTBURST_IDX = 0   # index into db[ACTIVE_SOURCE]["outbursts"]
```

**Cell 1-B** — fetches ZTF lightcurve via ALeRCE, crops to window, converts to mJy:
$$F_\nu \;[\mathrm{mJy}] = 3631 \times 10^{-0.4\,m_\mathrm{AB}} \times 1000$$

**Cell 1-C** — computes:
- **Crab conversion:** $F_X\;[\mathrm{erg\,cm^{-2}\,s^{-1}}] = \mathrm{flux}_{2\text{–}20} \times 7.27\times10^{-9}$
- **Hardness ratio:** $\mathrm{HR} = F_{4\text{–}10} / F_{2\text{–}4}$
- **State labels:** `hard` (HR > 1.0) · `soft` (HR < 0.5) · `int` (intermediate)
- **FRED smooth model** `fred_smooth_erg(t)` — mentor's FRED in erg/cm²/s at arbitrary MJD

Produces two diagnostic plots: MAXI flux + states + FRED smooth, and ZTF flux in window.

> You can tune `HR_HARD` and `HR_SOFT` in Section 0 if the state assignment looks wrong.

---

## Phase 2 — Calibration

**Cell 2-A** — for each ZTF epoch in the window:
- Finds nearest MAXI epoch within `DT_MATCH = 1.0` day
- Keeps only `state == "hard"` pairs
- Evaluates $F_X^\mathrm{smooth}$ from FRED model at the ZTF MJD
- Computes per-epoch calibration coefficient:
  $$C_i = \log_{10}(F_\mathrm{OIR,i}) - \beta \cdot \log_{10}(F_{X,i})$$
- Weighted mean → $C_A$ (β = 0.50) and $C_B$ (β = 0.61)

Also computes **τ_lag** = ZTF peak MJD − FRED peak MJD (optical delay relative to X-ray).

If `N_calib < 3`, possible causes:
- Source goes directly to soft state (no hard-state rise visible in ZTF)
- ZTF coverage gap during the hard-state rise
- HR threshold too strict → lower `HR_HARD` in Section 0

**Cell 2-B** — saves $C_A$, $C_B$, $N_\mathrm{calib}$, $\tau_\mathrm{lag}$ to JSON.
`RMS_A` and `RMS_B` remain `null` until Phase 3.

---

## JSON Summary Cell

Run the final cell at any time to see the full state of the database:

```
======================================================================
  XRB Outburst Database  |  xrb_outbursts.json
======================================================================

  J1911+005        Aql X-1                NS  (LMXB)
  ZTF: ZTF18accedeu  (r-band)
  outbursts=5  fitted=4  calibrated=2
    [0] MJD 57580→57750  status=fitted        peak=57584.6  χ²=2.31  C_A=15.2341  ...
    [1] MJD 58100→58250  status=fitted        peak=58112.1  χ²=1.88  C_A=15.1902  ...
```

---

## Tunable Constants (Section 0)

| Constant | Default | What it controls |
|:---|:---:|:---|
| `BETA_A` | 0.50 | X-ray reprocessing prior slope (Run A) |
| `BETA_B` | 0.61 | Russell et al. 2006 empirical slope (Run B) |
| `HR_HARD` | 1.0 | HR threshold above which epoch = hard state |
| `HR_SOFT` | 0.5 | HR threshold below which epoch = soft state |
| `DT_MATCH` | 1.0 | Max \|ΔMJD\| for ZTF–MAXI pairing [days] |
| `MAXI_CONV` | 7.27e-9 | Crab conversion: ct/s/cm² → erg/cm²/s |

---

## Tips

- **HMXB sources** (Her X-1, SS 433, KS 1947+300, GRO J2058+42, IGR J06074+2205): the OIR–X-ray reprocessing correlation was derived for LMXBs. Phases 1–2 will still run, but interpret $C_A$/$C_B$ with caution.
- **Complex outbursts** with multiple peaks or re-brightenings will not be perfectly described by a single FRED. That is expected and fine — χ²_red is only stored for your information, never used as a rejection criterion. The smooth model is still used for Phase 2 calibration.
- **Kernel restart**: all fitted data is safe in the JSON. Just re-run Section 0 to reload `db` and continue.
- **Adding a new source**: add an entry to `SOURCE_CATALOGUE` in the notebook and re-run the catalogue cell. The new source will be merged into the existing JSON automatically.
- **Window-only entries** appear as blue bands in 0-C — a useful reminder of outbursts you spotted but haven't fitted yet.
