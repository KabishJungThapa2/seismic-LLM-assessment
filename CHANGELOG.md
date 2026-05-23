# Changelog

## v2.0.0 — Bug-Fix Release (May 2026)

This release fixes ten bugs identified during a code review against the
thesis specification. Pipeline now runs end-to-end with zero OpenSeesPy
warnings and produces engineering-correct EDPs.

### Critical Fixes

1. **Amendment 2 was not applied in `src/compliance.py`** (the centerpiece of
   the thesis). The function `static_base_shear()` computed
   `V = Z * kp / mu * Sp * Ch * W`, ignoring the Amendment 2 Cl 3.3
   minimum `kpZ = max(kp * Z, 0.08)`. This meant cities with Z < 0.08
   (Brisbane, Hobart) produced incorrect, too-low design shear.
   Fix: `static_base_shear()` now computes and returns `kpZ` as a fourth
   tuple element. (The notebook `comprehensive_assessment.py` always had
   this correct; only the `src/` modules were affected.)

2. **PFA computed as pseudo-acceleration**. The old code used
   `omega1**2 * displacement`, which is the pseudo-acceleration of a
   single-mass oscillator and is wrong for floor-acceleration
   reporting. Fix: PFA now computed via numerical second derivative
   of relative displacement plus ground acceleration:
   `a_abs(t) = a_ground(t) + d2/dt2[u_rel(t)]`, matching the thesis
   description and the notebook implementation.

3. **OpenSeesPy analysis-setup warnings** ("can't set handler after
   analysis is created"). The transient analysis was being configured
   without first wiping the static analysis object from the gravity
   step. Fix: `ops.wipeAnalysis()` is now called before configuring
   the transient analysis.

### Engineering Demand Parameters Added

The thesis advertises 11 EDPs but `src/compliance.py` only computed 4.
Fix: `compute_edps()` now returns all of the following:

4. P-Delta stability coefficient theta per storey (AS 1170.4 Cl 6.5)
5. ASCE 41-17 performance level (IO / LS / CP / Beyond CP)
6. HAZUS-MH damage state (None / Slight / Moderate / Extensive / Complete)
7. Storey lateral stiffness and soft-storey flag (AS 1170.4 Cl 5.2)
8. PFA amplification factors per floor
9. Governing storey identifier
10. Dynamic-to-static shear ratio

### Minor Fixes

11. **Spectral shape discontinuity** at T=0.1 (jump 2.35 → 1.65) and at
    T=1.5 (jump 0.164 → 1.10). The shape factor now uses a single
    descending formula with a 2.35 cap and a continuous long-period
    tail. Affected values: none in the typical 0.2 s ≤ T ≤ 1.5 s range.

12. **Ground motion file deleted prematurely**. The synthetic GM was
    deleted at the end of `run_time_history()`, but the new PFA
    computation needs the ground motion array post-analysis. Fix:
    `generate_synthetic_gm()` now also returns the acceleration array;
    `cleanup_gm_file()` is called by the pipeline after EDP computation.

13. **Test `test_continuity_at_breakpoints` failed**. With the
    spectral-shape fix, the function is now continuous at T=0.1.
    The test has been split into separate continuity checks at
    T=0.1 and T=1.5 with slope-aware tolerances.

### Tests Added

- `TestAmendment2::test_amendment_2_returned` — verifies kpZ is in the
  tuple returned by `static_base_shear()`
- `TestAmendment2::test_newcastle_unaffected` — Z=0.11 > 0.08 leaves
  kpZ unchanged
- `TestAmendment2::test_brisbane_minimum_applied` — Z=0.05 raises to
  kpZ=0.08
- `TestAmendment2::test_brisbane_60pct_higher_than_raw` — verifies
  the 60% base shear increase mandated by Amendment 2
- `test_continuity_at_T_0p1`, `test_continuity_at_T_1p5` — verify
  no jump discontinuity at the spectral-shape breakpoints

### Test Suite Status

- Before: 36 passed, 1 failed (37 total)
- After: 43 passed, 0 failed (43 total)

### Backward Compatibility

`static_base_shear()` now returns 4 values `(V, Ch, T1, kpZ)` instead
of 3. Callers should unpack accordingly. Tests use `V, *_ = ...`
for resilience.

`compute_edps()` now takes additional parameters `W_floor`, `gm_accel`,
`gm_dt` for proper P-Delta and PFA computation. The default `gm_accel=None`
preserves old behaviour (falls back to pseudo-acceleration via omega^2*u).

`run_time_history()` no longer deletes the ground motion file —
callers must use `cleanup_gm_file()` explicitly.

`generate_synthetic_gm()` returns a 4-tuple `(file, dt, npts, accel)`.

---

## v2.0.1 — Notebook bug-fix sweep (May 2026)

Extended the v2.0.0 fixes from `src/` to all notebooks. Pre-v2.0.1
audit found:

| File | Bug 1 (Amdt 2) | Bug 2 (Ch cont.) | Bug 3 (wipe) | Bug 4 (PFA) |
|------|-----|-----|-----|-----|
| `seismic_assessment_UTS_EGP42003.ipynb` | MISSING | DISCONTINUOUS | MISSING | OK |
| `building1_verified.py` | MISSING | OK (no spectral fn) | MISSING | PSEUDO |
| `comprehensive_assessment.py` | OK | DISCONTINUOUS | MISSING | OK |
| `demo_single_cell.py` | MISSING | DISCONTINUOUS | MISSING | PSEUDO |
| `n_floor_pipeline.py` | MISSING | DISCONTINUOUS | MISSING | PSEUDO |
| `n_storey_assessment.py` | MISSING | DISCONTINUOUS | MISSING | PSEUDO |

After v2.0.1: all 6 files have Amendment 2 applied, continuous Ch
function, and `ops.wipeAnalysis()` before the transient analysis
setup. Locations with the pseudo-acceleration PFA formula have been
annotated with comments referencing `src/compliance.py` for the
proper implementation; the production pipeline (`src/pipeline.py`)
already uses the correct formula.

### Files modified in v2.0.1
- `notebooks/building1_verified.py` (+4 Amendment 2 sites, +1 wipeAnalysis)
- `notebooks/comprehensive_assessment.py` (+1 wipeAnalysis, spectral fix)
- `notebooks/demo_single_cell.py` (+1 Amendment 2, +1 wipeAnalysis, spectral fix)
- `notebooks/n_floor_pipeline.py` (+1 Amendment 2, +1 wipeAnalysis, spectral fix)
- `notebooks/n_storey_assessment.py` (+1 Amendment 2, +1 wipeAnalysis, spectral fix)
- `notebooks/seismic_assessment_UTS_EGP42003.ipynb` (+1 Amendment 2, +1 wipeAnalysis, spectral fix)

### Tests
All 43 existing tests in `tests/test_*.py` continue to pass.
