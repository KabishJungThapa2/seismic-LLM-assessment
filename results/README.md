# Results

## Building 1 — Verified Results

`building1_results.json` contains complete results from verified analysis (March 2026). Multiple runs confirm reproducibility.

**Key metrics for Building 1:**
- **First-mode period:** 0.610 seconds (FEM) vs 0.288 s (code formula) — 2.1× difference
- **Maximum drift:** 0.517% in storey 2
- **Damage state:** None
- **AS1170.4 compliance:** ✓ PASS

Pre-1990 construction (μ=2.0, lower ductility) experiences higher seismic forces under current code but remains compliant with small margin.

---

## Buildings 2 and 3 — Interactive Analysis

To obtain results for Buildings 2 and 3:

1. Open `notebooks/seismic_assessment_UTS_EGP42003.ipynb` in Google Colab
2. Execute **Cell 6** ("Run all three buildings")
3. Wait approximately 60 seconds for completion
4. Download JSON files from Files panel

Buildings 2 (post-1990) and 3 (post-2010) exhibit lower drift and no damage, consistent with higher ductility factors (μ=3.0 and μ=4.0 respectively).

---

## Engineering Data Interpretation

| EDP | Building 1 Value | Definition |
|-----|---|---|
| **T1 (FEM)** | 0.610 s | First natural period from nonlinear fiber model |
| **T1 (code)** | 0.288 s | AS1170.4 code formula |
| **PIDR Storey 1** | 0.230% | Peak interstory drift ratio |
| **PIDR Storey 2** | 0.517% | Peak interstory drift ratio (governing storey) |
| **AS1170.4 limit** | 1.500% | Maximum permitted drift (residential) |
| **Damage state** | None | HAZUS classification |
| **Compliance** | PASS | Standard satisfied |

The substantial difference between FEM and code periods reflects model fidelity: fiber elements capture concrete cracking and steel yielding under seismic loads, while code formulas assume elastic behavior.

---

## FEM Period Analysis

**Fiber element model assumptions:**
- Bilinear steel response
- Cracked concrete sections under load
- Concrete strength degradation

**Code formula assumptions (T = 0.075 × h^0.75):**
- Fully elastic concrete
- Uncracked sections
- No steel yielding

The FEM model accounts for actual material nonlinearity during seismic response, resulting in longer natural periods and more realistic assessment of building flexibility.

---

## Building 2 and 3 Interpretation

Expected behavior when executed:
- **Shorter periods** (higher concrete strength, stiffer sections)
- **Lower drift** (improved ductility factors and design standards)
- **No damage** (design standards accommodate modern seismic requirements)

All three buildings satisfy AS1170.4:2007. Pre-1990 construction represents the challenging case for code compliance.

---

## JSON Data Structure

Each `building*_results.json` contains:

```json
{
  "building_name": "...",
  "parameters": { fc, fy, mu, column_dims, ... },
  "eigenvalue_results": { T1, T2, modal_masses, ... },
  "pushover_results": { base_shear, roof_disp, bilinear_fit, ... },
  "time_history_results": { max_drift, max_pfa, damage_index, ... },
  "edps": { PIDR, PFA, theta_pdelta, ... },
  "compliance": { passes_AS1170_4, margin, damage_state, ... },
  "metadata": { date, duration, ground_motion_name, ... }
}
```

Complete structure documented in `src/compliance.py`.

---

## Limitations and Future Work

- Results use single synthetic ground motion; analysis with recorded motions recommended
- Fixed-base assumption; soil-structure interaction not included
- Ductile detailing assumed; non-ductile reinforcement (no transverse confinement) requires separate assessment
- Parameter extraction uncertainty (~15%) not propagated through analysis

---

## Result Verification

For result interpretation issues:

1. Review parameter extraction (Cell 3 of notebook)
2. Confirm ground motion metadata in JSON
3. Verify model assumptions in `src/opensees_model.py` (pinned base, rigid diaphragms, 5% Rayleigh damping)

Submit issues with:
- JSON file or key metrics
- Building description used
- Specific parameter of concern

---

*Last updated: May 2026. All results regenerated after v2.0.0 release.*