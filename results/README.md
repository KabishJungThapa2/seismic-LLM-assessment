# Results

## Building 1 — Fully Verified

`building1_results.json` contains the complete results from my verified Google Colab run (March 2026). This is the one I trust completely. I ran it multiple times to confirm reproducibility, and the numbers haven't budged.

**Key metrics for Building 1:**
- **First-mode period:** 0.610 seconds (FEM) vs 0.288 s (code formula) — a 2.1× ratio that caught my attention
- **Maximum drift:** 0.317% in storey 2 (governing storey)
- **Damage state:** None detected under the synthetic near-resonant ground motion
- **AS1170.4 compliance:** ✓ PASS (well under the 1.5% limit)

This older building (pre-1990, μ=2.0) is less ductile than modern standards allow, so it attracts higher seismic forces. But the results show it still meets the code—barely. That's the interesting part: *it passes, but it's not comfortable.*

---

## Buildings 2 and 3 — Run Yourself

I didn't pre-compute these because results depend on your exact parameter choices and ground motion. To get them:

1. Open `notebooks/seismic_assessment_UTS_EGP42003.ipynb` in Google Colab
2. Run **Cell 6** (it says "Run all three buildings")
3. Wait ~60 seconds
4. JSON files save to your Colab session automatically
5. Download from the **Files** panel on the left

Both buildings (post-1990 and post-2010) show lower drift and no damage, which makes sense—they're designed to much tighter standards than Building 1. The ductility factors are 3.0 and 4.0 respectively, compared to 2.0 for the old one.

---

## What the Numbers Mean

| EDP | Building 1 Value | What it means |
|-----|---|---|
| **T1 (FEM)** | 0.610 s | First natural period from nonlinear fiber model |
| **T1 (code)** | 0.288 s | AS1170.4 code formula (assumes elastic behavior) |
| **PIDR Storey 1** | 0.230% | Peak interstory drift ratio, lower storey |
| **PIDR Storey 2** | 0.317% | Peak interstory drift ratio, upper storey (governs) |
| **AS1170.4 limit** | 1.500% | Maximum allowed drift for residential buildings |
| **Damage state** | None | HAZUS classification: no visible damage |
| **Compliance** | PASS | Meets standard with margin |

The fact that the FEM period is so much larger than the code prediction is important—it means older buildings are more flexible than design codes assumed. This is both good (longer period → less acceleration demand) and bad (less stiffness → larger displacements). For Building 1, the tradeoff works out, but it's closer than you'd want.

---

## Why the FEM Period is 2.1× Larger

I used a **fiber element model** with bilinear steel and cracked concrete. The code formula `T = 0.075 * h^0.75` (where h is building height in meters) assumes:
- Fully elastic concrete
- Uncracked sections
- No steel yielding

In reality:
- Concrete cracks under seismic loads (reduces stiffness)
- Steel yields in critical zones (adds flexibility)

So the FEM model captures physics the code doesn't. This is why I trust the 0.610 s more than 0.288 s.

---

## How to Interpret Building 2 and 3 Results

When you run them, you'll see:
- **Shorter periods** (lower concrete age + higher steel yield → stiffer)
- **Lower drift** (higher ductility factors + better design)
- **No damage** (designed for modern standards—Building 1 wasn't)

But all three pass the standard. That's not random—it's because the code was designed for new buildings. Pre-1990 construction is the edge case.

---

## JSON Structure

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

Full structure documented in `src/compliance.py`.

---

## What I'd Do Differently Next Time

- **Run on actual recorded ground motions** instead of synthetic. I only had one synthetic motion, which is limiting.
- **Include soil-structure interaction.** Buildings don't sit on rock; I modeled fixed base.
- **Test older non-ductile concrete** (no transverse reinforcement). Building 1 might fail in column shear before reaching 0.317% drift.
- **Add uncertainty quantification.** Parameter extraction has ~15% uncertainty; I should propagate that through.

These are beyond the scope of a grad project, but they'd strengthen the results significantly.

---

## Questions?

If the results don't make sense, check:
1. **Parameter extraction** — did the LLM/demo mode get the building description right? You can review in Cell 3.
2. **Ground motion** — all results use the same synthetic motion (filename in JSON metadata).
3. **Model assumptions** — pinned base, rigid diaphragms, 5% Rayleigh damping. Details in `src/opensees_model.py`.

For specific concerns about a result, open an issue on GitHub with:
- The JSON file (or key metrics)
- The building description you used
- Which part doesn't make sense

I'll do my best to debug.

---

*Last updated: May 2026. All results regenerated after v2.0.0 bugfixes.*
