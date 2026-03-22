# LLM-Orchestrated Seismic Vulnerability Assessment
### University of Technology Sydney — Engineering Graduate Project PG (42003)

**Student:** Kabish Jung Thapa (25631413)
**Supervisor:** Prof. Jianchun Li
**Subject:** 42003 Engineering Graduate Project PG — Autumn 2026
**Standard:** AS1170.4:2007 (Australian Earthquake Standard)

---

## What This Is

A proof-of-concept system that assesses the seismic vulnerability of a residential building from a plain English description. You describe the building — the system extracts structural parameters, runs a nonlinear structural analysis, and checks compliance with Australian earthquake code AS1170.4.

**No structural engineering software experience required to run it.**

```
"A 2-storey brick home in Newcastle built around 1975, 12m x 8m floor plan"
        ↓  keyword extraction (demo mode) or Claude API
{ fc: 20 MPa, fy: 250 MPa, mu: 2.0, col: 300x300mm, ... }
        ↓  OpenSeesPy nonlinear time-history analysis
PIDR = 0.317%  |  Damage: None  |  AS1170.4: COMPLIANT
```

---

## Quick Start — Google Colab (No Installation)

1. Open [Google Colab](https://colab.research.google.com)
2. Go to **File → Upload notebook**
3. Upload `notebooks/seismic_assessment_UTS_EGP42003.ipynb`
4. Run cells **top to bottom** (Shift+Enter each)
5. When Cell 6 runs, all three buildings are assessed automatically

**No API key needed.** Demo mode runs entirely on keyword-based extraction.

---

## Repository Structure

```
seismic-llm-assessment/
│
├── notebooks/
│   ├── seismic_assessment_UTS_EGP42003.ipynb  ← MAIN: Upload this to Colab
│   ├── demo_single_cell.py                    ← Alternative: paste into one Colab cell
│   └── building1_verified.py                  ← Standalone verified Building 1 script
│
├── src/
│   ├── config.py          ← Era defaults, AS1170.4 constants, Claude model config
│   ├── extractor.py       ← LLM + demo parameter extraction
│   ├── opensees_model.py  ← OpenSeesPy model builder (RCFrameModel class)
│   ├── analysis.py        ← Time-history analysis engine
│   ├── compliance.py      ← AS1170.4 EDP computation and compliance checking
│   └── pipeline.py        ← Main orchestrator (run this locally)
│
├── tests/
│   ├── test_extractor.py  ← 35 unit tests for parameter extraction
│   └── test_compliance.py ← 11 unit tests for AS1170.4 calculations
│
├── docs/
│   ├── LLM_CHOICE.md      ← Why Claude over GPT-4o (5 reasons)
│   └── KNOWN_ISSUES.md    ← Documented OpenSeesPy bugs and fixes
│
├── results/
│   └── building1_results.json  ← Verified Building 1 results
│
├── requirements.txt
└── README.md
```

---

## Verified Results — Three Newcastle Case Study Buildings

All buildings: 2 storeys, 3 bays x 4.0 m, 8 m wide, Newcastle (Z=0.11, Site De)

| Building | Era | f'c | fy | mu | T1 FEM | Max PIDR | Damage | AS1170.4 |
|---|---|---|---|---|---|---|---|---|
| Building 1 | Pre-1990 | 20 MPa | 250 MPa | 2.0 | 0.610 s | 0.317% | None | PASS |
| Building 2 | Post-1990 | 32 MPa | 500 MPa | 3.0 | TBC | TBC | TBC | TBC |
| Building 3 | Post-2010 | 40 MPa | 500 MPa | 4.0 | TBC | TBC | TBC | TBC |

> Building 1 results are verified. Buildings 2 and 3 run automatically in Cell 6 of the notebook.

**Key finding — T1 ratio:** The FEM period (0.610 s) is 2.1x longer than the code approximation (0.288 s). This is physically correct — the AS1170.4 formula is calibrated for elastic sections. The fibre model captures actual cracked-section stiffness.

**Key finding — Design base shear:** Pre-1990 buildings attract 2.2x the design base shear of post-2010 buildings due to lower ductility factor (mu=2.0 vs 4.0), yet have lower structural capacity. This represents the highest seismic risk category in the Newcastle residential stock.

---

## What the Notebook Produces

Per building (Cells 1–5 define, Cell 6 runs):
- **8-panel results figure** — displacement, drift, pushover curve, drift profile, floor accelerations, damage state indicator, hysteresis loop, fragility bar chart
- **Fragility curves** — lognormal P(DS >= ds | PIDR) for Slight/Moderate/Extensive/Complete
- **Comparison chart** — all buildings side by side (drift, period, base shear, pushover, PFA, summary table)
- **Convergence report** — algorithm usage and convergence rate per building
- **JSON report** — all EDPs, pushover results, fragility probabilities, convergence log

---

## LLM Mode — Claude vs Demo

The pipeline has two extraction modes:

**Demo mode (default — no API key needed)**
```python
# Just press Enter when asked for API key
params = extract_parameters("A 2-storey home in Newcastle built 1975...")
```

**Claude API mode (optional)**
```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
params = extract("A 2-storey home in Newcastle built 1975...", api_key=os.environ["ANTHROPIC_API_KEY"])
```
Get a free API key at [console.anthropic.com](https://console.anthropic.com).
Students can apply for up to USD 300 in free credits at [anthropic.com/contact-sales/for-student-builders](https://www.anthropic.com/contact-sales/for-student-builders).

**Why Claude over GPT-4o** — see [docs/LLM_CHOICE.md](docs/LLM_CHOICE.md). Summary:
1. Constitutional AI training — more explicit about uncertainty (safer for structural assessment)
2. ~6-10x cheaper per extraction than GPT-4o
3. Research novelty — Liang et al. (2025a) used GPT-4o; using Claude is an independent contribution
4. More comprehensive assumption flagging in structured JSON output
5. Anthropic's open research culture aligns better with academic reproducibility norms

---

## Critical Technical Finding — OpenSeesPy ARPACK Bug

**This affects any OpenSeesPy model using rigid diaphragm constraints.**

**Symptom:**
```
ArpackSolver::Error with _saupd info = -9
Starting vector is zero.
WARNING StaticAnalysis::eigen() - EigenSOE failed
```

**Fix:**
```python
# WRONG — fails with equalDOF constraints
eigenvalues = ops.eigen(num_modes)

# CORRECT
eigenvalues = ops.eigen('-fullGenLapack', num_modes)

# Also required: assign mass ONLY to master nodes
for fi in range(1, num_storeys + 1):
    ops.mass(node_id[fi][0], M_floor, M_floor, 0.0)
```

Full explanation in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

---

## Running Tests Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/seismic-llm-assessment.git
cd seismic-llm-assessment

# Install
pip install -r requirements.txt

# Run tests (does not require OpenSeesPy — tests only extractor and compliance)
cd src
python -m pytest ../tests/ -v
```

**Note:** Google Colab is the recommended environment. OpenSeesPy has known compatibility issues on macOS Apple Silicon (M1/M2/M3). See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

---

## Pipeline Architecture

```
User Input (plain English description)
         │
         ▼
 ┌───────────────────┐
 │  LLM Extraction   │  ← Claude API or demo keyword mode
 │  (extractor.py)   │
 └────────┬──────────┘
          │ JSON params
          ▼
 ┌────────────────────────┐
 │ Human Verification     │  ← Engineer reviews ALL parameters
 │ Checkpoint             │  ← Can approve / edit / reject
 └────────┬───────────────┘
          │ approved params
          ▼
 ┌────────────────────────────────────────────┐
 │  OpenSeesPy Analysis Engine                │
 │  ┌─────────────┐  ┌──────────────────────┐│
 │  │ Static      │  │ Eigenvalue (T1, T2)  ││
 │  │ AS1170.4    │  │ fullGenLapack solver  ││
 │  └─────────────┘  └──────────────────────┘│
 │  ┌─────────────┐  ┌──────────────────────┐│
 │  │ Pushover    │  │ Time-History         ││
 │  │ 3% drift    │  │ Newmark + Rayleigh   ││
 │  └─────────────┘  └──────────────────────┘│
 └────────┬───────────────────────────────────┘
          │ EDPs
          ▼
 ┌───────────────────────────────┐
 │  Post-Processing              │
 │  PIDR | PFA | Damage State    │
 │  Fragility Curves | JSON      │
 └───────────────────────────────┘
```

---

## References

- Liang, H., Talebi Kalaleh, M., & Mei, Q. (2025a). Integrating large language models for automated structural analysis. *arXiv:2504.09754*
- Liang, H. et al. (2025b). MASSE: Multi-agent system for structural engineering. *arXiv:2510.11004*
- Standards Australia. (2007). *AS1170.4:2007 — Structural design actions: Earthquake actions*
- Anthropic. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv:2212.08073*
- FEMA. (2003). *HAZUS-MH Technical Manual*. Federal Emergency Management Agency.

---

## Citation

```bibtex
@misc{thapa2026seismic,
  author     = {Thapa, Kabish Jung},
  title      = {LLM-Orchestrated Seismic Vulnerability Assessment of Residential Buildings},
  year       = {2026},
  school     = {University of Technology Sydney},
  note       = {Engineering Graduate Project PG (42003), supervised by Prof. Jianchun Li},
  url        = {https://github.com/YOUR_USERNAME/seismic-llm-assessment}
}
```

---

## Licence

MIT Licence — free to use, modify, and distribute with attribution.

---

*Built with OpenSeesPy, Claude (Anthropic), and Python. Assessed under AS1170.4:2007.*
