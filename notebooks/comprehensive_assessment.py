# =============================================================================
# COMPREHENSIVE RC FRAME SEISMIC ASSESSMENT — N STOREYS
# LLM-Orchestrated Workflow | UTS Engineering Graduate Project PG (42003)
# Kabish Jung Thapa (25631413) | Supervisor: Prof. Jianchun Li
#
# STANDARD: AS 1170.4:2007 incorporating Amendment No. 2 (February 2018)
#
# AMENDMENT 2 CHANGES APPLIED (AS 1170.4/Amdt 2/2018-02-22):
#   Cl 3.3 & Table 3.3 : Minimum kpZ = 0.08 for 1/500 exceedance
#   Cl 1.4 / Def.      : 'hazard factor' renamed 'hazard design factor'
#   Table 3.2           : Cities with Z ≤ 0.08 removed (floor enforced via Cl 3.3)
#   Clause 4.2.2(a)     : Soil Class A compressive strength range updated
#
# ENGINEERING ANALYSES PERFORMED:
#   1.  Equivalent static base shear  — AS1170.4 Cl 6.2 + Amd 2 Cl 3.3
#   2.  Lateral force distribution    — AS1170.4 Cl 6.3 (exponent k, T-dependent)
#   3.  Eigenvalue analysis           — fullGenLapack, N_MODES modes
#   4.  Modal mass participation      — effective mass ratios per mode
#   5.  Nonlinear time-history        — Newmark average acceleration, 5% Rayleigh
#   6.  Structural irregularity       — AS1170.4 Cl 5.2: stiffness, mass, geometry
#   7.  Storey shear profile          — from inertial forces (dynamic) + static
#   8.  Overturning moment profile    — M_OTM at each level, base uplift check
#   9.  P-Delta stability coefficient — AS1170.4 Cl 6.5: theta_i per storey
#  10.  Storey stiffness              — k_i = V_i/delta_i, soft-storey detection
#  11.  Inter-storey drift            — PIDR per storey, governing floor
#  12.  Peak floor acceleration       — numerical 2nd derivative (absolute)
#  13.  Column axial load ratio       — N/Nuo per AS3600 (ductility check)
#  14.  Performance level             — IO / LS / CP per ASCE 41 / ATC-40
#  15.  HAZUS damage state            — None / Slight / Moderate / Extensive / Complete
#  16.  Engineering dashboard         — 12-panel figure
#  17.  Comprehensive JSON report
#
# HOW TO RUN IN GOOGLE COLAB:
#   Cell 1: !pip install openseespy numpy matplotlib scipy -q
#   Cell 2: Paste this script → Shift+Enter
# =============================================================================

import subprocess, sys
subprocess.run(['pip', 'install', 'openseespy', 'numpy', 'matplotlib', 'scipy', '-q'],
               check=True)
if 'openseespy' in sys.modules:
    del sys.modules['openseespy']

import openseespy.opensees as ops
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import scipy.stats as stats
import tempfile, os, json
from datetime import datetime

print("All packages ready.\n")

# =============================================================================
# SECTION 1: BUILDING PARAMETERS — EDIT HERE
# =============================================================================

BUILDING_NAME = "Pre-1990 Non-Ductile RC Frame"

# ── Geometry ──────────────────────────────────────────────────────────────────
num_storeys   = 4       # ← CHANGE THIS: 1–8 storeys supported
storey_height = 3.0     # m — uniform floor height
num_bays      = 3       # bays in earthquake (X) direction
bay_width     = 4.0     # m per bay
floor_width   = 8.0     # m — building width perpendicular to frame

# ── Materials ─────────────────────────────────────────────────────────────────
fc  = 20.0    # MPa — concrete compressive strength
fy  = 250.0   # MPa — steel yield strength
Es  = 200000.0
Ec  = 0.043 * (2400**1.5) * (fc**0.5)   # MPa — AS3600

# ── Beams (uniform all floors) ────────────────────────────────────────────────
beam_b = 0.30   # m
beam_h = 0.45   # m

# ── Columns — per storey (lower floors larger, AS3600 practice) ───────────────
# Key = storey index (1 = ground storey, n = top storey)
COLUMN_SIZES = {
    1: (0.30, 0.30),
    2: (0.30, 0.30),
    3: (0.25, 0.25),
    4: (0.25, 0.25),
    5: (0.25, 0.25),
    6: (0.20, 0.20),
    7: (0.20, 0.20),
    8: (0.20, 0.20),
}

def get_col_size(floor_idx):
    if floor_idx in COLUMN_SIZES:
        return COLUMN_SIZES[floor_idx]
    return COLUMN_SIZES[max(COLUMN_SIZES.keys())]

# ── Reinforcement ─────────────────────────────────────────────────────────────
col_rho    = 0.015   # longitudinal steel ratio (all floors)
beam_rho_t = 0.008   # tension steel
beam_rho_c = 0.004   # compression steel

# ── Confinement (pre-1990: poor — widely spaced stirrups) ────────────────────
epsc0_core = -0.004
epsU_core  = -0.012

# ── Seismic parameters — AS1170.4:2007 + Amendment 2 (2018) ─────────────────
# Amendment 2 Cl 3.3: minimum kpZ = 0.08 for P(exceedance) = 1/500
# This is enforced automatically in static_analysis() below.
Z_raw      = 0.11    # hazard design factor Z (Amendment 2 terminology)
kp         = 1.0     # probability factor — Importance Level 2 (residential), 500yr
kpZ        = max(kp * Z_raw, 0.08)   # ← Amendment 2 Cl 3.3 minimum applied
Z          = kpZ     # use kpZ as effective Z throughout
site_class = "De"    # soft soil — coastal Newcastle/Sydney

mu         = 2.0     # structural ductility factor (pre-1990: limited ductility)
Sp         = 0.77    # structural performance factor
drift_limit = 0.015  # 1.5% — AS1170.4 Cl 5.4.4

# ── Gravity loads ─────────────────────────────────────────────────────────────
dead_load  = 5.0    # kPa superimposed dead load
live_load  = 2.0    # kPa residential (AS1170.1)

# ── Constants ─────────────────────────────────────────────────────────────────
COVER    = 0.040   # m concrete cover
G        = 9.81    # m/s²
N_MODES  = min(num_storeys, 4)

# =============================================================================
# SECTION 2: DERIVED QUANTITIES
# =============================================================================

fc_kN  = fc  * 1000
fy_kN  = fy  * 1000
Es_kN  = Es  * 1000
Ec_kN  = Ec  * 1000

floor_area  = num_bays * bay_width * floor_width
W_floor     = (dead_load + 0.3 * live_load) * floor_area     # seismic weight per floor (kN)
W_total     = W_floor * num_storeys
M_floor     = W_floor / G                                      # mass per floor (kN.s2/m)
P_int       = (dead_load + live_load) * (floor_width/2) * bay_width
P_ext       = P_int / 2
Hn          = num_storeys * storey_height

# ── Spectral shape factor Ch(T) — Site De (AS1170.4 Table 6.4) ───────────────
T1_approx = 0.075 * Hn**0.75   # code empirical period (Appendix B)

def Ch_De(T):
    """Spectral shape factor for Site Class De. AS1170.4 Table 6.4."""
    if   T <= 0.10: return 2.35
    elif T <  1.50: return 1.65 * (0.1 / T) ** 0.85
    else:           return 1.10 * (1.5 / T) ** 2.0

# ── Performance level thresholds (ASCE 41 / ATC-40 adapted for RC) ───────────
PERFORMANCE_LIMITS = {
    "Immediate Occupancy (IO)": 0.005,   # PIDR < 0.5%
    "Life Safety (LS)":         0.015,   # PIDR < 1.5%  ← matches AS1170.4 limit
    "Collapse Prevention (CP)": 0.025,   # PIDR < 2.5%
}

# ── HAZUS-MH damage thresholds (FEMA 2003, adapted for low-rise RC) ──────────
DAMAGE_STATES = [
    ("None",      0.000, 0.005, 0.00),
    ("Slight",    0.005, 0.010, 0.05),
    ("Moderate",  0.010, 0.020, 0.20),
    ("Extensive", 0.020, 0.040, 0.50),
    ("Complete",  0.040, 9.999, 1.00),
]

# ── Print parameter summary ───────────────────────────────────────────────────
print("=" * 68)
print(f"  {BUILDING_NAME}  ({num_storeys} storeys)")
print("  AS 1170.4:2007 + Amendment No. 2 (February 2018)")
print("=" * 68)
print(f"  Hn          = {Hn:.1f} m  |  num_bays = {num_bays}  |  bay_width = {bay_width} m")
print(f"  fc = {fc} MPa  |  fy = {fy} MPa  |  mu = {mu}  |  Sp = {Sp}")
print(f"  Z_raw = {Z_raw}  |  kp = {kp}  |  kpZ = {kpZ:.3f} (Amd2 min 0.08 applied)")
print(f"  W_floor = {W_floor:.1f} kN  |  W_total = {W_total:.1f} kN  |  M_floor = {M_floor:.3f} kN.s2/m")
print(f"  T1_approx (code) = {T1_approx:.3f} s  |  N_MODES = {N_MODES}")

# =============================================================================
# SECTION 3: AS1170.4:2007 + AMD 2 STATIC ANALYSIS
# =============================================================================

def static_analysis():
    """
    Equivalent static base shear and lateral force distribution.
    Implements AS1170.4:2007 Cl 6.2 and Cl 6.3 with Amendment 2 (2018).

    Key formula: V = (kpZ / mu) * Sp * Ch(T1) * W   [Cl 6.2]
    Minimum:     V >= 0.01 * W                        [Cl 6.2.3]
    Amendment 2: kpZ >= 0.08                           [Cl 3.3 Table 3.3]

    Lateral force distribution uses the k-exponent method (Cl 6.3):
      k = 1.0 for T1 <= 0.5s (triangular / inverted)
      k = 2.0 for T1 >= 2.5s (parabolic)
      interpolated between
    This more accurately represents higher-mode effects in taller buildings.

    Returns dict with V_static, force distribution, OTM, and all derived values.
    """
    Ch = Ch_De(T1_approx)
    V  = max((kpZ / mu) * Sp * Ch * W_total, 0.01 * W_total)

    # Exponent k for lateral force distribution (AS1170.4 Cl 6.3)
    if   T1_approx <= 0.5: k = 1.0
    elif T1_approx >= 2.5: k = 2.0
    else:                  k = 1.0 + (T1_approx - 0.5) / 2.0

    # Floor heights and weights
    heights = [(i + 1) * storey_height for i in range(num_storeys)]
    weights = [W_floor] * num_storeys   # uniform mass per floor

    # Lateral force at each floor Fi = V * (wi * hi^k) / Σ(wj * hj^k)
    denom  = sum(weights[i] * heights[i]**k for i in range(num_storeys))
    F_lat  = [V * (weights[i] * heights[i]**k) / denom for i in range(num_storeys)]

    # Storey shear from top down: Vi = Σ(Fj) for j >= i (1-based storey index)
    V_storey = np.zeros(num_storeys)
    for i in range(num_storeys - 1, -1, -1):
        V_storey[i] = sum(F_lat[j] for j in range(i, num_storeys))

    # Overturning moment at each level
    # OTM at base of storey i = Σ(Fj × (hj - hi-1)) for j >= i
    # At base (i=0): OTM_base = Σ Fj × hj
    OTM = np.zeros(num_storeys + 1)   # index 0 = base
    for i in range(num_storeys):
        h_base = i * storey_height    # base of storey i+1 (0-indexed)
        OTM[i] = sum(F_lat[j] * (heights[j] - h_base)
                     for j in range(i, num_storeys))
    OTM[-1] = 0.0   # at roof level OTM = 0

    # Column uplift force at base (simplified: OTM / frame width)
    frame_width = num_bays * bay_width
    N_uplift    = OTM[0] / frame_width   # kN tension on leeward column

    print(f"\n  AS1170.4 + Amd 2 Static Analysis:")
    print(f"    kpZ = {kpZ:.3f}  (Amendment 2 min 0.08 applied)")
    print(f"    Ch(T1_approx) = {Ch:.4f}  |  k = {k:.2f}")
    print(f"    V_static = {V:.1f} kN  (V/W = {V/W_total:.4f})")
    print(f"    OTM_base = {OTM[0]:.1f} kN.m  |  N_uplift (exterior col) = {N_uplift:.1f} kN")
    print(f"    Lateral forces per floor (kN): "
          + "  ".join(f"F{i+1}={f:.1f}" for i,f in enumerate(F_lat)))

    return {
        "V_static":  V,
        "Ch":        Ch,
        "k":         k,
        "F_lat":     F_lat,
        "V_storey":  V_storey,
        "OTM":       OTM,
        "N_uplift":  N_uplift,
        "kpZ":       kpZ,
    }

SA = static_analysis()

# =============================================================================
# SECTION 4: STRUCTURAL IRREGULARITY CHECK (AS1170.4 Cl 5.2)
# =============================================================================

def check_irregularity(static_results):
    """
    Pre-analysis irregularity screening per AS1170.4 Cl 5.2.
    Checks stiffness (soft storey), mass, and geometric irregularity.
    These checks use structural properties, not analysis results.

    Stiffness irregularity (soft storey): AS1170.4 / ASCE 7 Cl 12.3.2
      A soft storey exists if lateral stiffness of any storey is
      < 70% of the storey above, OR < 80% of avg of 3 storeys above.
      Here we use gross section stiffness as proxy:
      k_gross_i = (sum of 12EI/h³ for columns in storey i)

    Mass irregularity: any floor mass > 150% of adjacent floor mass.

    Geometric irregularity: any set-back > 130% of adjacent storey width.
    """
    flags = []
    details = {}

    # ── Approximate storey lateral stiffness (elastic, gross section) ────────
    k_gross = []
    for i in range(1, num_storeys + 1):
        col_b, col_h = get_col_size(i)
        Ig  = (col_b * col_h**3) / 12   # m4, strong axis
        EI  = Ec_kN * Ig * 1e-6         # kN.m2 (Ec in kN/m2, Ig in m4)
        h   = storey_height
        # Lateral stiffness of one column: 12EI/h³ (fixed-fixed assumption)
        k_col = 12 * EI / h**3           # kN/m
        n_col = num_bays + 1
        k_gross.append(k_col * n_col)    # total storey stiffness

    details["k_gross_kNm"] = [round(k, 1) for k in k_gross]

    # Soft storey: k_i < 0.70 * k_{i+1}
    soft_storeys = []
    for i in range(num_storeys - 1):
        ratio = k_gross[i] / k_gross[i+1]
        if ratio < 0.70:
            soft_storeys.append(i + 1)
            flags.append(
                f"SOFT STOREY: Storey {i+1} stiffness = {ratio:.2f} × Storey {i+2} "
                f"(< 0.70 limit) — vertical irregularity per AS1170.4 Cl 5.2")
    if not soft_storeys:
        flags.append("Stiffness regularity: OK (no soft storey detected)")

    # ── Mass irregularity ─────────────────────────────────────────────────────
    # Uniform floor mass assumed — flag if different masses were defined
    # (here all floors equal, so trivially regular)
    details["mass_per_floor_kN"] = [round(W_floor, 1)] * num_storeys
    flags.append("Mass irregularity: REGULAR (uniform floor mass)")

    # ── Geometric irregularity ────────────────────────────────────────────────
    # Uniform floor plan assumed (no set-backs)
    flags.append("Geometric irregularity: REGULAR (uniform floor plan)")

    # ── Column height uniformity ──────────────────────────────────────────────
    flags.append(f"Storey heights: REGULAR (uniform {storey_height} m all floors)")

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n  Structural Irregularity Checks (AS1170.4 Cl 5.2):")
    for fl in flags:
        marker = "  [!]" if "SOFT" in fl or "IRREG" in fl else "  [ ]"
        print(f"  {marker} {fl}")

    return {
        "flags":        flags,
        "k_gross":      k_gross,
        "soft_storeys": soft_storeys,
        "is_irregular": len(soft_storeys) > 0,
        "details":      details,
    }

IRREG = check_irregularity(SA)

# =============================================================================
# SECTION 5: BUILD OPENSEESPY MODEL
# =============================================================================

def build_model():
    """
    Build 2D nonlinear fibre RC frame — n storeys × num_bays bays.
    Per-storey column sections (tag = 10 + storey_idx).
    Rigid diaphragm via equalDOF on DOF 1 at each floor.
    Mass on master nodes ONLY (ARPACK incompatibility with slave mass + equalDOF).
    """
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    xs = [j * bay_width     for j in range(num_bays + 1)]
    ys = [i * storey_height for i in range(num_storeys + 1)]

    node_id = []
    for fi, y in enumerate(ys):
        row = []
        for ci, x in enumerate(xs):
            nid = (fi + 1) * 10 + (ci + 1)
            ops.node(nid, x, y)
            row.append(nid)
        node_id.append(row)

    for nid in node_id[0]:
        ops.fix(nid, 1, 1, 1)

    for fi in range(1, num_storeys + 1):
        master = node_id[fi][0]
        for slave in node_id[fi][1:]:
            ops.equalDOF(master, slave, 1)

    # Materials: 1=core concrete, 2=cover concrete, 3=steel
    ops.uniaxialMaterial('Concrete01', 1, -fc_kN, epsc0_core, -0.2*fc_kN, epsU_core)
    ops.uniaxialMaterial('Concrete01', 2, -fc_kN, -0.002,      0.0,        -0.004)
    ops.uniaxialMaterial('Steel01',    3,  fy_kN,  Es_kN,      0.01)

    # Column sections — one per storey, tag = 10 + storey_idx
    for fi in range(1, num_storeys + 1):
        col_b, col_h = get_col_size(fi)
        Ac  = col_b * col_h
        Asc = col_rho * Ac
        cy  = col_h / 2 - COVER
        cz  = col_b / 2 - COVER
        As_bar = max(Asc / 6, 1e-5)
        ops.section('Fiber', 10 + fi)
        ops.patch('rect', 1, 10, 10, -cy, -cz,  cy,  cz)
        ops.patch('rect', 2, 10,  2,  cy, -col_b/2,  col_h/2, col_b/2)
        ops.patch('rect', 2, 10,  2, -col_h/2, -col_b/2, -cy, col_b/2)
        ops.patch('rect', 2,  2, 10, -cy, -col_b/2,  cy, -cz)
        ops.patch('rect', 2,  2, 10, -cy,  cz,        cy,  col_b/2)
        ops.layer('straight', 3, 3, As_bar, -cy, -cz, -cy, cz)
        ops.layer('straight', 3, 3, As_bar,  cy, -cz,  cy, cz)

    # Beam section — uniform, tag = 99
    Ab   = beam_b * beam_h
    Ast  = beam_rho_t * Ab
    Asc2 = beam_rho_c * Ab
    by   = beam_h / 2 - COVER
    bz   = beam_b / 2 - COVER
    ops.section('Fiber', 99)
    ops.patch('rect', 1, 10, 10, -by, -bz, by, bz)
    ops.patch('rect', 2, 10,  2,  by, -beam_b/2,  beam_h/2, beam_b/2)
    ops.patch('rect', 2, 10,  2, -beam_h/2, -beam_b/2, -by, beam_b/2)
    ops.layer('straight', 3, 3, Ast/3,  -by, -bz, -by, bz)
    ops.layer('straight', 3, 3, Asc2/3,  by, -bz,  by, bz)

    ops.geomTransf('PDelta', 1)   # columns — captures P-Delta
    ops.geomTransf('Linear', 2)   # beams

    eid = 100
    for fi in range(num_storeys):
        storey_sec = 10 + (fi + 1)
        for ci in range(num_bays + 1):
            ops.element('nonlinearBeamColumn', eid,
                        node_id[fi][ci], node_id[fi+1][ci], 5, storey_sec, 1)
            eid += 1
    for fi in range(1, num_storeys + 1):
        for ci in range(num_bays):
            ops.element('nonlinearBeamColumn', eid,
                        node_id[fi][ci], node_id[fi][ci+1], 5, 99, 2)
            eid += 1

    n_col = (num_bays + 1) * num_storeys
    n_bm  = num_bays * num_storeys
    print(f"  Model: {(num_bays+1)*(num_storeys+1)} nodes, "
          f"{n_col+n_bm} elements ({n_col} col + {n_bm} beam)")
    return node_id


def gravity_analysis(node_id):
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)
    for fi in range(1, num_storeys + 1):
        for ci, nid in enumerate(node_id[fi]):
            P = -P_ext if (ci == 0 or ci == num_bays) else -P_int
            ops.load(nid, 0.0, P, 0.0)
    ops.system('BandGeneral'); ops.numberer('RCM')
    ops.constraints('Plain');  ops.integrator('LoadControl', 0.1)
    ops.algorithm('Newton');   ops.analysis('Static')
    ok = ops.analyze(10)
    ops.loadConst('-time', 0.0)
    print("  Gravity: " + ("CONVERGED" if ok == 0 else "WARNING"))


def assign_masses(node_id):
    """
    Master-node mass assignment ONLY.
    ARPACK fails if slave nodes carry mass with equalDOF active.
    Use fullGenLapack for eigenvalue — ARPACK incompatible with this config.
    """
    for fi in range(1, num_storeys + 1):
        ops.mass(node_id[fi][0], M_floor, M_floor, 0.0)
    print(f"  Mass: {M_floor:.3f} kN.s2/m per floor (master nodes only)")

# =============================================================================
# SECTION 6: EIGENVALUE + MODAL MASS PARTICIPATION
# =============================================================================

def eigenvalue_analysis():
    """
    Extract natural periods and approximate modal mass participation factors.

    Modal mass participation (shear building approximation):
      For mode i:  Gamma_i = (phi_i^T M 1) / (phi_i^T M phi_i)
      M_eff_i = Gamma_i^2 × (phi_i^T M phi_i) / M_total
    
    Mode shapes approximated using relative floor displacements under
    a static analysis with the code lateral force pattern applied.
    For simplicity here, participation factors are estimated analytically
    for a uniform shear building (conservative approximation).
    """
    eigs   = ops.eigen('-fullGenLapack', N_MODES)
    T_list = [2 * np.pi / abs(eigs[i])**0.5 for i in range(N_MODES)]
    omega1 = abs(eigs[0]) ** 0.5
    omega2 = abs(eigs[1]) ** 0.5 if N_MODES >= 2 else omega1 * 3.0

    # Approximate modal mass participation for uniform shear building
    # Mode 1 participation ≈ 0.85 for n=1, decreases slightly with n
    # From Chopra 2007: for uniform shear building, mode 1 ≈ 8/(pi^2) = 81%
    n = num_storeys
    participation_approx = []
    for i in range(N_MODES):
        # Approximate from uniform shear building theory
        r    = (2*i + 1) * np.pi / (2 * n)   # approximate eigenvalue index
        Meff = (8 / ((2*i+1)**2 * np.pi**2))  # fraction of total mass
        participation_approx.append(min(Meff, 1.0))

    cum_mass = np.cumsum(participation_approx)

    print(f"\n  Eigenvalue analysis (fullGenLapack, {N_MODES} modes):")
    for i, Ti in enumerate(T_list, 1):
        code_str = f"  code approx: {T1_approx:.3f} s" if i == 1 else ""
        mass_str = f"  Meff~{participation_approx[i-1]*100:.0f}%  cumul~{cum_mass[i-1]*100:.0f}%"
        print(f"    Mode {i}: T = {Ti:.4f} s{code_str}")
        print(f"           {mass_str}")

    if cum_mass[-1] < 0.90:
        print("  NOTE: Include more modes for >90% mass participation (taller buildings)")

    return T_list, eigs, omega1, omega2, participation_approx

# =============================================================================
# SECTION 7: PUSHOVER ANALYSIS (Nonlinear Static — ATC-40 / ASCE 41-17)
# =============================================================================

def pushover_analysis(node_id, T_list):
    """
    Displacement-controlled pushover analysis per ATC-40 / ASCE 41-17.

    LOAD PATTERN: Inverted-triangle (proportional to floor height / Hn).
    This is the k=1 distribution from AS 1170.4 Cl 6.3, appropriate for
    T1 ≤ 0.5 s. For taller buildings where k > 1, a parabolic pattern
    should be used.

    CONTROL NODE: Roof master node (node_id[num_storeys][0]).
    Target drift: 3% of Hn (180 mm for 6 m building) — well beyond
    typical collapse prevention limits but needed to capture post-peak.

    OUTPUTS:
      - capacity curve: base shear vs. roof drift (%)
      - V_max: peak base shear
      - Δ_yield: yield displacement (bilinear at 60% V_max secant)
      - ductility ratio: Δ_ultimate / Δ_yield
      - overstrength ratio: V_max / V_static
      - T_eff: effective period at peak response

    NOTE: The model is rebuilt from scratch inside this function to avoid
    contaminating the time-history model state. This is the correct
    OpenSeesPy practice — always wipe() before a new analysis type.
    """
    print("\n  Pushover analysis (displacement-controlled, inverted triangle)")

    # ── Rebuild model for pushover ─────────────────────────────────────────
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    xs = [j * bay_width     for j in range(num_bays + 1)]
    ys = [i * storey_height for i in range(num_storeys + 1)]

    node_id_po = []
    for fi, y in enumerate(ys):
        row = []
        for ci, x in enumerate(xs):
            nid = (fi + 1) * 10 + (ci + 1)
            ops.node(nid, x, y)
            row.append(nid)
        node_id_po.append(row)

    for nid in node_id_po[0]:
        ops.fix(nid, 1, 1, 1)

    for fi in range(1, num_storeys + 1):
        master = node_id_po[fi][0]
        for slave in node_id_po[fi][1:]:
            ops.equalDOF(master, slave, 1)

    ops.uniaxialMaterial('Concrete01', 1, -fc_kN, epsc0_core, -0.2*fc_kN, epsU_core)
    ops.uniaxialMaterial('Concrete01', 2, -fc_kN, -0.002,      0.0,        -0.004)
    ops.uniaxialMaterial('Steel01',    3,  fy_kN,  Es_kN,      0.01)

    for fi in range(1, num_storeys + 1):
        col_b, col_h = get_col_size(fi)
        Ac  = col_b * col_h
        Asc = col_rho * Ac
        cy  = col_h / 2 - COVER
        cz  = col_b / 2 - COVER
        As_bar = max(Asc / 6, 1e-5)
        ops.section('Fiber', 10 + fi)
        ops.patch('rect', 1, 10, 10, -cy, -cz, cy, cz)
        ops.patch('rect', 2, 10,  2,  cy, -col_b/2, col_h/2, col_b/2)
        ops.patch('rect', 2, 10,  2, -col_h/2, -col_b/2, -cy, col_b/2)
        ops.patch('rect', 2,  2, 10, -cy, -col_b/2, cy, -cz)
        ops.patch('rect', 2,  2, 10, -cy, cz, cy, col_b/2)
        ops.layer('straight', 3, 3, As_bar, -cy, -cz, -cy, cz)
        ops.layer('straight', 3, 3, As_bar,  cy, -cz,  cy, cz)

    Ab   = beam_b * beam_h
    Ast  = beam_rho_t * Ab
    Asc2 = beam_rho_c * Ab
    by   = beam_h / 2 - COVER
    bz   = beam_b / 2 - COVER
    ops.section('Fiber', 99)
    ops.patch('rect', 1, 10, 10, -by, -bz, by, bz)
    ops.patch('rect', 2, 10,  2,  by, -beam_b/2, beam_h/2, beam_b/2)
    ops.patch('rect', 2, 10,  2, -beam_h/2, -beam_b/2, -by, beam_b/2)
    ops.layer('straight', 3, 3, Ast/3,  -by, -bz, -by, bz)
    ops.layer('straight', 3, 3, Asc2/3,  by, -bz,  by, bz)

    ops.geomTransf('PDelta', 1)
    ops.geomTransf('Linear', 2)

    eid = 100
    for fi in range(num_storeys):
        for ci in range(num_bays + 1):
            ops.element('nonlinearBeamColumn', eid,
                        node_id_po[fi][ci], node_id_po[fi+1][ci],
                        5, 10 + (fi + 1), 1)
            eid += 1
    for fi in range(1, num_storeys + 1):
        for ci in range(num_bays):
            ops.element('nonlinearBeamColumn', eid,
                        node_id_po[fi][ci], node_id_po[fi][ci+1],
                        5, 99, 2)
            eid += 1

    # ── Gravity loads (constant throughout pushover) ───────────────────────
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)
    for fi in range(1, num_storeys + 1):
        for ci, nid in enumerate(node_id_po[fi]):
            P = -P_ext if (ci == 0 or ci == num_bays) else -P_int
            ops.load(nid, 0.0, P, 0.0)
    ops.system('BandGeneral'); ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.integrator('LoadControl', 0.1)
    ops.algorithm('Newton'); ops.analysis('Static')
    ops.analyze(10)
    ops.loadConst('-time', 0.0)

    # ── Lateral load pattern — inverted triangle ───────────────────────────
    # F_i ∝ W_i × h_i / Σ(W_j × h_j)  [k=1, AS 1170.4 Cl 6.3]
    heights  = [(i + 1) * storey_height for i in range(num_storeys)]
    denom    = sum(heights)
    # Normalised lateral forces sum to 1.0 (scaled by actual base shear)
    F_norm   = [h / denom for h in heights]

    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    for fi in range(1, num_storeys + 1):
        nid = node_id_po[fi][0]   # master node
        ops.load(nid, F_norm[fi - 1], 0.0, 0.0)

    # ── Displacement-controlled pushover ───────────────────────────────────
    control_node = node_id_po[num_storeys][0]   # roof master node
    target_disp  = 0.03 * Hn                    # 3% roof drift = 0.18 m
    n_steps      = 200
    d_step       = target_disp / n_steps

    ops.system('UmfPack'); ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.test('NormDispIncr', 1.0e-8, 50, 0)
    ops.integrator('DisplacementControl', control_node, 1, d_step)
    ops.algorithm('Newton')
    ops.analysis('Static')

    roof_disp  = [0.0]
    base_shear = [0.0]
    n_fail     = 0

    for step in range(n_steps):
        ok = ops.analyze(1)
        if ok != 0:
            n_fail += 1
            ops.algorithm('KrylovNewton')
            ok = ops.analyze(1)
            if ok != 0:
                ops.test('NormDispIncr', 1.0e-6, 100, 0)
                ops.algorithm('ModifiedNewton', '-initial')
                ok = ops.analyze(1)
            ops.algorithm('Newton')
            ops.test('NormDispIncr', 1.0e-8, 50, 0)
            if ok != 0:
                print(f"  Pushover: convergence failure at step {step+1} "
                      f"(drift {roof_disp[-1]/Hn*100:.2f}%) — stopping")
                break

        u_roof = ops.nodeDisp(control_node, 1)
        roof_disp.append(u_roof)

        # Base shear = sum of reaction forces at ground nodes (DOF 1)
        V_base = sum(abs(ops.nodeReaction(nid, 1))
                     for nid in node_id_po[0])
        base_shear.append(V_base)

    roof_disp  = np.array(roof_disp)
    base_shear = np.array(base_shear)
    roof_drift_pct = roof_disp / Hn * 100   # %

    # ── Post-process capacity curve ────────────────────────────────────────
    V_max     = float(np.max(base_shear))
    idx_vmax  = int(np.argmax(base_shear))
    u_at_vmax = roof_disp[idx_vmax]

    # Bilinear yield: secant stiffness at 60% V_max through origin
    V_60     = 0.60 * V_max
    # Find first point on curve at or above V_60
    above_60 = np.where(base_shear >= V_60)[0]
    if len(above_60) > 0:
        idx_60   = above_60[0]
        k_secant = base_shear[idx_60] / max(roof_disp[idx_60], 1e-6)   # kN/m
        u_yield  = V_max / k_secant   # projected yield displacement
    else:
        u_yield  = u_at_vmax * 0.6

    u_ultimate   = roof_disp[-1]   # last converged point
    ductility    = float(u_ultimate / max(u_yield, 1e-6))
    overstrength = V_max / max(SA['V_static'], 1e-3)

    # Effective period at peak (secant stiffness method)
    k_eff_push = V_max / max(u_at_vmax, 1e-6)   # kN/m
    T_eff      = 2 * np.pi * np.sqrt((W_total / G) / max(k_eff_push, 1e-6))

    print(f"  V_max       = {V_max:.1f} kN  "
          f"(overstrength = {overstrength:.2f}× V_static)")
    print(f"  u_yield     ~ {u_yield*1000:.1f} mm  "
          f"(drift {u_yield/Hn*100:.3f}%)")
    print(f"  u_ultimate  = {u_ultimate*1000:.1f} mm  "
          f"(drift {u_ultimate/Hn*100:.3f}%)")
    print(f"  Ductility μ = {ductility:.2f}  (Δ_ult / Δ_yield)")
    print(f"  T_eff       = {T_eff:.3f} s  (at V_max, secant)")
    print(f"  Fallback steps: {n_fail}/{len(roof_disp)-1}")

    po_results = {
        "roof_disp_m":    roof_disp,
        "base_shear_kN":  base_shear,
        "roof_drift_pct": roof_drift_pct,
        "V_max":          V_max,
        "V_max_idx":      idx_vmax,
        "u_yield_m":      u_yield,
        "u_ultimate_m":   u_ultimate,
        "ductility":      ductility,
        "overstrength":   overstrength,
        "T_eff":          T_eff,
        "k_eff_push":     k_eff_push,
    }

    # ── Rebuild time-history model (wipe clears everything) ────────────────
    # Return node_id_po — caller must call build_model() again before TH
    return po_results


# =============================================================================
# SECTION 7B: PLOT PUSHOVER CAPACITY CURVE
# =============================================================================

def plot_pushover(po_results, T_list):
    """
    Plot pushover capacity curve with key engineering features annotated:
    - Capacity curve (V vs roof drift %)
    - Bilinear idealisation (secant yield stiffness + plateau)
    - AS 1170.4 design base shear (horizontal reference line)
    - Yield point, peak, and ultimate markers
    - Performance level drift limits (IO, LS, CP per ASCE 41)
    - Design spectrum demand curve (Acceleration-Displacement format overlay)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f"{BUILDING_NAME} — Pushover Capacity Curve\n"
        f"AS 1170.4:2007 + Amendment 2 (2018)  |  kpZ = {kpZ:.3f}",
        fontsize=11, fontweight='bold')

    # ── Panel 1: Force-displacement capacity curve ─────────────────────────
    ax = axes[0]
    d_pct = po_results['roof_drift_pct']
    V     = po_results['base_shear_kN']

    ax.plot(d_pct, V, 'steelblue', lw=2, label='Capacity curve')

    # Bilinear idealisation lines
    u_y   = po_results['u_yield_m'] / Hn * 100
    u_ult = po_results['u_ultimate_m'] / Hn * 100
    V_max = po_results['V_max']
    # Elastic branch: origin to yield point
    ax.plot([0, u_y],   [0, V_max], 'k--', lw=1.5, label='Bilinear idealisation')
    # Plateau: yield to ultimate (approximate rigid-plastic post-yield)
    ax.plot([u_y, u_ult], [V_max, V_max * 0.85], 'k--', lw=1.5)

    # Reference lines
    ax.axhline(SA['V_static'], color='red', ls=':', lw=1.5,
               label=f"V_static = {SA['V_static']:.1f} kN (AS 1170.4)")

    # Performance level drift limits
    perf_colours = {'IO (0.5%)': ('#2ecc71', 0.5),
                    'LS (1.5%)': ('#f1c40f', 1.5),
                    'CP (2.5%)': ('#e74c3c', 2.5)}
    for label, (col, lim) in perf_colours.items():
        ax.axvline(lim, color=col, ls='--', lw=1.2, alpha=0.8, label=label)

    # Key markers
    idx = po_results['V_max_idx']
    ax.scatter([d_pct[idx]], [V_max], color='red', s=80, zorder=6,
               label=f'V_max = {V_max:.1f} kN  (drift {d_pct[idx]:.2f}%)')
    ax.scatter([u_y], [V_max], color='orange', s=80, zorder=6,
               label=f'Yield ~ {u_y:.3f}%')

    ax.set_xlabel('Roof Drift (%)')
    ax.set_ylabel('Base Shear (kN)')
    ax.set_title('Capacity Curve (V vs Roof Drift)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # ── Panel 2: ADRS — Acceleration-Displacement Response Spectrum ────────
    # Convert capacity curve to Sa-Sd format (spectral coordinates)
    # Sa = V / (M_total × Gamma₁²) — using mode 1 participation Gamma₁ ≈ 0.9
    # Sd = Δ_roof / (Gamma₁ × phi₁_roof) — phi₁_roof ≈ 1.0 for uniform mass
    ax = axes[1]
    Gamma1   = 0.85   # mode 1 participation factor (approximate)
    M_total  = M_floor * num_storeys
    Sa_cap   = V / max(M_total * G * Gamma1**2, 1e-6)         # g units
    Sd_cap   = po_results['roof_disp_m'] / Gamma1 * 1000      # mm units

    ax.plot(Sd_cap, Sa_cap, 'steelblue', lw=2, label='Capacity spectrum (ADRS)')

    # AS 1170.4 Site De demand spectrum (5% damping)
    T_range  = np.linspace(0.01, 2.5, 200)
    Sa_dem   = np.array([kpZ / mu * Sp * Ch_De(t) for t in T_range])
    Sd_dem   = Sa_dem * G * (T_range / (2 * np.pi))**2 * 1000   # mm
    ax.plot(Sd_dem, Sa_dem, 'red', lw=1.5, ls='--', label='AS 1170.4 demand (5%)')

    # Period rays for T1, T2
    T_colours = ['navy', 'darkorange']
    for i, Ti in enumerate(T_list[:2]):
        Sa_i   = kpZ / mu * Sp * Ch_De(Ti)
        Sd_i   = Sa_i * G * (Ti / (2 * np.pi))**2 * 1000
        ax.plot([0, Sd_i * 1.3], [0, Sa_i * 1.3],
                color=T_colours[i], ls=':', lw=1,
                label=f'T{i+1} = {Ti:.3f} s')
        ax.scatter([Sd_i], [Sa_i], color=T_colours[i], s=50, zorder=6)

    # Performance point (intersection of capacity and demand — approximate)
    # Simple: find where Sa_cap ≈ Sa_dem (same Sd)
    ax.set_xlabel('Spectral Displacement Sd (mm)')
    ax.set_ylabel('Spectral Acceleration Sa (g)')
    ax.set_title('ADRS — Capacity vs. Demand Spectrum')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Summary annotation
    summary = (
        f"Pushover Summary\n"
        f"─────────────────────\n"
        f"V_max    = {V_max:.1f} kN\n"
        f"V_static = {SA['V_static']:.1f} kN\n"
        f"Overstr. = {po_results['overstrength']:.2f}×\n"
        f"Ductility μ = {po_results['ductility']:.2f}\n"
        f"T1_eff = {po_results['T_eff']:.3f} s\n"
        f"T1_FEM = {T_list[0]:.3f} s"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=8, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

    plt.tight_layout()
    fn = f'pushover_{num_storeys}storey_{BUILDING_NAME[:15].replace(" ","_")}.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Pushover plot saved: {fn}")
    return fn


# =============================================================================
# SECTION 8: GROUND MOTION
# =============================================================================

def generate_ground_motion(T1, dt=0.01, duration=20.0):
    """
    Synthetic sine-wave ground motion tuned near T1.
    Saves to tempfile and returns acceleration array for post-processing.
    NOTE: Replace with real spectrum-compatible records for final analysis.
    """
    t     = np.arange(0, duration, dt)
    freq  = min(1.0 / T1, 4.0)
    env   = np.sin(np.pi * t / duration)
    accel = Z * G * env * np.sin(2 * np.pi * freq * t)   # m/s² array

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    np.savetxt(tmp.name, accel, fmt='%.8f')
    tmp.close()

    print(f"  Ground motion: PGA={Z:.3f}g ({Z*G:.3f} m/s²)  "
          f"f_dom={freq:.2f} Hz  duration={duration}s")
    return tmp.name, dt, len(t), accel   # return accel array too

# =============================================================================
# SECTION 8: TIME-HISTORY ANALYSIS
# =============================================================================

def time_history_analysis(node_id, gm_file, dt, npts, omega1, omega2):
    """
    Nonlinear transient analysis.
    Records ALL floor displacements for post-processing.
    Convergence fallback: Newton → KrylovNewton → ModifiedNewton.
    """
    xi = 0.05
    a0 = xi * 2 * omega1 * omega2 / (omega1 + omega2)
    a1 = xi * 2 / (omega1 + omega2)
    ops.rayleigh(a0, 0.0, 0.0, a1)
    print(f"  Rayleigh: a0={a0:.5f}  a1={a1:.7f}  (5% at modes 1 & 2)")

    ops.timeSeries('Path', 2, '-dt', dt, '-filePath', gm_file, '-factor', 1.0)
    ops.pattern('UniformExcitation', 2, 1, '-accel', 2)

    ops.system('UmfPack');  ops.numberer('RCM')
    ops.constraints('Transformation')        # REQUIRED with equalDOF
    ops.test('NormDispIncr', 1.0e-8, 10, 0)
    ops.integrator('Newmark', 0.5, 0.25)     # unconditionally stable
    ops.algorithm('Newton')
    ops.analysis('Transient')

    dt_sub  = dt / 2
    n_steps = int(npts * dt / dt_sub)

    time_h      = np.zeros(n_steps)
    floor_disps = np.zeros((n_steps, num_storeys + 1))
    n_fail = 0

    print(f"  Running: {n_steps} steps × dt={dt_sub:.4f} s")

    for step in range(n_steps):
        ok = ops.analyze(1, dt_sub)
        if ok != 0:
            n_fail += 1
            ops.algorithm('KrylovNewton')
            ok = ops.analyze(1, dt_sub / 5)
            if ok != 0:
                ops.test('NormDispIncr', 1.0e-6, 100, 0)
                ops.algorithm('ModifiedNewton', '-initial')
                ops.analyze(1, dt_sub / 10)
            ops.algorithm('Newton')
            ops.test('NormDispIncr', 1.0e-8, 10, 0)

        time_h[step] = ops.getTime()
        floor_disps[step, 0] = ops.nodeDisp(node_id[0][0], 1)
        for fi in range(1, num_storeys + 1):
            floor_disps[step, fi] = ops.nodeDisp(node_id[fi][0], 1)

    conv_rate = (1 - n_fail / n_steps) * 100
    print(f"  TH complete: {n_fail}/{n_steps} fallback steps  "
          f"(convergence rate {conv_rate:.1f}%)")

    return time_h, floor_disps, conv_rate

# =============================================================================
# SECTION 9: COMPREHENSIVE ENGINEERING POST-PROCESSING
# =============================================================================

def compute_all_edps(time_h, floor_disps, gm_accel, gm_dt, T_list, static_results):
    """
    Compute all engineering demand parameters engineers care about:
    PIDR, storey shear, overturning moment, P-Delta stability coefficient,
    storey stiffness, peak floor acceleration (absolute), column axial ratio,
    performance level, damage state, cumulative ductility demand.
    """
    h  = storey_height
    n  = num_storeys
    T1 = T_list[0]

    # ── 1. INTER-STOREY DRIFT RATIOS ──────────────────────────────────────────
    drift_th = np.zeros((n, len(time_h)))
    PIDR     = np.zeros(n)
    for i in range(1, n + 1):
        drift_th[i-1, :] = (floor_disps[:, i] - floor_disps[:, i-1]) / h
        PIDR[i-1]        = float(np.max(np.abs(drift_th[i-1, :])))

    PIDR_max      = float(np.max(PIDR))
    govern_storey = int(np.argmax(PIDR)) + 1   # 1-based

    # ── 2. STOREY SHEAR FROM TIME HISTORY (inertial force method) ─────────────
    # Absolute acceleration at each floor = ground accel + relative floor accel
    # F_i(t) = M_floor × a_abs_i(t)
    # V_storey_i(t) = Σ_{j=i}^{n} F_j(t)  (sum from storey i to roof)
    dt_arr = np.diff(time_h)
    dt_avg = float(np.mean(dt_arr[dt_arr > 0])) if len(dt_arr) > 0 else gm_dt / 2

    # Interpolate ground motion to match sub-stepped time array
    gm_time = np.arange(len(gm_accel)) * gm_dt
    gm_interp = np.interp(time_h, gm_time, gm_accel)

    abs_accel = np.zeros((len(time_h), n + 1))
    abs_accel[:, 0] = gm_interp   # ground (input)
    for fi in range(1, n + 1):
        rel_accel = np.gradient(np.gradient(floor_disps[:, fi], dt_avg), dt_avg)
        abs_accel[:, fi] = gm_interp + rel_accel   # absolute

    # Storey inertial forces and shear time histories
    F_inertial = M_floor * abs_accel[:, 1:]      # (n_steps, n_storeys)
    V_th = np.zeros((len(time_h), n))
    for i in range(n):
        V_th[:, i] = np.sum(F_inertial[:, i:], axis=1)   # shear at storey i+1

    V_storey_peak = np.max(np.abs(V_th), axis=0)   # peak storey shear per storey

    # ── 3. OVERTURNING MOMENT FROM TIME HISTORY ───────────────────────────────
    # OTM at base of storey i = Σ_{j=i}^{n} F_j(t) × (h_j - h_base_i)
    floor_ht = [(i + 1) * h for i in range(n)]   # heights of floor slabs
    OTM_th = np.zeros((len(time_h), n))
    for i in range(n):
        h_base = i * h
        OTM_th[:, i] = sum(
            F_inertial[:, j] * (floor_ht[j] - h_base)
            for j in range(i, n)
        )

    OTM_peak = np.max(np.abs(OTM_th), axis=0)   # peak OTM per level
    OTM_base = OTM_peak[0]

    # ── 4. P-DELTA STABILITY COEFFICIENT (AS1170.4 Cl 6.5) ───────────────────
    # theta_i = (P_i × delta_i) / (V_i × h_i)
    # P_i = total gravity above base of storey i = (n - i + 1) × W_floor
    # delta_i = peak inter-storey displacement = PIDR_i × h
    # V_i = peak storey shear (dynamic)
    # Limits: theta > 0.10 → P-Delta must be considered
    #         theta > 0.25 → structure potentially unstable
    theta = np.zeros(n)
    for i in range(n):
        P_above  = (n - i) * W_floor      # gravity load above storey i
        delta_i  = PIDR[i] * h            # peak inter-storey displacement
        V_i      = max(V_storey_peak[i], 1.0)   # avoid div/0
        theta[i] = (P_above * delta_i) / (V_i * h)

    theta_max     = float(np.max(theta))
    pdelta_critical = any(th > 0.10 for th in theta)
    pdelta_unstable = any(th > 0.25 for th in theta)

    # ── 5. STOREY STIFFNESS + SOFT STOREY DETECTION ───────────────────────────
    # k_i = V_storey_i / delta_i  (secant stiffness at peak response)
    k_eff = np.zeros(n)
    for i in range(n):
        delta_i = PIDR[i] * h
        k_eff[i] = V_storey_peak[i] / max(delta_i, 1e-6)   # kN/m

    # Soft storey: k_i < 0.70 × k_{i+1}
    soft_storeys_dyn = []
    for i in range(n - 1):
        if k_eff[i] < 0.70 * k_eff[i + 1]:
            soft_storeys_dyn.append(i + 1)

    # ── 6. PEAK FLOOR ACCELERATIONS (absolute) ────────────────────────────────
    PFA = np.max(np.abs(abs_accel), axis=0)   # includes ground (index 0)
    amp_factors = PFA / max(PFA[0], 1e-6)

    # ── 7. DYNAMIC BASE SHEAR ─────────────────────────────────────────────────
    V_dynamic = float(V_storey_peak[0])   # base shear = storey 1 shear

    # ── 8. COLUMN AXIAL LOAD RATIO (N/Nuo) ───────────────────────────────────
    # Gravity axial load on exterior ground-storey column:
    #   N_grav = num_storeys × P_ext  (each floor contributes P_ext)
    # Seismic axial load (overturning, simplified):
    #   N_seis = OTM_base / (frame_width)  [leeward column in tension]
    # Nuo = 0.85 × f'c × Ag  (AS3600 simplified, no reduction for slenderness)
    col_b_gnd, col_h_gnd = get_col_size(1)
    Ag_gnd  = col_b_gnd * col_h_gnd   # m²
    Nuo     = 0.85 * fc_kN * Ag_gnd   # kN  (Nuo per AS3600 simplified)
    N_grav  = num_storeys * P_ext      # kN  gravity on exterior ground col
    N_seis  = OTM_base / (num_bays * bay_width)   # kN seismic axial demand
    N_total = N_grav + N_seis
    N_ratio = N_total / max(Nuo, 1.0)

    # AS3600 Cl 10.6.3: for mu_c ductility, N/Nuo <= 0.30 recommended
    col_axial_flag = N_ratio > 0.30

    # ── 9. PERFORMANCE LEVEL ──────────────────────────────────────────────────
    if PIDR_max <= PERFORMANCE_LIMITS["Immediate Occupancy (IO)"]:
        perf_level = "Immediate Occupancy (IO)"
    elif PIDR_max <= PERFORMANCE_LIMITS["Life Safety (LS)"]:
        perf_level = "Life Safety (LS)"
    elif PIDR_max <= PERFORMANCE_LIMITS["Collapse Prevention (CP)"]:
        perf_level = "Collapse Prevention (CP)"
    else:
        perf_level = "Beyond Collapse Prevention"

    # ── 10. HAZUS DAMAGE STATE ────────────────────────────────────────────────
    damage_state = "Complete"
    repair_ratio = 1.0
    for name, lo, hi, rr in DAMAGE_STATES:
        if lo <= PIDR_max < hi:
            damage_state = name
            repair_ratio = rr
            break

    # ── 11. CUMULATIVE DUCTILITY DEMAND (governing storey) ────────────────────
    # Estimate yield displacement from bilinear approximation:
    # delta_y ≈ fy / (E * h) × h = fy / Es_steel  [strain at yield × storey height]
    # More practically: delta_y ≈ 0.6% PIDR × h (yield drift for pre-1990)
    delta_y_est = 0.006 * h   # m — approximate yield inter-storey disp for pre-1990
    gov_drift_th = drift_th[govern_storey - 1, :] * h   # displacement time history
    # Cumulative ductility = sum of all displacement excursions / delta_y
    sign_changes = np.where(np.diff(np.sign(gov_drift_th)))[0]
    excursion_sum = 0.0
    for k_idx in range(len(sign_changes) - 1):
        i1, i2 = sign_changes[k_idx], sign_changes[k_idx + 1]
        excursion = float(np.max(np.abs(gov_drift_th[i1:i2+1])))
        if excursion > delta_y_est:
            excursion_sum += (excursion - delta_y_est)
    cumulative_ductility = excursion_sum / max(delta_y_est, 1e-6)

    # ── 12. MAX ROOF DISPLACEMENT ─────────────────────────────────────────────
    max_roof_mm = float(np.max(np.abs(floor_disps[:, n]))) * 1000

    # ── Print comprehensive report ────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  COMPREHENSIVE ENGINEERING DEMAND PARAMETERS — {n} STOREYS")
    print(f"  AS 1170.4:2007 + Amendment 2 (2018)")
    print("=" * 70)

    print(f"\n  [1] STRUCTURAL PERIODS")
    for i, Ti in enumerate(T_list, 1):
        print(f"      T{i} = {Ti:.4f} s")
    print(f"      T1 code approx = {T1_approx:.4f} s  "
          f"(FEM/code ratio = {T_list[0]/T1_approx:.3f}x)")

    print(f"\n  [2] INTER-STOREY DRIFT RATIOS  (limit {drift_limit*100:.1f}%)")
    for i in range(n):
        status = "FAIL" if PIDR[i] > drift_limit else "pass"
        gov    = " ← GOVERNING" if (i+1) == govern_storey else ""
        print(f"      Storey {i+1:2d}: PIDR = {PIDR[i]*100:.4f}%  [{status}]{gov}")
    print(f"      Max PIDR = {PIDR_max*100:.4f}%  |  Governing: Storey {govern_storey}")

    print(f"\n  [3] STOREY SHEAR  (kN)")
    for i in range(n):
        stat_V = SA['V_storey'][i]
        print(f"      Storey {i+1:2d}: V_dyn = {V_storey_peak[i]:7.1f}  "
              f"V_stat = {stat_V:7.1f}  "
              f"ratio = {V_storey_peak[i]/max(stat_V,0.01):.2f}x")

    print(f"\n  [4] OVERTURNING MOMENT  (kN.m)")
    print(f"      Base OTM (dynamic) = {OTM_base:.1f} kN.m")
    print(f"      Base OTM (static)  = {SA['OTM'][0]:.1f} kN.m")
    for i in range(n):
        print(f"      Level {i+1:2d}: {OTM_peak[i]:.1f} kN.m  "
              f"(stat: {SA['OTM'][i]:.1f})")

    print(f"\n  [5] P-DELTA STABILITY (AS1170.4 Cl 6.5)  θ = P.δ / (V.h)")
    for i in range(n):
        flag = ""
        if theta[i] > 0.25: flag = "  !! POTENTIALLY UNSTABLE (θ > 0.25)"
        elif theta[i] > 0.10: flag = "  ! Must include P-Delta (θ > 0.10)"
        print(f"      Storey {i+1:2d}: θ = {theta[i]:.4f}{flag}")
    if not pdelta_critical:
        print(f"      P-Delta: OK  (all θ < 0.10 — P-Delta effects negligible)")

    print(f"\n  [6] STOREY STIFFNESS  (kN/m)")
    for i in range(n):
        ss_flag = "  ← SOFT STOREY" if (i+1) in soft_storeys_dyn else ""
        print(f"      Storey {i+1:2d}: k_eff = {k_eff[i]:.1f} kN/m{ss_flag}")
    if soft_storeys_dyn:
        print(f"      !! Dynamic soft storeys detected: {soft_storeys_dyn}")
    else:
        print(f"      Stiffness regularity: OK (no dynamic soft storey)")

    print(f"\n  [7] PEAK FLOOR ACCELERATIONS  (absolute)")
    for fi in range(n + 1):
        label = "Ground" if fi == 0 else (f"Roof  " if fi == n else f"Floor {fi}")
        print(f"      {label}: {PFA[fi]:.4f} m/s2  "
              f"({PFA[fi]/G:.4f}g)  amp = {amp_factors[fi]:.2f}x")

    print(f"\n  [8] COLUMN AXIAL LOAD RATIO (ground storey exterior col)")
    print(f"      N_grav = {N_grav:.1f} kN  N_seis = {N_seis:.1f} kN  "
          f"N_total = {N_total:.1f} kN")
    print(f"      Nuo = {Nuo:.1f} kN  (0.85 f'c Ag)")
    print(f"      N/Nuo = {N_ratio:.3f}  "
          f"{'!! EXCEEDS 0.30 — reduced ductility (AS3600)' if col_axial_flag else 'OK (< 0.30)'}")

    print(f"\n  [9] PERFORMANCE LEVEL  (ASCE 41 / ATC-40)")
    for level, pidr_lim in PERFORMANCE_LIMITS.items():
        achieved = "✓" if PIDR_max <= pidr_lim else "✗"
        print(f"      {achieved} {level}: limit {pidr_lim*100:.1f}%")
    print(f"      Performance Level Achieved: {perf_level}")

    print(f"\n  [10] DAMAGE ASSESSMENT")
    print(f"      HAZUS damage state: {damage_state}  "
          f"(repair cost ratio ~ {repair_ratio*100:.0f}%)")
    print(f"      Cumulative ductility demand (Storey {govern_storey}): "
          f"{cumulative_ductility:.2f}")

    print(f"\n  [11] BASE SHEAR COMPARISON")
    print(f"      V_static  (AS1170.4 + Amd 2) = {SA['V_static']:.1f} kN  "
          f"(V/W = {SA['V_static']/W_total:.4f})")
    print(f"      V_dynamic (inertial forces)   = {V_dynamic:.1f} kN  "
          f"(ratio = {V_dynamic/max(SA['V_static'],0.01):.2f}x)")

    comply = PIDR_max <= drift_limit
    print()
    print("─" * 70)
    print(f"  AS1170.4:2007 + Amendment 2: "
          f"{'COMPLIANT' if comply else 'NON-COMPLIANT'}")
    print(f"  kpZ = {kpZ:.3f} (Amd 2 min 0.08 applied)  |  "
          f"Max PIDR = {PIDR_max*100:.4f}%  |  Limit = 1.500%")
    print("─" * 70)

    return {
        # Time histories
        "drift_th":         drift_th,
        "floor_disps":      floor_disps,
        "V_th":             V_th,
        "OTM_th":           OTM_th,
        "abs_accel":        abs_accel,
        # Scalars and per-storey arrays
        "PIDR":             PIDR,
        "PIDR_max":         PIDR_max,
        "govern_storey":    govern_storey,
        "V_storey_peak":    V_storey_peak,
        "V_dynamic":        V_dynamic,
        "OTM_peak":         OTM_peak,
        "OTM_base":         OTM_base,
        "theta":            theta,
        "theta_max":        theta_max,
        "pdelta_critical":  pdelta_critical,
        "pdelta_unstable":  pdelta_unstable,
        "k_eff":            k_eff,
        "soft_storeys_dyn": soft_storeys_dyn,
        "PFA":              PFA,
        "amp_factors":      amp_factors,
        "N_grav":           N_grav,
        "N_seis":           N_seis,
        "N_total":          N_total,
        "Nuo":              Nuo,
        "N_ratio":          N_ratio,
        "col_axial_flag":   col_axial_flag,
        "perf_level":       perf_level,
        "damage_state":     damage_state,
        "repair_ratio":     repair_ratio,
        "cumulative_ductility": cumulative_ductility,
        "max_roof_mm":      max_roof_mm,
        "drift_pass":       PIDR_max <= drift_limit,
        "compliant":        PIDR_max <= drift_limit,
        "T_list":           T_list,
        "T1_approx":        T1_approx,
    }

# =============================================================================
# SECTION 10: ENGINEERING DASHBOARD — 12-PANEL FIGURE
# =============================================================================

STATE_COL = {
    "None":"#2ecc71","Slight":"#f1c40f",
    "Moderate":"#e67e22","Extensive":"#e74c3c","Complete":"#8e44ad"
}

def plot_engineering_dashboard(time_h, edp, gm_accel, gm_dt, T_list, static_results):
    """
    12-panel engineering dashboard. All panels scale to num_storeys.
    Panels:
      Row 1: Roof disp TH | All-storey drift TH | Design spectrum
      Row 2: PIDR profile  | Storey shear profile | OTM profile
      Row 3: P-Delta θ     | Storey stiffness     | PFA profile
      Row 4: Hysteresis    | Performance level     | Summary table
    """
    n   = num_storeys
    lim = drift_limit * 100

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(
        f"{BUILDING_NAME} — {n}-Storey Seismic Assessment\n"
        f"AS 1170.4:2007 + Amendment 2 (2018)  |  kpZ = {kpZ:.3f}  "
        f"|  Z_raw = {Z_raw}  |  Site {site_class}",
        fontsize=12, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)

    cmap_n = plt.cm.get_cmap('plasma', max(n, 2))

    # ─── Panel 1: Roof displacement time history ─────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(time_h, edp['floor_disps'][:, n]*1000, color='steelblue', lw=1)
    ax.fill_between(time_h, edp['floor_disps'][:, n]*1000, 0,
                    alpha=0.12, color='steelblue')
    ax.axhline(0, color='k', lw=0.4)
    ax.set(xlabel='Time (s)', ylabel='Displacement (mm)',
           title=f'Roof Displacement  (max {edp["max_roof_mm"]:.1f} mm)')
    ax.grid(True, alpha=0.3)

    # ─── Panel 2: All-storey drift time histories ────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for i in range(n):
        col = cmap_n(i / max(n-1, 1))
        ax.plot(time_h, edp['drift_th'][i, :]*100,
                color=col, lw=0.8, alpha=0.9, label=f'S{i+1}')
    ax.axhline( lim, color='red', ls='--', lw=1.5)
    ax.axhline(-lim, color='red', ls='--', lw=1.5, label=f'{lim:.0f}% limit')
    ax.set(xlabel='Time (s)', ylabel='Drift (%)',
           title='Inter-Storey Drift — All Storeys')
    ax.legend(fontsize=7, ncol=min(n, 4))
    ax.grid(True, alpha=0.3)

    # ─── Panel 3: Design spectrum + building periods ─────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    T_range = np.linspace(0.01, 3.0, 500)
    Sa_De   = [kpZ / mu * Sp * Ch_De(t) for t in T_range]
    ax.plot(T_range, Sa_De, color='navy', lw=2, label=f'Site De (kpZ={kpZ:.3f})')
    colors_modes = ['red','orange','green','purple']
    for i, Ti in enumerate(T_list):
        Sa_i = kpZ / mu * Sp * Ch_De(Ti)
        ax.axvline(Ti, color=colors_modes[i], ls='--', lw=1.5,
                   label=f'T{i+1}={Ti:.3f}s  Sa={Sa_i:.4f}g')
        ax.scatter([Ti], [Sa_i], color=colors_modes[i], s=60, zorder=5)
    ax.axvline(T1_approx, color='gray', ls=':', lw=1,
               label=f'T1_code={T1_approx:.3f}s')
    ax.set(xlabel='Period (s)', ylabel='Sa (g)',
           title='AS1170.4 + Amd 2 Design Spectrum')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ─── Panel 4: PIDR profile (horizontal bars) ─────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    storeys  = list(range(1, n+1))
    pidr_pct = edp['PIDR'] * 100
    bar_cols = ['#d9534f' if v > drift_limit else '#5cb85c' for v in edp['PIDR']]
    bars = ax.barh(storeys, pidr_pct, color=bar_cols, edgecolor='white', height=0.6)
    ax.axvline(lim, color='red', ls='--', lw=1.5, label=f'{lim:.0f}% limit')
    ax.set(xlabel='PIDR (%)', ylabel='Storey',
           title='Peak Inter-Storey Drift Profile')
    ax.set_yticks(storeys)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, pidr_pct):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}%', va='center', fontsize=8)

    # ─── Panel 5: Storey shear profile (static vs dynamic) ───────────────────
    ax = fig.add_subplot(gs[1, 1])
    ys = np.arange(1, n+1)
    ax.barh(ys - 0.2, edp['V_storey_peak'], height=0.35,
            color='steelblue', alpha=0.9, label='Dynamic (peak)')
    ax.barh(ys + 0.2, static_results['V_storey'], height=0.35,
            color='lightgray', edgecolor='gray', label='Static (AS1170.4)')
    ax.set(xlabel='Storey Shear (kN)', ylabel='Storey',
           title='Storey Shear Profile')
    ax.set_yticks(ys)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

    # ─── Panel 6: Overturning moment profile ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    level_ht = [i * storey_height for i in range(n+1)]
    ax.step([0] + list(edp['OTM_peak']),
            level_ht[::-1],
            where='post', color='steelblue', lw=2, label='Dynamic OTM')
    ax.step([0] + list(static_results['OTM'][:n]),
            level_ht[::-1],
            where='post', color='gray', ls='--', lw=1.5, label='Static OTM')
    ax.set(xlabel='Overturning Moment (kN.m)', ylabel='Height (m)',
           title='Overturning Moment Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, Hn + 0.5)

    # ─── Panel 7: P-Delta stability coefficient ───────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    theta_pct = edp['theta']
    colours_th = ['#d9534f' if th > 0.10 else '#f0ad4e' if th > 0.05
                  else '#5cb85c' for th in theta_pct]
    ax.barh(storeys, theta_pct, color=colours_th, edgecolor='white', height=0.6)
    ax.axvline(0.10, color='orange', ls='--', lw=1.5, label='0.10 — P-Δ required')
    ax.axvline(0.25, color='red',    ls='--', lw=1.5, label='0.25 — instability risk')
    ax.set(xlabel='Stability Coefficient θ', ylabel='Storey',
           title='P-Delta Stability  (AS1170.4 Cl 6.5)')
    ax.set_yticks(storeys)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='x')
    for i, th in enumerate(theta_pct):
        ax.text(th + 0.001, i+1, f'{th:.4f}', va='center', fontsize=8)

    # ─── Panel 8: Storey stiffness ────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    keff_cols = ['#d9534f' if (i+1) in edp['soft_storeys_dyn']
                 else '#5cb85c' for i in range(n)]
    ax.barh(storeys, edp['k_eff'], color=keff_cols, edgecolor='white', height=0.6)
    ax.set(xlabel='Effective Stiffness (kN/m)', ylabel='Storey',
           title='Storey Lateral Stiffness\n(red = soft storey)')
    ax.set_yticks(storeys)
    ax.grid(True, alpha=0.3, axis='x')
    # Add 0.70× reference line for storey above each
    for i in range(n-1):
        ref = 0.70 * edp['k_eff'][i+1]
        ax.plot([ref, ref], [i+0.7, i+1.3], color='red', ls=':', lw=1)

    # ─── Panel 9: Peak floor acceleration profile ─────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    floor_labels = ['Ground'] + [f'F{i}' if i < n else 'Roof'
                                  for i in range(1, n+1)]
    pfa_g = edp['PFA'] / G
    ax.barh(range(n+1), pfa_g, color='steelblue', alpha=0.85)
    ax.set(xlabel='PFA (g)', ylabel='Floor level',
           title='Peak Floor Acceleration (absolute)')
    ax.set_yticks(range(n+1))
    ax.set_yticklabels(floor_labels, fontsize=8)
    ax.axvline(Z, color='red', ls='--', lw=1.2, label=f'PGA = {Z:.3f}g')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='x')
    for fi, val in enumerate(pfa_g):
        ax.text(val+0.002, fi, f'{val:.3f}g', va='center', fontsize=8)

    # ─── Panel 10: Hysteresis — governing storey ─────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    gs_idx = edp['govern_storey'] - 1
    drift_gov = edp['drift_th'][gs_idx, :] * 100
    disp_abv   = edp['floor_disps'][:, edp['govern_storey']] * 1000
    ax.plot(drift_gov, disp_abv, color='darkorchid', lw=0.7, alpha=0.9)
    ax.axvline(0, color='k', lw=0.4); ax.axhline(0, color='k', lw=0.4)
    ax.axvline( lim, color='red', ls=':', lw=0.8)
    ax.axvline(-lim, color='red', ls=':', lw=0.8)
    ax.set(xlabel=f'Storey {edp["govern_storey"]} Drift (%)',
           ylabel='Floor Displacement (mm)',
           title=f'Hysteresis — Governing Storey {edp["govern_storey"]}')
    ax.grid(True, alpha=0.3)

    # ─── Panel 11: Performance level indicator ────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    ax.axis('off')
    # Colour-coded performance level bar
    perf_levels_list = list(PERFORMANCE_LIMITS.items())
    bar_y = 0.60
    level_colours = ['#2ecc71', '#f0ad4e', '#e74c3c', '#8e44ad']
    xlims = [0, 0.005, 0.015, 0.025, 0.040]
    achieved_idx = 3   # default: beyond CP
    for idx, (name, lim_v) in enumerate(perf_levels_list):
        if edp['PIDR_max'] <= lim_v:
            achieved_idx = idx
            break

    for idx in range(3):
        name, lim_v = perf_levels_list[idx]
        col = level_colours[idx]
        w   = (xlims[idx+1] - xlims[idx]) / 0.040
        x0  = xlims[idx] / 0.040
        rect = plt.Rectangle((x0, bar_y), w, 0.08, color=col, alpha=0.7,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x0 + w/2, bar_y + 0.04, name.split('(')[1].strip(')'),
                ha='center', va='center', fontsize=8, transform=ax.transAxes)

    # Mark actual PIDR
    marker_x = min(edp['PIDR_max'] / 0.040, 1.0)
    ax.annotate('', xy=(marker_x, bar_y), xytext=(marker_x, bar_y + 0.18),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(marker_x, bar_y + 0.22,
            f"PIDR = {edp['PIDR_max']*100:.3f}%",
            ha='center', fontsize=9, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.30,
            f"Performance Level: {edp['perf_level']}",
            ha='center', fontsize=10, fontweight='bold',
            transform=ax.transAxes,
            color=level_colours[min(achieved_idx, 2)])
    ax.text(0.5, 0.15, f"HAZUS: {edp['damage_state']}  "
            f"(~{edp['repair_ratio']*100:.0f}% repair cost)",
            ha='center', fontsize=9, transform=ax.transAxes)
    ax.set_title('Performance Level Assessment')

    # ─── Panel 12: Engineering summary table ─────────────────────────────────
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    comply   = 'COMPLIANT' if edp['compliant'] else 'NON-COMPLIANT'
    pdelta_s = ("!! CHECK REQUIRED" if edp['pdelta_critical']
                else "OK (θ_max < 0.10)")
    axial_s  = (f"!! HIGH: {edp['N_ratio']:.2f}" if edp['col_axial_flag']
                else f"OK: {edp['N_ratio']:.2f}")

    pidr_rows = "\n".join(
        f"    S{i+1:2d}: {edp['PIDR'][i]*100:.4f}%"
        + (" [FAIL]" if edp['PIDR'][i] > drift_limit else " [pass]")
        + (" ← GOVERN" if (i+1) == edp['govern_storey'] else "")
        for i in range(n))

    lines = [
        "=" * 46,
        f"  {BUILDING_NAME[:30]}",
        f"  {n} storeys | kpZ = {kpZ:.3f} (Amd 2) | Z_raw = {Z_raw}",
        "=" * 46,
        f"  Periods:  T1={T_list[0]:.4f}s  T2={T_list[1]:.4f}s",
        f"  T1_code = {T1_approx:.4f}s  (ratio {T_list[0]/T1_approx:.2f}x)",
        "─" * 46,
        "  PIDR per storey:",
        pidr_rows,
        f"  Max PIDR: {edp['PIDR_max']*100:.4f}%  (limit 1.5%)",
        "─" * 46,
        f"  V_static  = {SA['V_static']:.1f} kN  (V/W={SA['V_static']/W_total:.4f})",
        f"  V_dynamic = {edp['V_dynamic']:.1f} kN",
        f"  OTM_base  = {edp['OTM_base']:.1f} kN.m",
        f"  θ_max     = {edp['theta_max']:.4f}  P-Δ: {pdelta_s}",
        f"  N/Nuo     = {axial_s}",
        f"  Performance: {edp['perf_level']}",
        f"  Damage:      {edp['damage_state']}",
        "=" * 46,
        f"  AS1170.4 + Amd2: {comply}",
    ]
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
            fontsize=7.5, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))
    ax.set_title('Engineering Summary')

    fn = (f'engineering_assessment_{n}storey_'
          f'{BUILDING_NAME[:15].replace(" ","_")}.png')
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"\n  Engineering dashboard saved: {fn}")
    return fn

# =============================================================================
# SECTION 11: JSON REPORT
# =============================================================================

def save_json_report(edp, static_results, irreg, T_list, conv_rate):
    """
    Comprehensive JSON report — all engineering quantities stored.
    Per-storey arrays saved as lists for downstream processing.
    """
    def rnd(v, d=4):
        return round(float(v), d)

    report = {
        "generated":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "project":       "UTS Engineering Graduate Project PG (42003)",
        "student":       "Kabish Jung Thapa (25631413)",
        "standard":      "AS 1170.4:2007 incorporating Amendment No. 2 (Feb 2018)",
        "amendment_2_applied": True,
        "kpZ_used":      kpZ,
        "kpZ_minimum_Amd2": 0.08,

        "building": {
            "name":          BUILDING_NAME,
            "num_storeys":   num_storeys,
            "storey_height_m": storey_height,
            "num_bays":      num_bays,
            "bay_width_m":   bay_width,
            "floor_width_m": floor_width,
            "Hn_m":          Hn,
        },
        "materials": {
            "fc_MPa": fc, "fy_MPa": fy, "Es_MPa": Es,
            "epsc0_core": epsc0_core, "epsU_core": epsU_core,
        },
        "column_sizes_per_storey": {
            str(fi): list(get_col_size(fi))
            for fi in range(1, num_storeys + 1)
        },
        "seismic_parameters": {
            "Z_raw":       Z_raw,
            "kp":          kp,
            "kpZ":         kpZ,
            "site_class":  site_class,
            "mu":          mu,
            "Sp":          Sp,
            "drift_limit_%": drift_limit * 100,
            "Ch_T1":       rnd(static_results['Ch']),
            "k_exponent":  rnd(static_results['k']),
        },
        "loads": {
            "dead_load_kPa":   dead_load,
            "live_load_kPa":   live_load,
            "W_floor_kN":      rnd(W_floor, 2),
            "W_total_kN":      rnd(W_total, 2),
            "M_floor_kNs2m":   rnd(M_floor, 4),
        },
        "static_analysis": {
            "V_static_kN":        rnd(static_results['V_static'], 2),
            "V_over_W":           rnd(static_results['V_static']/W_total, 5),
            "F_lat_per_floor_kN": [rnd(f, 3) for f in static_results['F_lat']],
            "V_storey_kN":        [rnd(v, 2) for v in static_results['V_storey']],
            "OTM_per_level_kNm":  [rnd(m, 2) for m in static_results['OTM']],
            "N_uplift_ext_col_kN":rnd(static_results['N_uplift'], 2),
        },
        "structural_irregularity": {
            "is_irregular":   irreg['is_irregular'],
            "soft_storeys_gross": irreg['soft_storeys'],
            "k_gross_kNm":    [rnd(k, 1) for k in irreg['k_gross']],
            "flags":          irreg['flags'],
        },
        "periods": {
            "T_list_s":   [rnd(t, 5) for t in T_list],
            "T1_FEM_s":   rnd(T_list[0], 5),
            "T1_code_s":  rnd(T1_approx, 5),
            "T1_ratio":   rnd(T_list[0]/T1_approx, 4),
        },
        "time_history_convergence": {
            "convergence_rate_%": rnd(conv_rate, 2),
        },
        "inter_storey_drift": {
            "PIDR_per_storey_%":  [rnd(v*100, 5) for v in edp['PIDR']],
            "PIDR_max_%":         rnd(edp['PIDR_max']*100, 5),
            "governing_storey":   edp['govern_storey'],
            "drift_limit_%":      drift_limit * 100,
            "drift_pass":         edp['drift_pass'],
        },
        "storey_shear": {
            "V_storey_dynamic_kN": [rnd(v, 2) for v in edp['V_storey_peak']],
            "V_base_dynamic_kN":   rnd(edp['V_dynamic'], 2),
        },
        "overturning_moment": {
            "OTM_dynamic_kNm": [rnd(m, 2) for m in edp['OTM_peak']],
            "OTM_base_kNm":    rnd(edp['OTM_base'], 2),
        },
        "pdelta_stability": {
            "theta_per_storey":     [rnd(th, 5) for th in edp['theta']],
            "theta_max":            rnd(edp['theta_max'], 5),
            "pdelta_required":      edp['pdelta_critical'],
            "pdelta_instability":   edp['pdelta_unstable'],
            "note": "AS1170.4 Cl 6.5: theta > 0.10 requires P-Delta; > 0.25 = instability risk",
        },
        "storey_stiffness": {
            "k_eff_kNm":         [rnd(k, 1) for k in edp['k_eff']],
            "soft_storeys_dyn":  edp['soft_storeys_dyn'],
        },
        "floor_accelerations": {
            "PFA_ms2":     [rnd(v, 5) for v in edp['PFA']],
            "PFA_g":       [rnd(v/G, 5) for v in edp['PFA']],
            "amp_factors": [rnd(v, 4) for v in edp['amp_factors']],
        },
        "column_axial_check": {
            "N_grav_kN":  rnd(edp['N_grav'], 2),
            "N_seis_kN":  rnd(edp['N_seis'], 2),
            "N_total_kN": rnd(edp['N_total'], 2),
            "Nuo_kN":     rnd(edp['Nuo'], 2),
            "N_over_Nuo": rnd(edp['N_ratio'], 4),
            "flag_exceeds_0p30": edp['col_axial_flag'],
            "note": "AS3600 Cl 10.6.3: N/Nuo <= 0.30 for full ductility",
        },
        "performance_assessment": {
            "performance_level":    edp['perf_level'],
            "hazus_damage_state":   edp['damage_state'],
            "repair_cost_ratio_%":  edp['repair_ratio'] * 100,
            "cumulative_ductility": rnd(edp['cumulative_ductility'], 3),
            "as1170_4_result":      "COMPLIANT" if edp['compliant'] else "NON-COMPLIANT",
        },
        "pushover_analysis": {
            "V_max_kN":           rnd(edp['pushover']['V_max'], 2),
            "V_static_kN":        rnd(SA['V_static'], 2),
            "overstrength_ratio": rnd(edp['pushover']['overstrength'], 3),
            "u_yield_mm":         rnd(edp['pushover']['u_yield_m'] * 1000, 2),
            "u_yield_drift_%":    rnd(edp['pushover']['u_yield_m'] / Hn * 100, 4),
            "u_ultimate_mm":      rnd(edp['pushover']['u_ultimate_m'] * 1000, 2),
            "ductility_mu":       rnd(edp['pushover']['ductility'], 3),
            "T_eff_s":            rnd(edp['pushover']['T_eff'], 4),
            "k_eff_kNm":          rnd(edp['pushover']['k_eff_push'], 1),
            "note": ("Displacement-controlled pushover. Inverted-triangle load pattern. "
                     "Bilinear yield at 60% V_max secant. ATC-40 / ASCE 41-17."),
        },
    }

    fn = f"report_{num_storeys}storey_comprehensive.json"
    with open(fn, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report saved: {fn}")
    return fn

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__" or True:

    print("\n" + "=" * 68)
    print(f"  STEP 1: Build OpenSeesPy model  ({num_storeys} storeys, {num_bays} bays)")
    print("=" * 68)
    node_id = build_model()

    print("\n" + "=" * 68)
    print("  STEP 2: Gravity analysis")
    print("=" * 68)
    gravity_analysis(node_id)

    print("\n" + "=" * 68)
    print("  STEP 3: Assign masses")
    print("=" * 68)
    assign_masses(node_id)

    print("\n" + "=" * 68)
    print(f"  STEP 4: Eigenvalue analysis ({N_MODES} modes)")
    print("=" * 68)
    T_list, eigs, omega1, omega2, participation = eigenvalue_analysis()

    print("\n" + "=" * 68)
    print("  STEP 5: Pushover analysis (capacity curve + ADRS)")
    print("=" * 68)
    po_results = pushover_analysis(node_id, T_list)
    plot_pushover(po_results, T_list)

    # Pushover wipes the model — rebuild for time-history
    print("\n" + "=" * 68)
    print(f"  STEP 5b: Rebuild model after pushover wipe")
    print("=" * 68)
    node_id = build_model()
    gravity_analysis(node_id)
    assign_masses(node_id)
    T_list2, eigs2, omega1, omega2, _ = eigenvalue_analysis()
    # Use original T_list from Step 4 for consistency
    print("  Model rebuilt — ready for time-history")

    print("\n" + "=" * 68)
    print("  STEP 6: Ground motion")
    print("=" * 68)
    gm_file, gm_dt, gm_npts, gm_accel = generate_ground_motion(T_list[0])

    print("\n" + "=" * 68)
    print("  STEP 7: Nonlinear time-history analysis")
    print("=" * 68)
    time_h, floor_disps, conv_rate = time_history_analysis(
        node_id, gm_file, gm_dt, gm_npts, omega1, omega2)

    print("\n" + "=" * 68)
    print("  STEP 8: Comprehensive engineering post-processing")
    print("=" * 68)
    edp = compute_all_edps(
        time_h, floor_disps, gm_accel, gm_dt, T_list, SA)
    edp['pushover'] = po_results   # attach pushover to EDP dict

    print("\n" + "=" * 68)
    print("  STEP 9: Engineering dashboard (12 panels)")
    print("=" * 68)
    plot_engineering_dashboard(time_h, edp, gm_accel, gm_dt, T_list, SA)

    print("\n" + "=" * 68)
    print("  STEP 10: Save comprehensive JSON report")
    print("=" * 68)
    save_json_report(edp, SA, IRREG, T_list, conv_rate)

    try:
        os.remove(gm_file)
    except Exception:
        pass

    print()
    print("=" * 68)
    print("  COMPLETE")
    print("=" * 68)
    print(f"  Building     : {BUILDING_NAME}  ({num_storeys} storeys)")
    print(f"  kpZ          : {kpZ:.3f}  (Amendment 2 minimum 0.08 applied)")
    print(f"  T1 (FEM)     : {T_list[0]:.4f} s")
    print(f"  Max PIDR     : {edp['PIDR_max']*100:.4f}%  "
          f"(Storey {edp['govern_storey']})")
    print(f"  Performance  : {edp['perf_level']}")
    print(f"  Damage       : {edp['damage_state']}")
    print(f"  P-Delta θmax : {edp['theta_max']:.4f}  "
          f"({'CRITICAL' if edp['pdelta_critical'] else 'OK'})")
    print(f"  N/Nuo        : {edp['N_ratio']:.3f}  "
          f"({'HIGH' if edp['col_axial_flag'] else 'OK'})")
    print(f"  AS1170.4     : {'COMPLIANT' if edp['compliant'] else 'NON-COMPLIANT'}")
    print()
    print("  Change  num_storeys  in Section 1 to analyse any building height.")
    print("  Era quick-reference:")
    print("    Pre-1990  : fc=20, fy=250, mu=2.0, Sp=0.77 (this file)")
    print("    Post-1990 : fc=32, fy=500, mu=3.0, Sp=0.67")
    print("    Post-2010 : fc=40, fy=500, mu=4.0, Sp=0.67")

# =============================================================================
# SECTION 12: PUSHOVER ANALYSIS (Displacement-Controlled, ATC-40 / ASCE 41-17)
# =============================================================================
#
# Displacement-controlled pushover under inverted-triangle lateral load pattern.
# Control node: roof master node. Target: 3% roof drift = 0.03 × Hn.
#
# Outputs:
#   - Capacity curve: base shear vs roof drift (%)
#   - V_max (peak base shear), Δ_yield, ductility ratio μ_actual
#   - Performance point (intersection with demand spectrum) — qualitative
#   - Overstrength factor Ω = V_max / V_static
#   - Effective initial stiffness k_eff = V_max / Δ_yield
#
# Integration with main pipeline:
#   Call run_pushover(node_id, SA) BEFORE time-history analysis (wipe is called
#   at the start so model must be rebuilt afterwards if TH analysis follows).
#   The function returns a results dict; the main block rebuilds the model.
# =============================================================================

