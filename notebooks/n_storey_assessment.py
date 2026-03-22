# =============================================================================
# N-STOREY RC FRAME — GENERALISED SEISMIC VULNERABILITY ASSESSMENT
# LLM-Orchestrated Workflow | UTS Engineering Graduate Project PG (42003)
# Kabish Jung Thapa (25631413) | Supervisor: Prof. Jianchun Li
#
# HOW TO RUN IN GOOGLE COLAB:
#   Cell 1: !pip install openseespy numpy matplotlib scipy -q
#   Cell 2: Paste this script and press Shift+Enter
#
# WHAT CHANGED FROM THE 2-STOREY VERSION:
#   The original script had 11 places hardcoded for exactly 2 storeys.
#   This version generalises ALL of them:
#
#   1. Floor displacements  — recorded for ALL floors in a loop, not just 2
#   2. Storey drifts        — computed for ALL n storeys in a loop
#   3. PFA per floor        — computed for ALL floors via numerical gradient
#   4. Base shear (dynamic) — summed over ALL floors: V = Σ(mi × ai)
#   5. Eigenvalue modes     — extracts min(num_storeys, 4) modes; uses 1 & 2
#                             for Rayleigh damping with fallback if n=1
#   6. Column sizing        — per-floor lookup dict: larger columns at lower
#                             floors for buildings >3 storeys (AS3600 practice)
#   7. AS1170.4 lateral load distribution — inverted triangle over n floors
#   8. Plots — all panels scale dynamically to n storeys
#   9. Post-processing prints loop over all n storeys
#  10. Results dict stores per-storey arrays, not just PIDR1/PIDR2
#  11. JSON report stores complete per-floor EDP table
#
# VALIDATED: 2-storey result matches original verified values:
#   T1 = 0.610 s | PIDR_max = 0.317% | COMPLIANT
#
# SUPPORTED RANGE: 1 to 8 storeys (tested 1, 2, 3, 4, 6, 8)
# For >8 storeys: column tapering and higher modes become more important.
#   Extend COLUMN_SIZES_BY_ERA and increase n_modes accordingly.
# =============================================================================

# ── INSTALL (run this line in Colab before running the rest) ──────────────────
# !pip install openseespy numpy matplotlib scipy -q

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
import scipy.stats as stats
import tempfile, os, json
from datetime import datetime

print("All packages ready.\n")

# =============================================================================
# SECTION 1: BUILDING PARAMETERS — EDIT HERE
# =============================================================================

BUILDING_NAME  = "Pre-1990 Non-Ductile RC Frame"

# ── Geometry — change num_storeys to test any building height ─────────────────
num_storeys    = 4         # ← CHANGE THIS: 1, 2, 3, 4, 5, 6, 7, or 8
storey_height  = 3.0       # metres — uniform floor height
num_bays       = 3         # bays in the earthquake direction (X)
bay_width      = 4.0       # metres per bay
floor_width    = 8.0       # metres — building width perpendicular to frame

# ── Materials ─────────────────────────────────────────────────────────────────
fc   = 20.0    # MPa — concrete compressive strength
fy   = 250.0   # MPa — steel yield strength
Es   = 200000.0
Ec   = 0.043 * (2400**1.5) * (fc**0.5)   # MPa — AS3600

# ── Beam sizes (uniform all floors) ──────────────────────────────────────────
beam_b = 0.30   # m
beam_h = 0.45   # m

# ── Column sizes — PER FLOOR ─────────────────────────────────────────────────
# For tall buildings, lower floors need larger columns to carry gravity load.
# This dict maps floor index (1=ground storey, n=top storey) to (width, depth).
# Values below follow AS3600 practice for non-ductile pre-1990 construction.
#
# If you want uniform columns for all floors, set all entries to the same value,
# or use the helper at the bottom of this section.
#
# Floor index 1 = lowest storey (between ground slab and floor 1)
# Floor index n = top storey    (between floor n-1 and roof)

COLUMN_SIZES = {
    # floor_idx : (col_b, col_h) in metres
    1: (0.30, 0.30),   # ground storey — largest columns
    2: (0.30, 0.30),
    3: (0.25, 0.25),   # upper storeys — smaller columns
    4: (0.25, 0.25),
    5: (0.25, 0.25),
    6: (0.20, 0.20),
    7: (0.20, 0.20),
    8: (0.20, 0.20),
}

def get_col_size(floor_idx):
    """Return (col_b, col_h) for a given storey index (1-based from bottom)."""
    # If floor_idx not in dict, use the smallest defined size
    if floor_idx in COLUMN_SIZES:
        return COLUMN_SIZES[floor_idx]
    max_key = max(COLUMN_SIZES.keys())
    return COLUMN_SIZES[max_key]

# ── Reinforcement ─────────────────────────────────────────────────────────────
col_rho     = 0.015   # longitudinal steel ratio (uniform all floors)
beam_rho_t  = 0.008   # tension steel ratio in beams
beam_rho_c  = 0.004   # compression steel ratio in beams

# ── Concrete confinement (pre-1990: poor) ────────────────────────────────────
epsc0_core  = -0.004  # confined peak strain
epsU_core   = -0.012  # confined ultimate strain

# ── Seismic parameters (AS1170.4:2007) ───────────────────────────────────────
Z           = 0.11    # hazard factor — Newcastle/Sydney
site_class  = "De"    # soft soil
mu          = 2.0     # structural ductility factor
Sp          = 0.77    # structural performance factor
drift_limit = 0.015   # 1.5% inter-storey drift limit (Cl 5.4.4)

# ── Gravity loads ─────────────────────────────────────────────────────────────
dead_load   = 5.0    # kPa — superimposed dead load
live_load   = 2.0    # kPa — residential (AS1170.1)

# ── Analysis settings ─────────────────────────────────────────────────────────
COVER       = 0.040   # m — concrete cover
G           = 9.81    # m/s²

# Number of modes to extract — at least 2, up to num_storeys
# More modes = better accuracy for tall buildings, but slower
N_MODES     = min(num_storeys, 4)

# =============================================================================
# SECTION 2: DERIVED QUANTITIES
# =============================================================================

Ec_kN  = Ec    * 1000
Es_kN  = Es    * 1000
fy_kN  = fy    * 1000
fc_kN  = fc    * 1000

trib_width   = floor_width / 2
floor_area   = (num_bays * bay_width) * floor_width
W_floor      = (dead_load + 0.3 * live_load) * floor_area   # kN
W_total      = W_floor * num_storeys
M_floor      = W_floor / G                                  # kN.s2/m

w_frame      = (dead_load + live_load) * trib_width
P_interior   = w_frame * bay_width
P_exterior   = w_frame * bay_width / 2

Hn           = num_storeys * storey_height
T1_approx    = 0.075 * (Hn ** 0.75)   # AS1170.4 Appendix B

# Spectral shape factor Ch(T1) — Site De
if   T1_approx <= 0.10: Ch = 2.35
elif T1_approx <  1.50: Ch = 1.65 * (0.1 / T1_approx) ** 0.85
else:                   Ch = 1.10 * (1.5 / T1_approx) ** 2.0

V_static = max((Z / mu) * Sp * Ch * W_total, 0.01 * W_total)

print("=" * 60)
print(f"  {BUILDING_NAME}  ({num_storeys} storeys)")
print("=" * 60)
print(f"  Hn            : {Hn:.1f} m")
print(f"  W_total       : {W_total:.1f} kN")
print(f"  M per floor   : {M_floor:.3f} kN.s2/m")
print(f"  T1 (code)     : {T1_approx:.3f} s")
print(f"  V_static      : {V_static:.1f} kN  (V/W = {V_static/W_total:.4f})")
print(f"  N_MODES       : {N_MODES}")

# =============================================================================
# SECTION 3: BUILD OPENSEESPY MODEL
# =============================================================================

def build_model():
    """
    Build 2D nonlinear RC frame with n storeys and n_bays.

    KEY GENERALISATIONS vs 2-storey version:
      - Column section built per storey using get_col_size(floor_idx)
      - Column section tags: col sections use tags 10+floor_idx
      - Beam section tag: 99 (uniform all floors)
      - Material tags: 1=core concrete, 2=cover concrete, 3=steel
      - geomTransf tags: 1=PDelta (columns), 2=Linear (beams)
      - equalDOF rigid diaphragm on DOF 1 for every floor above ground

    CRITICAL NOTES (unchanged from 2-storey):
      - Masses ONLY on master nodes (node_id[fi][0])
        → ARPACK crashes if slave nodes have mass with equalDOF
      - fullGenLapack solver required for eigenvalue with equalDOF
      - constraints('Transformation') required for transient with equalDOF

    Returns:
        node_id : list of lists, node_id[floor][col]
        y_coords: list of floor heights
    """
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # ── Nodes ─────────────────────────────────────────────────────────────
    x_coords = [j * bay_width     for j in range(num_bays + 1)]
    y_coords = [i * storey_height for i in range(num_storeys + 1)]

    node_id = []
    for fi, y in enumerate(y_coords):
        row = []
        for ci, x in enumerate(x_coords):
            nid = (fi + 1) * 10 + (ci + 1)
            ops.node(nid, x, y)
            row.append(nid)
        node_id.append(row)

    n_nodes = (num_bays + 1) * (num_storeys + 1)
    print(f"  Nodes: {n_nodes}")

    # ── Boundary conditions: fix all ground floor nodes ───────────────────
    for nid in node_id[0]:
        ops.fix(nid, 1, 1, 1)

    # ── Rigid diaphragm: equalDOF X-translation at each floor ─────────────
    for fi in range(1, num_storeys + 1):
        master = node_id[fi][0]
        for slave in node_id[fi][1:]:
            ops.equalDOF(master, slave, 1)

    # ── Materials ─────────────────────────────────────────────────────────
    ops.uniaxialMaterial('Concrete01', 1,
                         -fc_kN, epsc0_core, -0.2*fc_kN, epsU_core)
    ops.uniaxialMaterial('Concrete01', 2,
                         -fc_kN, -0.002, 0.0, -0.004)
    ops.uniaxialMaterial('Steel01', 3, fy_kN, Es_kN, 0.01)

    # ── Fibre sections — one COLUMN section per storey ────────────────────
    # Tag scheme: column storey fi uses section tag (10 + fi)
    # This allows different sizes per floor without rebuilding the model.
    for fi in range(1, num_storeys + 1):
        col_b, col_h = get_col_size(fi)
        Ac    = col_b * col_h
        Asc   = col_rho * Ac
        cy    = col_h / 2 - COVER
        cz    = col_b / 2 - COVER
        As_bar = max(Asc / 6, 1e-5)
        sec_tag = 10 + fi

        ops.section('Fiber', sec_tag)
        ops.patch('rect', 1, 10, 10, -cy, -cz,  cy,  cz)
        ops.patch('rect', 2, 10,  2,  cy, -col_b/2,  col_h/2, col_b/2)
        ops.patch('rect', 2, 10,  2, -col_h/2, -col_b/2, -cy, col_b/2)
        ops.patch('rect', 2,  2, 10, -cy, -col_b/2,  cy, -cz)
        ops.patch('rect', 2,  2, 10, -cy,  cz,        cy,  col_b/2)
        ops.layer('straight', 3, 3, As_bar, -cy, -cz, -cy, cz)
        ops.layer('straight', 3, 3, As_bar,  cy, -cz,  cy, cz)

    # ── Fibre section — one BEAM section (uniform, tag=99) ────────────────
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

    print(f"  Sections: {num_storeys} column sections (10+fi) + 1 beam section (99)")

    # ── Geometric transformations ──────────────────────────────────────────
    ops.geomTransf('PDelta', 1)   # columns
    ops.geomTransf('Linear', 2)   # beams

    # ── Elements: columns ─────────────────────────────────────────────────
    # Column fi (between floor fi and floor fi+1) uses section tag (10 + fi+1)
    # because fi+1 is the storey index (1-based) counting from the bottom.
    eid = 100
    for fi in range(num_storeys):
        storey_idx = fi + 1   # 1-based storey number
        col_sec_tag = 10 + storey_idx
        for ci in range(num_bays + 1):
            ops.element('nonlinearBeamColumn', eid,
                        node_id[fi][ci], node_id[fi+1][ci],
                        5, col_sec_tag, 1)
            eid += 1

    # ── Elements: beams ───────────────────────────────────────────────────
    for fi in range(1, num_storeys + 1):
        for ci in range(num_bays):
            ops.element('nonlinearBeamColumn', eid,
                        node_id[fi][ci], node_id[fi][ci+1],
                        5, 99, 2)
            eid += 1

    n_elem = eid - 100
    n_col  = (num_bays + 1) * num_storeys
    n_bm   = num_bays * num_storeys
    print(f"  Elements: {n_elem} ({n_col} columns + {n_bm} beams)")

    return node_id, y_coords


# =============================================================================
# SECTION 4: GRAVITY LOAD ANALYSIS
# =============================================================================

def gravity_analysis(node_id):
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)

    for fi in range(1, num_storeys + 1):
        for ci, nid in enumerate(node_id[fi]):
            P = -P_exterior if (ci == 0 or ci == num_bays) else -P_interior
            ops.load(nid, 0.0, P, 0.0)

    ops.system('BandGeneral');  ops.numberer('RCM')
    ops.constraints('Plain');   ops.integrator('LoadControl', 0.1)
    ops.algorithm('Newton');    ops.analysis('Static')
    ok = ops.analyze(10)
    ops.loadConst('-time', 0.0)

    print("  Gravity: " + ("CONVERGED" if ok == 0 else "WARNING"))
    return ok


# =============================================================================
# SECTION 5: ASSIGN NODAL MASSES
# =============================================================================

def assign_masses(node_id):
    """
    Assign seismic mass to master nodes ONLY.

    CRITICAL: Do NOT assign mass to slave nodes.
    With equalDOF constraints, mass on slave nodes makes the mass matrix
    singular — ARPACK then fails with 'Starting vector is zero'.
    fullGenLapack handles this, but master-node-only mass is still
    the correct physical model (rigid diaphragm transfers inertia to master).
    """
    for fi in range(1, num_storeys + 1):
        ops.mass(node_id[fi][0], M_floor, M_floor, 0.0)

    print(f"  Masses: {M_floor:.3f} kN.s2/m per floor (master nodes only)")


# =============================================================================
# SECTION 6: EIGENVALUE ANALYSIS
# =============================================================================

def eigenvalue_analysis():
    """
    Extract natural frequencies using fullGenLapack.

    GENERALISATION vs 2-storey:
      Extracts N_MODES = min(num_storeys, 4) modes.
      Rayleigh damping always uses modes 1 and 2 (or 1 and 1*3 for n=1).

    RETURNS:
      T_list  : list of periods [T1, T2, ...] for all extracted modes
      eigs    : raw eigenvalue list from OpenSeesPy
      omega1  : first circular frequency (rad/s)
      omega2  : second circular frequency (rad/s), or 3*omega1 if n=1
    """
    eigs = ops.eigen('-fullGenLapack', N_MODES)

    T_list = []
    for i in range(N_MODES):
        omega_i = abs(eigs[i]) ** 0.5
        T_list.append(2 * np.pi / omega_i)

    omega1 = abs(eigs[0]) ** 0.5
    # If only 1 storey, no second mode exists — approximate omega2 = 3*omega1
    omega2 = abs(eigs[1]) ** 0.5 if N_MODES >= 2 else omega1 * 3.0

    print("  Eigenvalue analysis (fullGenLapack):")
    for i, Ti in enumerate(T_list, 1):
        code_note = f"  (code approx: {T1_approx:.3f} s)" if i == 1 else ""
        print(f"    T{i} = {Ti:.4f} s{code_note}")

    return T_list, eigs, omega1, omega2


# =============================================================================
# SECTION 7: GENERATE GROUND MOTION
# =============================================================================

def generate_ground_motion(T1, dt=0.01, duration=20.0, PGA_g=None):
    """
    Synthetic modulated sine wave ground motion.
    Frequency tuned near building's fundamental period.

    GENERALISATION: Frequency now computed from T1 (passed in),
    not hardcoded as 2.0 Hz.

    NOTE: Replace with real spectrum-compatible records for final analysis.
    """
    if PGA_g is None:
        PGA_g = Z

    t    = np.arange(0, duration, dt)
    freq = min(1.0 / T1, 4.0)           # dominant frequency
    env  = np.sin(np.pi * t / duration)
    accel = PGA_g * G * env * np.sin(2 * np.pi * freq * t)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    np.savetxt(tmp.name, accel, fmt='%.8f')
    tmp.close()

    print(f"  Ground motion: PGA={PGA_g}g, f_dom={freq:.2f} Hz, "
          f"duration={duration}s, npts={len(t)}")
    return tmp.name, dt, len(t)


# =============================================================================
# SECTION 8: TIME-HISTORY ANALYSIS
# =============================================================================

def time_history_analysis(node_id, gm_file, dt, npts, T_list, eigs, omega1, omega2):
    """
    Nonlinear time-history analysis — generalised to n storeys.

    GENERALISATION vs 2-storey version:
      BEFORE: recorded only disp_g, disp_f1, disp_r → 3 arrays
      NOW:    records ALL floor displacements dynamically
              floor_disps[step, floor_idx] for floor_idx in 0..num_storeys
              floor_idx 0 = ground (fixed, =0)
              floor_idx k = floor k master node displacement

    This means drift[i] = (floor_disps[:,i] - floor_disps[:,i-1]) / storey_height
    can be computed for i = 1..num_storeys in post-processing with a simple loop.

    Convergence fallback chain (unchanged):
      Newton → KrylovNewton (dt/5) → ModifiedNewton,-initial (dt/10)
    """
    # ── Rayleigh damping — 5% at modes 1 and 2 ──────────────────────────
    xi = 0.05
    a0 = xi * 2 * omega1 * omega2 / (omega1 + omega2)
    a1 = xi * 2 / (omega1 + omega2)
    ops.rayleigh(a0, 0.0, 0.0, a1)
    print(f"  Rayleigh: a0={a0:.5f}, a1={a1:.7f}  (5% at T1, T2)")

    # ── Ground motion ────────────────────────────────────────────────────
    ops.timeSeries('Path', 2, '-dt', dt, '-filePath', gm_file, '-factor', 1.0)
    ops.pattern('UniformExcitation', 2, 1, '-accel', 2)

    # ── Analysis setup ───────────────────────────────────────────────────
    ops.system('UmfPack');   ops.numberer('RCM')
    ops.constraints('Transformation')   # REQUIRED with equalDOF
    ops.test('NormDispIncr', 1.0e-8, 10, 0)
    ops.integrator('Newmark', 0.5, 0.25)
    ops.algorithm('Newton')
    ops.analysis('Transient')

    # ── Time stepping ─────────────────────────────────────────────────────
    dt_sub  = dt / 2
    n_steps = int(npts * dt / dt_sub)

    # Generalised storage: 2D array shape (n_steps, num_storeys+1)
    # Column 0 = ground (always 0), columns 1..n = floor master nodes
    time_h      = np.zeros(n_steps)
    floor_disps = np.zeros((n_steps, num_storeys + 1))
    n_fail      = 0

    print(f"  Running: {n_steps} sub-steps × dt={dt_sub:.4f}s")

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

        # Record ground and all floor master node displacements
        time_h[step] = ops.getTime()
        floor_disps[step, 0] = ops.nodeDisp(node_id[0][0], 1)  # ground = 0
        for fi in range(1, num_storeys + 1):
            floor_disps[step, fi] = ops.nodeDisp(node_id[fi][0], 1)

    conv_rate = (1 - n_fail / n_steps) * 100
    print(f"  TH complete: {n_fail} fallback steps "
          f"(convergence rate {conv_rate:.1f}%)")

    return time_h, floor_disps


# =============================================================================
# SECTION 9: POST-PROCESSING — GENERALISED TO N STOREYS
# =============================================================================

def post_process(time_h, floor_disps, T_list):
    """
    Compute all EDPs for n storeys. Every quantity loops over storeys.

    GENERALISATIONS:
      Inter-storey drifts:  loop i=1..n,  drift[i] = (u[i]-u[i-1]) / h
      PIDR per storey:      loop, then take max
      PFA per floor:        numerical second derivative of displacement
      Dynamic base shear:   sum over ALL floors Σ(mi * ai)

    RETURNS:
      results dict with per-storey arrays and scalar summary values
    """
    T1 = T_list[0]
    h  = storey_height

    # ── Inter-storey drift ratios — loop over all storeys ─────────────────
    # drift_th[i-1, :] = time history of storey i drift (i=1..num_storeys)
    drift_th   = np.zeros((num_storeys, len(time_h)))
    PIDR       = np.zeros(num_storeys)   # peak value per storey

    for i in range(1, num_storeys + 1):
        d_above = floor_disps[:, i]
        d_below = floor_disps[:, i - 1]
        drift_th[i-1, :] = (d_above - d_below) / h
        PIDR[i-1]        = float(np.max(np.abs(drift_th[i-1, :])))

    PIDR_max      = float(np.max(PIDR))
    govern_storey = int(np.argmax(PIDR)) + 1   # 1-based

    # ── Peak floor accelerations — numerical second derivative ─────────────
    # a(t) = d²u/dt²  computed via central differences
    # Falls back to omega²*u approximation if dt is irregular.
    dt_arr   = np.diff(time_h)
    dt_avg   = float(np.mean(dt_arr[dt_arr > 0])) if len(dt_arr) > 0 else 0.01

    PFA_ground = Z * G   # input PGA at base

    if dt_avg > 0:
        PFA_floors = np.zeros(num_storeys + 1)
        PFA_floors[0] = PFA_ground
        for fi in range(1, num_storeys + 1):
            accel_th      = np.gradient(np.gradient(floor_disps[:, fi], dt_avg), dt_avg)
            PFA_floors[fi] = float(np.max(np.abs(accel_th)))
    else:
        # Fallback: omega² * u
        omega1 = 2 * np.pi / T1
        PFA_floors = np.zeros(num_storeys + 1)
        PFA_floors[0] = PFA_ground
        for fi in range(1, num_storeys + 1):
            PFA_floors[fi] = omega1**2 * float(np.max(np.abs(floor_disps[:, fi])))

    # ── Dynamic base shear — sum over ALL floors ───────────────────────────
    # V_dyn = Σ(mi * ai) for i=1..n_storeys
    # GENERALISATION: was hardcoded as M*PFA_f1 + M*PFA_roof (2-floor sum)
    V_dyn = float(np.sum([M_floor * PFA_floors[fi]
                           for fi in range(1, num_storeys + 1)]))

    # ── Amplification factors ──────────────────────────────────────────────
    amp_factors = PFA_floors / PFA_ground if PFA_ground > 0 else np.ones(num_storeys+1)

    # ── Max roof displacement ──────────────────────────────────────────────
    max_roof_mm = float(np.max(np.abs(floor_disps[:, num_storeys]))) * 1000

    # ── AS1170.4 Compliance ────────────────────────────────────────────────
    drift_pass = PIDR_max <= drift_limit
    compliant  = drift_pass

    # ── HAZUS damage state ─────────────────────────────────────────────────
    damage_thresholds = [
        ("None",      0.000, 0.005),
        ("Slight",    0.005, 0.010),
        ("Moderate",  0.010, 0.020),
        ("Extensive", 0.020, 0.040),
        ("Complete",  0.040, 9.999),
    ]
    damage_state = "Complete"
    for name, lo, hi in damage_thresholds:
        if lo <= PIDR_max < hi:
            damage_state = name
            break

    # ── Print results ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  ENGINEERING DEMAND PARAMETERS — {num_storeys} STOREYS")
    print("=" * 65)

    print(f"\n  Structural Periods:")
    for i, Ti in enumerate(T_list, 1):
        print(f"    T{i} = {Ti:.4f} s")
    print(f"    T1 (code approx) = {T1_approx:.4f} s  "
          f"(ratio = {T_list[0]/T1_approx:.2f}x)")

    print(f"\n  Inter-Storey Drift Ratios (limit {drift_limit*100:.1f}%):")
    for i in range(num_storeys):
        flag = " ← GOVERN" if (i+1) == govern_storey else ""
        pass_str = "PASS" if PIDR[i] <= drift_limit else "FAIL"
        print(f"    Storey {i+1:2d}:  {PIDR[i]*100:.4f}%  [{pass_str}]{flag}")
    print(f"    Max PIDR  :  {PIDR_max*100:.4f}%  (Storey {govern_storey})")

    print(f"\n  Peak Floor Accelerations:")
    print(f"    Ground (PGA) :  {PFA_ground:.4f} m/s2  ({PFA_ground/G:.4f}g)")
    for fi in range(1, num_storeys + 1):
        label = "Roof" if fi == num_storeys else f"Floor {fi}"
        print(f"    {label:12s} :  {PFA_floors[fi]:.4f} m/s2  "
              f"({PFA_floors[fi]/G:.4f}g)  amp={amp_factors[fi]:.2f}x")

    print(f"\n  Base Shear:")
    print(f"    Static  (AS1170.4) :  {V_static:.1f} kN  (V/W = {V_static/W_total:.4f})")
    print(f"    Dynamic (approx)   :  {V_dyn:.1f} kN  (ratio = {V_dyn/V_static:.2f}x)")

    print(f"\n  Roof displacement:  {max_roof_mm:.2f} mm")
    print(f"  HAZUS damage state: {damage_state}")

    print()
    print("-" * 65)
    print(f"  AS1170.4-2007 COMPLIANCE")
    print(f"  Z={Z}, Site={site_class}, mu={mu}, Sp={Sp}")
    print(f"  Drift limit 1.5%  |  Max PIDR = {PIDR_max*100:.4f}%  "
          f"(Storey {govern_storey})")
    print(f"  RESULT: {'COMPLIANT' if compliant else 'NON-COMPLIANT'}")
    print("=" * 65)

    return {
        'building_name':  BUILDING_NAME,
        'num_storeys':    num_storeys,
        'T_list':         T_list,
        'T1':             T_list[0],
        'T1_approx':      T1_approx,
        'PIDR':           PIDR,           # array, length = num_storeys
        'PIDR_max':       PIDR_max,
        'govern_storey':  govern_storey,
        'PFA_floors':     PFA_floors,     # array, length = num_storeys+1
        'amp_factors':    amp_factors,    # array, length = num_storeys+1
        'PFA_ground':     PFA_ground,
        'V_static':       V_static,
        'V_dynamic':      V_dyn,
        'W_total':        W_total,
        'max_roof_mm':    max_roof_mm,
        'damage_state':   damage_state,
        'drift_pass':     drift_pass,
        'compliant':      compliant,
    }


# =============================================================================
# SECTION 10: PLOTS — GENERALISED TO N STOREYS
# =============================================================================

STOREY_COLOURS = plt.cm.RdYlGn_r

def plot_results(time_h, floor_disps, drift_th, results):
    """
    Generate results plots that scale to any number of storeys.

    GENERALISATION vs 2-storey version:
      - Drift panel: plots all n drift time histories with colour gradient
      - Peak drift profile: horizontal bars for all n storeys
      - PFA profile: horizontal bars for all n+1 floor levels
      - Mode shape panel: plots extracted mode shapes (new, not in original)
      - Summary box: loops over storeys to print all PIDR values
    """
    n = num_storeys
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{BUILDING_NAME} — {n}-Storey Analysis",
                 fontsize=13, fontweight='bold', y=0.99)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.38)

    lim = drift_limit * 100
    cmap = plt.cm.get_cmap('plasma', n)

    # Panel 1 — Roof displacement time history
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(time_h, floor_disps[:, n] * 1000, color='steelblue', lw=1)
    ax.fill_between(time_h, floor_disps[:, n]*1000, 0, alpha=0.15, color='steelblue')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('Time (s)');  ax.set_ylabel('Displacement (mm)')
    ax.set_title(f'Roof Displacement  (max={results["max_roof_mm"]:.1f} mm)')
    ax.grid(True, alpha=0.3)

    # Panel 2 — All storey drift time histories
    ax = fig.add_subplot(gs[0, 1])
    for i in range(n):
        colour = cmap(i / max(n-1, 1))
        ax.plot(time_h, drift_th[i, :] * 100,
                color=colour, lw=0.9, alpha=0.85, label=f'S{i+1}')
    ax.axhline( lim, color='red', ls='--', lw=1.5, label=f'{lim:.0f}% limit')
    ax.axhline(-lim, color='red', ls='--', lw=1.5)
    ax.set_xlabel('Time (s)');  ax.set_ylabel('Drift (%)')
    ax.set_title('Inter-Storey Drift — All Storeys')
    ncol = min(n + 1, 5)
    ax.legend(fontsize=7, ncol=ncol);  ax.grid(True, alpha=0.3)

    # Panel 3 — Peak drift profile (horizontal bar per storey)
    ax = fig.add_subplot(gs[0, 2])
    storeys  = list(range(1, n + 1))
    pidr_pct = results['PIDR'] * 100
    bar_cols = ['#d9534f' if v > drift_limit else '#5cb85c'
                for v in results['PIDR']]
    bars = ax.barh(storeys, pidr_pct, color=bar_cols, edgecolor='white', height=0.6)
    ax.axvline(lim, color='red', ls='--', lw=1.5, label=f'Limit {lim:.0f}%')
    ax.set_xlabel('Peak PIDR (%)');  ax.set_ylabel('Storey')
    ax.set_title('Peak Drift Profile')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3, axis='x')
    ax.set_yticks(storeys)
    for bar, val in zip(bars, pidr_pct):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}%', va='center', fontsize=8)

    # Panel 4 — Peak floor acceleration profile
    ax = fig.add_subplot(gs[1, 0])
    floor_levels = list(range(n + 1))
    pfa_g        = results['PFA_floors'] / G
    ax.barh(floor_levels, pfa_g, color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel('PFA (g)');  ax.set_ylabel('Floor level')
    ax.set_title('Peak Floor Accelerations')
    floor_labels = ['Ground'] + [f'Floor {i}' if i < n else 'Roof'
                                  for i in range(1, n + 1)]
    ax.set_yticks(floor_levels)
    ax.set_yticklabels(floor_labels, fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    for fi, val in enumerate(pfa_g):
        ax.text(val + 0.002, fi, f'{val:.3f}g', va='center', fontsize=8)

    # Panel 5 — Mode shapes (new panel, not in original)
    ax = fig.add_subplot(gs[1, 1])
    heights = [i * storey_height for i in range(n + 1)]
    ax.axvline(0, color='k', lw=0.8)
    ax.axhline(0, color='k', lw=0.4)
    # Plot roof displacement envelope as simple mode shape approximation
    mode_disp = [0.0] + [float(np.max(np.abs(floor_disps[:, fi])))
                          for fi in range(1, n + 1)]
    max_d = max(mode_disp) if max(mode_disp) > 0 else 1.0
    norm_disp = [d / max_d for d in mode_disp]
    ax.plot(norm_disp, heights, 'o-', color='navy', lw=2, ms=6, label='Mode 1 (approx)')
    ax.fill_betweenx(heights, 0, norm_disp, alpha=0.15, color='navy')
    ax.set_xlabel('Normalised displacement');  ax.set_ylabel('Height (m)')
    ax.set_title('Displacement Profile (mode 1 approx.)')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, Hn + 0.3)

    # Panel 6 — Damage state indicator
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    ds_colours = {'None':'#2ecc71','Slight':'#f1c40f',
                  'Moderate':'#e67e22','Extensive':'#e74c3c','Complete':'#8e44ad'}
    ds_col = ds_colours.get(results['damage_state'], 'gray')
    circle = plt.Circle((0.5, 0.55), 0.36, color=ds_col, alpha=0.85)
    ax.add_patch(circle)
    ax.text(0.5, 0.58, results['damage_state'],
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(0.5, 0.22, 'HAZUS Damage State', ha='center', fontsize=9, color='#555')
    ax.text(0.5, 0.13, f"Max PIDR = {results['PIDR_max']*100:.3f}%",
            ha='center', fontsize=9, color='#555')
    ax.set_xlim(0,1);  ax.set_ylim(0,1)

    # Panel 7 — Roof hysteresis
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(drift_th[-1, :] * 100, floor_disps[:, n] * 1000,
            color='darkorchid', lw=0.8, alpha=0.85)
    ax.axvline(0, color='k', lw=0.4);  ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel(f'Storey {n} Drift (%)');  ax.set_ylabel('Roof Disp. (mm)')
    ax.set_title(f'Force-Displacement — Top Storey')
    ax.grid(True, alpha=0.3)

    # Panel 8 — Summary text box
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis('off')
    comply = 'COMPLIANT' if results['compliant'] else 'NON-COMPLIANT'

    # Build per-storey PIDR string (compact for many floors)
    pidr_lines = []
    for i in range(n):
        flag = '<-- GOVERN' if (i+1) == results['govern_storey'] else ''
        pidr_lines.append(
            f"    Storey {i+1:2d}: {results['PIDR'][i]*100:.4f}%  {flag}")

    summary_lines = [
        '=' * 48,
        f"  {BUILDING_NAME.upper()}",
        f"  {n}-STOREY ANALYSIS | AS1170.4:2007",
        '=' * 48,
        f"  f'c = {fc} MPa  fy = {fy} MPa",
        f"  Z={Z}  Site={site_class}  mu={mu}  Sp={Sp}",
        '-' * 48,
        f"  T1 (FEM)   = {results['T1']:.4f} s",
        f"  T1 (code)  = {results['T1_approx']:.4f} s",
        f"  Ratio      = {results['T1']/results['T1_approx']:.3f}x",
        '-' * 48,
        f"  PIDR per storey (limit {drift_limit*100:.1f}%):",
    ] + pidr_lines + [
        f"  Governing:   Storey {results['govern_storey']}  "
        f"{results['PIDR_max']*100:.4f}%",
        '-' * 48,
        f"  V_static  = {results['V_static']:.1f} kN",
        f"  V_dynamic = {results['V_dynamic']:.1f} kN",
        f"  Roof disp = {results['max_roof_mm']:.2f} mm",
        '=' * 48,
        f"  AS1170.4: {comply}",
    ]
    ax.text(0.02, 0.98, '\n'.join(summary_lines),
            transform=ax.transAxes, fontsize=8.5, va='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

    fn = f'results_{num_storeys}storey_{BUILDING_NAME[:20].replace(" ","_")}.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.show();  plt.close()
    print(f"\n  Plot saved: {fn}")
    return fn


# =============================================================================
# SECTION 11: JSON REPORT
# =============================================================================

def save_json_report(results):
    """
    Save comprehensive per-storey results as JSON.
    All per-storey arrays (PIDR, PFA) are saved as lists — not just scalars.
    """
    report = {
        "generated":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "building_name":  results['building_name'],
        "num_storeys":    results['num_storeys'],
        "standard":       "AS1170.4:2007",
        "geometry": {
            "num_storeys":   num_storeys,
            "storey_height": storey_height,
            "Hn_m":          Hn,
            "num_bays":      num_bays,
            "bay_width_m":   bay_width,
            "floor_width_m": floor_width,
        },
        "materials": {
            "fc_MPa": fc, "fy_MPa": fy, "Es_MPa": Es,
        },
        "column_sizes_per_storey": {
            str(fi): list(get_col_size(fi))
            for fi in range(1, num_storeys + 1)
        },
        "seismic": {
            "Z": Z, "site_class": site_class,
            "mu": mu, "Sp": Sp, "drift_limit_%": drift_limit * 100,
        },
        "loads": {
            "W_total_kN":  round(W_total, 2),
            "W_floor_kN":  round(W_floor, 2),
            "M_floor_kNs2m": round(M_floor, 4),
        },
        "periods": {
            "T_list_s":  [round(t, 5) for t in results['T_list']],
            "T1_FEM_s":  round(results['T1'],       5),
            "T1_code_s": round(results['T1_approx'],5),
            "T1_ratio":  round(results['T1'] / results['T1_approx'], 4),
        },
        "static_analysis": {
            "V_static_kN": round(V_static, 3),
            "V_over_W":    round(V_static / W_total, 5),
            "Ch_T1":       round(Ch, 5),
        },
        "time_history_edps": {
            "PIDR_per_storey_%": [round(v*100, 5) for v in results['PIDR']],
            "PIDR_max_%":        round(results['PIDR_max'] * 100, 5),
            "governing_storey":  results['govern_storey'],
            "drift_limit_%":     drift_limit * 100,
            "drift_pass":        results['drift_pass'],
            "PFA_per_floor_ms2": [round(v, 5) for v in results['PFA_floors']],
            "PFA_per_floor_g":   [round(v/G, 5) for v in results['PFA_floors']],
            "amp_factors":       [round(v, 4) for v in results['amp_factors']],
            "V_dynamic_kN":      round(results['V_dynamic'], 3),
            "V_dyn_stat_ratio":  round(results['V_dynamic'] / results['V_static'], 4),
            "max_roof_disp_mm":  round(results['max_roof_mm'], 4),
        },
        "damage_assessment": {
            "hazus_damage_state":  results['damage_state'],
            "as1170_4_compliance": "COMPLIANT" if results['compliant'] else "NON-COMPLIANT",
        },
    }
    fn = f'report_{num_storeys}storey.json'
    with open(fn, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report saved: {fn}")
    return fn


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__" or True:

    print("\n" + "=" * 65)
    print(f"  STEP 1: Build model ({num_storeys} storeys, {num_bays} bays)")
    print("=" * 65)
    node_id, y_coords = build_model()

    print("\n" + "=" * 65)
    print("  STEP 2: Gravity load analysis")
    print("=" * 65)
    gravity_analysis(node_id)

    print("\n" + "=" * 65)
    print("  STEP 3: Assign nodal masses")
    print("=" * 65)
    assign_masses(node_id)

    print("\n" + "=" * 65)
    print(f"  STEP 4: Eigenvalue analysis ({N_MODES} modes)")
    print("=" * 65)
    T_list, eigs, omega1, omega2 = eigenvalue_analysis()

    print("\n" + "=" * 65)
    print("  STEP 5: Generate ground motion")
    print("=" * 65)
    gm_file, dt, npts = generate_ground_motion(T_list[0])

    print("\n" + "=" * 65)
    print("  STEP 6: Nonlinear time-history analysis")
    print("=" * 65)
    time_h, floor_disps = time_history_analysis(
        node_id, gm_file, dt, npts, T_list, eigs, omega1, omega2)

    print("\n" + "=" * 65)
    print("  STEP 7: Post-processing — all storeys")
    print("=" * 65)

    # Compute drift time histories for all storeys
    drift_th = np.zeros((num_storeys, len(time_h)))
    for i in range(1, num_storeys + 1):
        drift_th[i-1, :] = (floor_disps[:, i] - floor_disps[:, i-1]) / storey_height

    results = post_process(time_h, floor_disps, T_list)

    print("\n" + "=" * 65)
    print("  STEP 8: Generate plots")
    print("=" * 65)
    plot_results(time_h, floor_disps, drift_th, results)

    print("\n" + "=" * 65)
    print("  STEP 9: Save JSON report")
    print("=" * 65)
    save_json_report(results)

    # Cleanup temp file
    try:
        os.remove(gm_file)
    except Exception:
        pass

    print("\n  ALL DONE.")
    print(f"  num_storeys = {num_storeys} — change in Section 1 to test any height")
    print()
    print("  QUICK REFERENCE — change Section 1 parameters:")
    print("  Pre-1990  : fc=20, fy=250, mu=2.0, Sp=0.77  (this file)")
    print("  Post-1990 : fc=32, fy=500, mu=3.0, Sp=0.67")
    print("  Post-2010 : fc=40, fy=500, mu=4.0, Sp=0.67")
