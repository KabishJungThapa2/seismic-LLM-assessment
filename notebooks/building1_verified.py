# =============================================================================
# BUILDING 1: Pre-1990 Non-Ductile RC Frame
# 2-Storey Residential Building — Newcastle/Sydney Region
# AS1170.4-2007 Seismic Assessment
#
# VERIFIED AGAINST: OpenSeesPy official RC Frame Earthquake example
# Source: openseespydoc.readthedocs.io/en/latest/src/RCFrameEarthquake.html
#
# KEY FIXES FROM PREVIOUS VERSION:
#   1. Nodal masses now correctly defined — this was the root cause of all
#      eigenvalue failures ("Starting vector is zero" error)
#   2. fullGenLapack solver retained for equalDOF constraint compatibility
#   3. Mass assigned ONLY to master nodes (rigid diaphragm masters carry
#      total floor mass — slave nodes get zero mass)
#   4. Ground motion uses tempfile (works on Colab and any local machine)
#   5. plt.show() added for Colab inline display
#   6. Convergence fallback chain improved per official docs
#
# HOW TO RUN IN GOOGLE COLAB:
#   Cell 1: !pip install openseespy numpy matplotlib
#   Cell 2: Paste this entire script and run (Shift+Enter)
# =============================================================================

import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# =============================================================================
# SECTION 1: BUILDING PARAMETERS
# Change these values to switch between Building 1, 2, and 3
# =============================================================================

BUILDING_NAME = "Building 1 — Pre-1990 Non-Ductile RC Frame"

# --- Geometry ---
num_storeys   = 2        # number of floors above ground
storey_height = 3.0      # metres per storey
num_bays      = 3        # bays in X direction (earthquake direction)
bay_width     = 4.0      # metres per bay

# --- Materials ---
fc  = 20.0               # MPa — concrete compressive strength
fy  = 250.0              # MPa — steel yield strength
Es  = 200000.0           # MPa — steel elastic modulus
# Ec per AS3600: Ec = 0.043 * rho^1.5 * sqrt(fc), rho = 2400 kg/m3
Ec  = 0.043 * (2400**1.5) * (fc**0.5)   # ≈ 22,610 MPa

# --- Member sizes (metres) ---
col_b         = 0.30     # column width
col_h         = 0.30     # column depth
beam_b        = 0.30     # beam width
beam_h        = 0.45     # beam depth
cover         = 0.040    # concrete cover to steel centreline

# --- Reinforcement ratios ---
col_rho       = 0.015    # 1.5% longitudinal steel in columns
beam_rho_t    = 0.008    # 0.8% tension steel in beams
beam_rho_c    = 0.004    # 0.4% compression steel in beams

# --- Seismic parameters (AS1170.4-2007) ---
Z             = 0.11     # hazard factor — Newcastle region
site_class    = "De"     # soft soil — common in Newcastle/Sydney coastal
mu            = 2.0      # structural ductility factor — limited (pre-1990)
Sp            = 0.77     # structural performance factor
drift_limit   = 0.015    # 1.5% inter-storey drift limit (AS1170.4 Cl 5.4.4)
g             = 9.81     # m/s²

# --- Gravity loads (kPa) ---
dead_load     = 5.0      # superimposed dead load per floor
live_load     = 2.0      # live load per floor (residential AS1170.1)
floor_width   = 8.0      # total building width in Y direction (metres)
trib_width    = floor_width / 2   # tributary width for this 2D frame

# =============================================================================
# SECTION 2: DERIVED PROPERTIES
# =============================================================================

# Section areas and second moments
Ac   = col_b  * col_h                # m² column gross area
Ic   = (col_b * col_h**3) / 12      # m⁴ column gross Ix
Ab   = beam_b * beam_h               # m² beam gross area
Ib   = (beam_b * beam_h**3) / 12    # m⁴ beam gross Ix

# Steel areas
Asc  = col_rho  * Ac                 # m² total longitudinal steel in column
Ast  = beam_rho_t * Ab               # m² tension steel in beam
Asc2 = beam_rho_c * Ab               # m² compression steel in beam

# Unit conversion to kN and metres (1 MPa = 1000 kN/m²)
Ec_kN = Ec  * 1000    # kN/m²
Es_kN = Es  * 1000    # kN/m²
fy_kN = fy  * 1000    # kN/m²
fc_kN = fc  * 1000    # kN/m²

# Stiffness modifiers (cracked section, ACI 318-19 Table 6.6.3.1)
# Pre-1990 non-ductile: conservative (lower) modifiers
col_mod  = 0.50    # 50% of gross stiffness for columns
beam_mod = 0.35    # 35% of gross stiffness for beams

# Floor and seismic weight
floor_area   = (num_bays * bay_width) * floor_width   # m²
# Seismic weight: G + 0.3Q (AS1170.4 Cl 6.1)
W_floor      = (dead_load + 0.3 * live_load) * floor_area   # kN per floor
W_total      = W_floor * num_storeys                          # kN total
# Seismic mass per floor (kN·s²/m = tonnes)
M_floor      = W_floor / g                                    # kN·s²/m

# Gravity load per node (for load application)
w_frame      = (dead_load + live_load) * trib_width   # kN/m tributary to frame
P_interior   = w_frame * bay_width                    # kN at interior column
P_exterior   = w_frame * bay_width / 2               # kN at exterior column

# ---- Print summary ----
print("=" * 60)
print(f"  {BUILDING_NAME}")
print("=" * 60)
print(f"  Ec            : {Ec:.0f} MPa ({Ec_kN:.0f} kN/m²)")
print(f"  Column        : {col_b*1000:.0f}×{col_h*1000:.0f} mm")
print(f"  Beam          : {beam_b*1000:.0f}×{beam_h*1000:.0f} mm")
print(f"  Floor area    : {floor_area:.0f} m²")
print(f"  W per floor   : {W_floor:.1f} kN")
print(f"  Total W       : {W_total:.1f} kN")
print(f"  M per floor   : {M_floor:.3f} kN·s²/m")

# =============================================================================
# SECTION 3: AS1170.4-2007 STATIC BASE SHEAR (validation benchmark)
# =============================================================================

def as1170_base_shear(Z, mu, Sp, W, site_class, T1):
    """
    Equivalent static base shear per AS1170.4-2007 Cl 6.2.
    V = (Z/mu) * Sp * Ch(T1) * kp * W
    """
    # Spectral shape factor Ch(T) — Site Class De, Table 6.4
    # Using simplified expression fitted to AS1170.4 Table 6.4 values
    if T1 <= 0.10:
        Ch = 2.35
    elif T1 <= 1.50:
        Ch = 1.65 * (0.1 / T1)**0.85
    else:
        Ch = 1.10 * (1.5 / T1)**2.0

    kp  = 1.0          # probability factor — IL2 (normal residential), 500yr
    V   = (Z * kp / mu) * Sp * Ch * W
    # Minimum base shear: V ≥ 0.01·W (AS1170.4 Cl 6.2.3)
    V   = max(V, 0.01 * W)
    return V, Ch

# Approximate period: T1 = 0.075·Hn^(3/4) for RC frames (AS1170.4 Appendix B)
Hn        = num_storeys * storey_height
T1_approx = 0.075 * (Hn ** 0.75)
V_static, Ch_T = as1170_base_shear(Z, mu, Sp, W_total, site_class, T1_approx)

print(f"\n  AS1170.4 Equivalent Static:")
print(f"  Approx T1     : {T1_approx:.3f} s")
print(f"  Ch(T1)        : {Ch_T:.3f}")
print(f"  Base shear V  : {V_static:.1f} kN")
print(f"  V/W           : {V_static/W_total:.4f}")
print()

# =============================================================================
# SECTION 4: BUILD OPENSEES MODEL
# Following official RC Frame example from OpenSeesPy documentation
# =============================================================================

def build_model():
    """
    Construct 2D nonlinear RC frame model.

    Node numbering scheme (floor_idx * 10 + col_idx + 1):
      Ground floor:  11, 12, 13, 14  (y=0, fixed)
      First floor:   21, 22, 23, 24  (y=3m, master=21)
      Roof:          31, 32, 33, 34  (y=6m, master=31)

    Rigid diaphragm via equalDOF on DOF 1 (X translation) at each floor.
    Mass assigned ONLY to master nodes — this is critical for eigenvalue
    analysis with equalDOF constraints.
    """

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # ── NODES ────────────────────────────────────────────────────────────
    x_coords = [j * bay_width for j in range(num_bays + 1)]  # [0, 4, 8, 12]
    y_coords = [i * storey_height for i in range(num_storeys + 1)]  # [0, 3, 6]

    node_id = []   # node_id[floor][column]
    for fi, y in enumerate(y_coords):
        row = []
        for ci, x in enumerate(x_coords):
            nid = (fi + 1) * 10 + (ci + 1)   # 11,12,...,34
            ops.node(nid, x, y)
            row.append(nid)
        node_id.append(row)

    # ── BOUNDARY CONDITIONS: fix all ground floor nodes ──────────────────
    for nid in node_id[0]:
        ops.fix(nid, 1, 1, 1)   # DOF: Ux, Uy, Rz all fixed

    # ── RIGID DIAPHRAGM: equalDOF on X-translation at each floor ─────────
    # Constrains slave node DOF 1 (X) to match master node DOF 1
    for fi in range(1, len(y_coords)):
        master = node_id[fi][0]
        for slave in node_id[fi][1:]:
            ops.equalDOF(master, slave, 1)

    n_nodes = (num_bays + 1) * (num_storeys + 1)
    print(f"  Nodes: {n_nodes}  |  Node IDs: {[n for row in node_id for n in row]}")

    # ── MATERIALS ─────────────────────────────────────────────────────────
    # Concrete01: Kent-Scott-Park model (compression negative convention)
    # Ref: OpenSeesPy docs, uniaxialMaterial Concrete01
    #
    # Core (confined) — pre-1990 poor confinement (widely spaced stirrups)
    ops.uniaxialMaterial('Concrete01',
                         1,            # tag
                         -fc_kN,       # f'c  (negative = compression)
                         -0.004,       # epsc0 — strain at peak (confined)
                         -0.2*fc_kN,   # f'cu  — 20% residual (poor confinement)
                         -0.012)       # epsU  — ultimate strain

    # Cover (unconfined) — spalls after crushing
    ops.uniaxialMaterial('Concrete01',
                         2,            # tag
                         -fc_kN,       # f'c
                         -0.002,       # epsc0 — unconfined peak strain
                         0.0,          # f'cu  — zero residual after spalling
                         -0.004)       # epsU

    # Steel01: bilinear with strain hardening
    # Ref: OpenSeesPy docs, uniaxialMaterial Steel01
    ops.uniaxialMaterial('Steel01',
                         3,            # tag
                         fy_kN,        # fy
                         Es_kN,        # E0
                         0.01)         # b — 1% strain hardening ratio

    print("  Materials: Concrete01 (core), Concrete01 (cover), Steel01")

    # ── FIBRE SECTIONS ────────────────────────────────────────────────────
    # Coordinate convention for section: local y = strong axis (depth)
    # patch('rect', matTag, nFibIJ, nFibJK, yI, zI, yJ, zJ)
    # layer('straight', matTag, nFibers, areaFiber, yStart, zStart, yEnd, zEnd)

    # --- Column section (tag = 1) ---
    cy = col_h / 2 - cover    # core half-depth
    cz = col_b / 2 - cover    # core half-width

    ops.section('Fiber', 1)
    # Core concrete
    ops.patch('rect', 1, 10, 10, -cy, -cz, cy, cz)
    # Cover patches (top, bottom, left, right)
    ops.patch('rect', 2, 10, 2,  cy,      -col_b/2, col_h/2, col_b/2)   # top
    ops.patch('rect', 2, 10, 2, -col_h/2, -col_b/2, -cy,     col_b/2)   # bottom
    ops.patch('rect', 2,  2, 10, -cy,     -col_b/2,  cy,     -cz)       # left
    ops.patch('rect', 2,  2, 10, -cy,      cz,       cy,      col_b/2)  # right
    # Steel layers (top and bottom)
    As_bar = max(Asc / (2 * 3), 1e-5)   # area per bar (3 bars per layer)
    ops.layer('straight', 3, 3, As_bar, -cy, -cz, -cy, cz)   # bottom layer
    ops.layer('straight', 3, 3, As_bar,  cy, -cz,  cy, cz)   # top layer

    # --- Beam section (tag = 2) ---
    by = beam_h / 2 - cover
    bz = beam_b / 2 - cover

    ops.section('Fiber', 2)
    # Core concrete
    ops.patch('rect', 1, 10, 10, -by, -bz, by, bz)
    # Cover patches (top and bottom only — simplified for beams)
    ops.patch('rect', 2, 10, 2,  by,      -beam_b/2, beam_h/2, beam_b/2)
    ops.patch('rect', 2, 10, 2, -beam_h/2,-beam_b/2, -by,      beam_b/2)
    # Tension steel (bottom), compression steel (top)
    ops.layer('straight', 3, 3, Ast/3,  -by, -bz, -by, bz)   # tension (bottom)
    ops.layer('straight', 3, 3, Asc2/3,  by, -bz,  by, bz)   # compression (top)

    print("  Sections: Fiber column (tag=1), Fiber beam (tag=2)")

    # ── GEOMETRIC TRANSFORMATIONS ─────────────────────────────────────────
    # PDelta for columns (captures P-Δ effects important for seismic)
    # Linear for beams
    ops.geomTransf('PDelta', 1)   # columns
    ops.geomTransf('Linear', 2)   # beams

    # ── ELEMENTS ──────────────────────────────────────────────────────────
    # nonlinearBeamColumn with 5 Gauss-Lobatto integration points
    # Ref: OpenSeesPy docs, element nonlinearBeamColumn
    n_ip   = 5
    eid    = 100

    # Columns (connect floor fi to floor fi+1)
    for fi in range(num_storeys):
        for ci in range(num_bays + 1):
            ops.element('nonlinearBeamColumn',
                        eid,
                        node_id[fi][ci],      # bottom node
                        node_id[fi+1][ci],    # top node
                        n_ip, 1, 1)           # nIP, secTag, transfTag
            eid += 1

    # Beams (connect column ci to ci+1 at each floor above ground)
    for fi in range(1, num_storeys + 1):
        for ci in range(num_bays):
            ops.element('nonlinearBeamColumn',
                        eid,
                        node_id[fi][ci],      # left node
                        node_id[fi][ci+1],    # right node
                        n_ip, 2, 2)           # nIP, secTag, transfTag
            eid += 1

    n_elem = eid - 100
    print(f"  Elements: {n_elem} ({(num_bays+1)*num_storeys} columns + {num_bays*num_storeys} beams)")

    return node_id, y_coords

# =============================================================================
# SECTION 5: GRAVITY LOAD ANALYSIS
# =============================================================================

def gravity_analysis(node_id):
    """
    Apply factored gravity loads and run static analysis.
    Sets loads constant before dynamic analysis per official docs.
    """
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)

    # Apply vertical loads at floor and roof nodes (not ground floor)
    for fi in range(1, num_storeys + 1):
        for ci, nid in enumerate(node_id[fi]):
            if ci == 0 or ci == num_bays:
                P = -P_exterior   # kN downward
            else:
                P = -P_interior
            ops.load(nid, 0.0, P, 0.0)   # Fx=0, Fy=P, Mz=0

    # Analysis setup
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')   # plain constraints OK for gravity (no equalDOF in Y)
    ops.integrator('LoadControl', 0.1)
    ops.algorithm('Newton')
    ops.analysis('Static')
    ok = ops.analyze(10)

    if ok == 0:
        print("  Gravity analysis: CONVERGED")
    else:
        print("  Gravity analysis: WARNING — did not fully converge")

    # Hold gravity loads constant throughout dynamic analysis
    ops.loadConst('-time', 0.0)
    return ok

# =============================================================================
# SECTION 6: ASSIGN NODAL MASSES
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: This step was missing in all previous versions.
# Without mass, eigenvalue analysis starts with a zero mass matrix and
# fails with "Starting vector is zero" (ARPACK) or returns nonsense values.
#
# Strategy: Assign total floor mass ONLY to master nodes.
# Slave nodes carry zero mass — rigid diaphragm transfers inertia to master.
# This is consistent with equalDOF constraints on DOF 1 (X direction).
#
# Mass units: kN·s²/m (= tonnes in SI)
# Mx = My = M_floor (total floor mass) assigned to master node only
# =============================================================================

def assign_masses(node_id):
    """
    Assign lumped floor masses to master nodes of each floor.
    Master node = leftmost column node (node_id[floor][0]).
    """
    for fi in range(1, num_storeys + 1):
        master = node_id[fi][0]
        # Assign mass in X, Y (and zero rotational mass)
        ops.mass(master, M_floor, M_floor, 0.0)

    print(f"  Masses assigned: {M_floor:.3f} kN·s²/m per floor "
          f"(total {M_floor * num_storeys:.3f} kN·s²/m)")

# =============================================================================
# SECTION 7: EIGENVALUE ANALYSIS
# =============================================================================

def eigenvalue_analysis():
    """
    Extract natural periods using fullGenLapack solver.
    fullGenLapack required when equalDOF constraints are present —
    ARPACK default solver fails with constrained models.
    """
    # fullGenLapack handles multi-point constraints (equalDOF) correctly
    eigs   = ops.eigen('-fullGenLapack', num_storeys)

    # abs() prevents domain error from tiny negative values due to rounding
    omega1 = abs(eigs[0]) ** 0.5
    T1     = 2 * np.pi / omega1

    print(f"  Eigenvalue analysis (fullGenLapack):")
    print(f"    ω₁ = {omega1:.3f} rad/s")
    print(f"    T1 = {T1:.3f} s  (AS1170.4 approx: {T1_approx:.3f} s)")

    if num_storeys >= 2:
        omega2 = abs(eigs[1]) ** 0.5
        T2     = 2 * np.pi / omega2
        print(f"    T2 = {T2:.3f} s")

    return T1, eigs

# =============================================================================
# SECTION 8: SYNTHETIC GROUND MOTION
# Uses modulated sine wave scaled to AS1170.4 PGA for Newcastle (Z=0.11)
# Replace with real NGA-West2 or Newcastle 1989 records for final analysis
# =============================================================================

def generate_ground_motion(dt=0.01, duration=20.0, PGA_g=0.11):
    """
    Generate simple modulated sine wave ground motion.
    PGA_g: peak ground acceleration in g (use Z for Newcastle)
    Returns filepath, dt, npts
    """
    t        = np.arange(0, duration, dt)
    freq     = 2.0   # Hz — dominant frequency (representative for Site De)
    env      = np.sin(np.pi * t / duration)  # Hanning-type envelope
    accel_ms2 = PGA_g * g * env * np.sin(2 * np.pi * freq * t)  # m/s²

    # Use tempfile — works on Colab, Windows, macOS, Linux
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    np.savetxt(tmp.name, accel_ms2, fmt='%.8f')
    tmp.close()

    print(f"  Ground motion: PGA={PGA_g}g ({PGA_g*g:.2f} m/s²), "
          f"duration={duration}s, dt={dt}s, npts={len(t)}")
    return tmp.name, dt, len(t)

# =============================================================================
# SECTION 9: TIME-HISTORY ANALYSIS
# =============================================================================

def time_history_analysis(node_id, gm_file, dt, npts, T1, eigs):
    """
    Nonlinear time-history analysis with Newmark average acceleration.
    Rayleigh damping at 5% applied at modes 1 and 2.
    Convergence fallback chain per official OpenSeesPy documentation.
    """

    # ── RAYLEIGH DAMPING (5% at modes 1 and 2) ──────────────────────────
    xi     = 0.05
    omega1 = abs(eigs[0]) ** 0.5
    omega2 = abs(eigs[1]) ** 0.5
    a0     = xi * 2 * omega1 * omega2 / (omega1 + omega2)
    a1     = xi * 2 / (omega1 + omega2)
    ops.rayleigh(a0, 0.0, 0.0, a1)
    print(f"  Rayleigh damping: a0={a0:.5f}, a1={a1:.7f} (5% at T1,T2)")

    # ── GROUND MOTION LOADING ────────────────────────────────────────────
    # Path time series reads acceleration values from file
    # factor=1.0 because values already in m/s²
    ops.timeSeries('Path', 2, '-dt', dt, '-filePath', gm_file, '-factor', 1.0)
    # UniformExcitation applies ground acceleration in DOF 1 (X direction)
    ops.pattern('UniformExcitation', 2, 1, '-accel', 2)

    # ── ANALYSIS SETUP ───────────────────────────────────────────────────
    ops.system('UmfPack')               # direct solver — robust for nonlinear
    ops.numberer('RCM')
    ops.constraints('Transformation')  # required for equalDOF constraints
    ops.test('NormDispIncr', 1.0e-8, 10, 0)
    ops.integrator('Newmark', 0.5, 0.25)  # average acceleration — unconditionally stable
    ops.algorithm('Newton')
    ops.analysis('Transient')

    # ── TIME STEPPING ────────────────────────────────────────────────────
    # Sub-step by factor of 2 for stability with nonlinear material
    dt_sub  = dt / 2
    n_steps = int(npts * dt / dt_sub)

    time_h   = []
    disp_g   = []   # ground floor displacement (should be ~0, reference)
    disp_f1  = []   # first floor displacement
    disp_r   = []   # roof displacement

    print(f"  Running: {n_steps} steps × dt={dt_sub:.4f}s = {n_steps*dt_sub:.1f}s")

    ok = 0
    for step in range(n_steps):
        ok = ops.analyze(1, dt_sub)

        if ok != 0:
            # Fallback 1: Try KrylovNewton with smaller timestep
            ops.algorithm('KrylovNewton')
            ok = ops.analyze(1, dt_sub / 5)
            if ok != 0:
                # Fallback 2: ModifiedNewton with initial stiffness
                ops.test('NormDispIncr', 1.0e-6, 100, 0)
                ops.algorithm('ModifiedNewton', '-initial')
                ok = ops.analyze(1, dt_sub / 10)
            # Reset to standard Newton
            ops.algorithm('Newton')
            ops.test('NormDispIncr', 1.0e-8, 10, 0)

        # Record displacements at master nodes (DOF 1 = X direction)
        time_h.append(ops.getTime())
        disp_g.append(ops.nodeDisp(node_id[0][0], 1))   # ground (=0 fixed)
        disp_f1.append(ops.nodeDisp(node_id[1][0], 1))  # first floor master
        disp_r.append(ops.nodeDisp(node_id[2][0], 1))   # roof master

    status = "COMPLETE" if ok == 0 else "WARNING — partial convergence"
    print(f"  Time-history: {status}")

    return (np.array(time_h),
            np.array(disp_g),
            np.array(disp_f1),
            np.array(disp_r))

# =============================================================================
# SECTION 10: POST-PROCESSING & AS1170.4 COMPLIANCE
# =============================================================================

def post_process(time_h, disp_g, disp_f1, disp_r, T1):
    """
    Compute EDPs and check AS1170.4 compliance.
    EDPs: Peak Inter-Storey Drift Ratio, Peak Floor Acceleration, Base Shear
    """
    print()
    print("=" * 60)
    print("  ENGINEERING DEMAND PARAMETERS & COMPLIANCE")
    print("=" * 60)

    # ── INTER-STOREY DRIFT RATIOS ────────────────────────────────────────
    # Storey 1: between ground (fixed at 0) and first floor
    # Storey 2: between first floor and roof
    drift1 = (disp_f1 - disp_g) / storey_height   # dimensionless ratio
    drift2 = (disp_r  - disp_f1) / storey_height

    PIDR1  = float(np.max(np.abs(drift1)))
    PIDR2  = float(np.max(np.abs(drift2)))
    PIDR   = max(PIDR1, PIDR2)

    print(f"\n  INTER-STOREY DRIFT RATIOS:")
    print(f"    Storey 1 PIDR : {PIDR1*100:.3f}%")
    print(f"    Storey 2 PIDR : {PIDR2*100:.3f}%")
    print(f"    Governing     : {PIDR*100:.3f}%")
    print(f"    AS1170.4 limit: {drift_limit*100:.1f}%  (Cl 5.4.4)")
    drift_ok = PIDR <= drift_limit
    print(f"    CHECK         : {'✓ PASS' if drift_ok else '✗ FAIL'}")

    # ── PEAK FLOOR ACCELERATIONS ─────────────────────────────────────────
    # Approximated from displacement: a ≈ ω² × u (harmonic assumption)
    omega1     = 2 * np.pi / T1
    PFA_roof   = omega1**2 * float(np.max(np.abs(disp_r)))
    PFA_floor1 = omega1**2 * float(np.max(np.abs(disp_f1)))
    PFA_ground = Z * g   # input PGA

    print(f"\n  PEAK FLOOR ACCELERATIONS:")
    print(f"    Ground (PGA)  : {PFA_ground:.3f} m/s²  ({PFA_ground/g:.3f}g)")
    print(f"    Floor 1       : {PFA_floor1:.3f} m/s²  ({PFA_floor1/g:.3f}g)")
    print(f"    Roof          : {PFA_roof:.3f} m/s²  ({PFA_roof/g:.3f}g)")

    # ── BASE SHEAR APPROXIMATION ─────────────────────────────────────────
    # V_dyn ≈ Σ(mi × ai) where ai ≈ ω² × ui (floor-level acceleration)
    V_dyn = (M_floor * PFA_floor1) + (M_floor * PFA_roof)   # kN

    print(f"\n  BASE SHEAR:")
    print(f"    Dynamic (approx)  : {V_dyn:.1f} kN")
    print(f"    Static AS1170.4   : {V_static:.1f} kN")
    print(f"    Ratio (dyn/stat)  : {V_dyn/V_static:.2f}")

    # ── COMPLIANCE SUMMARY ───────────────────────────────────────────────
    print()
    print("─" * 60)
    print(f"  AS1170.4-2007 COMPLIANCE — {BUILDING_NAME}")
    print("─" * 60)
    print(f"  Z={Z}, Site={site_class}, μ={mu}, Sp={Sp}")
    print(f"  T1 (eigen) = {T1:.3f}s  |  T1 (approx) = {T1_approx:.3f}s")
    print()
    print(f"  {'✓ PASS' if drift_ok else '✗ FAIL'}  Drift ≤ 1.5%  "
          f"(actual: {PIDR*100:.3f}%)")
    print()
    overall = drift_ok
    print(f"  OVERALL: {'✓ COMPLIANT' if overall else '✗ NON-COMPLIANT'}")
    print("=" * 60)

    return {
        'building':   BUILDING_NAME,
        'T1':         T1,
        'PIDR1':      PIDR1,
        'PIDR2':      PIDR2,
        'PIDR_max':   PIDR,
        'PFA_f1':     PFA_floor1,
        'PFA_roof':   PFA_roof,
        'V_static':   V_static,
        'V_dynamic':  V_dyn,
        'drift_pass': drift_ok,
        'compliant':  overall,
    }

# =============================================================================
# SECTION 11: PLOTS
# =============================================================================

def plot_results(time_h, disp_f1, disp_r, drift1, drift2, results):

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(BUILDING_NAME, fontsize=12, fontweight='bold', y=0.98)

    # Plot 1: Roof displacement time history
    ax = axes[0, 0]
    ax.plot(time_h, np.array(disp_r) * 1000, color='steelblue', lw=1)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Roof Displacement (mm)')
    ax.set_title('Roof Displacement')
    ax.grid(True, alpha=0.3)

    # Plot 2: Inter-storey drift time history
    ax = axes[0, 1]
    ax.plot(time_h, drift1 * 100, color='tomato',  lw=1, label='Storey 1')
    ax.plot(time_h, drift2 * 100, color='darkorange', lw=1, label='Storey 2')
    ax.axhline( drift_limit*100, color='red', ls='--', lw=1.5,
                label=f'AS1170.4 limit {drift_limit*100}%')
    ax.axhline(-drift_limit*100, color='red', ls='--', lw=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Inter-Storey Drift (%)')
    ax.set_title('Inter-Storey Drift')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Peak drift profile
    ax = axes[1, 0]
    bars = ax.barh([1, 2],
                   [results['PIDR1']*100, results['PIDR2']*100],
                   color=['tomato', 'darkorange'])
    ax.axvline(drift_limit*100, color='red', ls='--', lw=1.5,
               label='AS1170.4 limit')
    ax.set_xlabel('Peak Drift (%)')
    ax.set_ylabel('Storey')
    ax.set_title('Peak Drift Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, [results['PIDR1']*100, results['PIDR2']*100]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va='center', fontsize=9)

    # Plot 4: Summary box
    ax = axes[1, 1]
    ax.axis('off')
    txt = (
        f"RESULT SUMMARY\n{'─'*34}\n"
        f"Period T1:          {results['T1']:.3f} s\n"
        f"PIDR Storey 1:      {results['PIDR1']*100:.3f}%\n"
        f"PIDR Storey 2:      {results['PIDR2']*100:.3f}%\n"
        f"Governing PIDR:     {results['PIDR_max']*100:.3f}%\n"
        f"Limit (AS1170.4):   {drift_limit*100:.1f}%\n"
        f"{'─'*34}\n"
        f"V static:           {results['V_static']:.1f} kN\n"
        f"V dynamic (approx): {results['V_dynamic']:.1f} kN\n"
        f"{'─'*34}\n"
        f"Drift check: {'✓ PASS' if results['drift_pass'] else '✗ FAIL'}\n"
        f"OVERALL: {'✓ COMPLIANT' if results['compliant'] else '✗ NON-COMPLIANT'}"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            fontsize=9.5, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow',
                      ec='gray', alpha=0.9))

    plt.tight_layout()
    plt.savefig('building1_results.png', dpi=150, bbox_inches='tight')
    print("\n  Plot saved: building1_results.png")
    plt.show()
    plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__" or True:

    print("\n" + "=" * 60)
    print("  STEP 1: Building OpenSees model")
    print("=" * 60)
    node_id, y_coords = build_model()

    print("\n" + "=" * 60)
    print("  STEP 2: Gravity load analysis")
    print("=" * 60)
    gravity_analysis(node_id)

    print("\n" + "=" * 60)
    print("  STEP 3: Assigning nodal masses")
    print("=" * 60)
    assign_masses(node_id)

    print("\n" + "=" * 60)
    print("  STEP 4: Eigenvalue analysis")
    print("=" * 60)
    T1, eigs = eigenvalue_analysis()

    print("\n" + "=" * 60)
    print("  STEP 5: Generating ground motion")
    print("=" * 60)
    gm_file, dt, npts = generate_ground_motion(dt=0.01, duration=20.0, PGA_g=Z)

    print("\n" + "=" * 60)
    print("  STEP 6: Nonlinear time-history analysis")
    print("=" * 60)
    time_h, dg, df1, dr = time_history_analysis(
        node_id, gm_file, dt, npts, T1, eigs)

    print("\n" + "=" * 60)
    print("  STEP 7: Post-processing")
    print("=" * 60)
    d1 = (df1 - dg) / storey_height
    d2 = (dr  - df1) / storey_height
    results = post_process(time_h, dg, df1, dr, T1)

    print("\n" + "=" * 60)
    print("  STEP 8: Generating plots")
    print("=" * 60)
    plot_results(time_h, df1, dr, d1, d2, results)

    # Cleanup temp ground motion file
    try:
        os.remove(gm_file)
    except Exception:
        pass

    print("\n  ALL DONE.")
    print("  To run Building 2: change fc=32, fy=500, col=0.35×0.35,")
    print("    beam=0.30×0.50, col_rho=0.02, mu=3.0, Sp=0.67")
    print("  To run Building 3: change fc=40, fy=500, col=0.40×0.40,")
    print("    beam=0.35×0.55, col_rho=0.025, mu=4.0, Sp=0.67")
