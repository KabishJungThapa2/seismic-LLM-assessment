# =============================================================================
# SEISMIC VULNERABILITY ASSESSMENT — N-STOREY GENERALISED PIPELINE
# LLM-Orchestrated Workflow | UTS Engineering Graduate Project PG (42003)
# Kabish Jung Thapa (25631413) | Supervisor: Prof. Jianchun Li
#
# HOW TO RUN IN GOOGLE COLAB:
#   Paste this entire script into ONE cell → Shift+Enter
#   No API key needed. Works for 1 to 20+ storeys.
#
# WHAT THIS FIXES OVER THE 2-STOREY VERSION:
#   1. Node IDs: (fi+1)*1000 + (ci+1) — unique for up to 999 floors/bays
#   2. Displacement recording: ALL floors recorded in a list, not just 2
#   3. PIDR: computed for every storey, not just S1 and S2
#   4. PFA: computed at every floor level
#   5. AS1170.4 lateral force: proper floor-by-floor distribution
#   6. Pushover: inverted triangle over all N floors
#   7. Eigenvalue: requests min(N,3) modes; Rayleigh handles N=1 case
#   8. Plots: fully dynamic — adapt to any number of storeys
#   9. JSON report: storey-by-storey EDP table
#
# TESTED CONFIGURATIONS:
#   1 storey  (single storey bungalow)
#   2 storeys (standard residential)
#   4 storeys (low-rise apartment)
#   8 storeys (medium-rise)
# =============================================================================

# ── INSTALL ───────────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run(['pip','install','openseespy','numpy','matplotlib','scipy','-q'],
               check=True)
for mod in list(sys.modules.keys()):
    if 'opensees' in mod: del sys.modules[mod]

import openseespy.opensees as ops
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tempfile, os, json, re, time, warnings
from datetime import datetime
warnings.filterwarnings('ignore')
print("Packages ready.\n")

# =============================================================================
# SECTION 1 — CONFIGURATION
# Edit the BUILDING dict below to define your building.
# =============================================================================

# ─── CHANGE THESE VALUES ──────────────────────────────────────────────────────
BUILDING = dict(
    name          = "Pre-1990 RC Frame — Newcastle",
    era           = "pre-1990",

    # Geometry — change num_storeys to any integer 1–20
    num_storeys   = 4,       # ← CHANGE THIS: 1, 2, 3, 4, 6, 8, 10, ...
    storey_height = 3.0,     # m — height of each storey (can be a list for variable heights)
    num_bays      = 3,       # bays in earthquake direction
    bay_width     = 4.0,     # m — width of each bay (uniform)
    floor_width   = 8.0,     # m — building dimension perpendicular to frame

    # Materials
    fc  = 20.0,   # MPa — concrete compressive strength
    fy  = 250.0,  # MPa — steel yield strength
    Es  = 200000.0,

    # Member sizes (m)
    col_b  = 0.30, col_h  = 0.30,   # column width × depth
    beam_b = 0.30, beam_h = 0.45,   # beam width × depth

    # Reinforcement ratios
    col_rho    = 0.015,   # longitudinal steel ratio
    beam_rho_t = 0.008,   # tension steel
    beam_rho_c = 0.004,   # compression steel

    # Seismic parameters — AS1170.4:2007
    Z          = 0.11,    # hazard factor (Newcastle)
    site_class = "De",    # soft soil
    mu         = 2.0,     # ductility factor
    Sp         = 0.77,    # structural performance factor

    # Loads (kPa)
    dead_load  = 5.0,
    live_load  = 2.0,

    # Concrete confinement (Concrete01 parameters)
    epsc0_core = -0.004,  # strain at peak — poor confinement pre-1990
    epsU_core  = -0.012,  # ultimate strain
)

# OPTIONAL: per-storey column/beam sizes for setback buildings
# Leave empty {} to use uniform sizes defined above.
# Example for a 4-storey with larger lower columns:
#   PER_STOREY = {1: dict(col_b=0.40,col_h=0.40), 2: dict(col_b=0.35,col_h=0.35)}
PER_STOREY = {}   # {} = uniform, all storeys use values from BUILDING above

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
G            = 9.81
COVER        = 0.040
DRIFT_LIMIT  = 0.015

DAMAGE_STATES = [
    ("None",      0.000, 0.005, 0.00),
    ("Slight",    0.005, 0.010, 0.05),
    ("Moderate",  0.010, 0.020, 0.20),
    ("Extensive", 0.020, 0.040, 0.50),
    ("Complete",  0.040, 9.999, 1.00),
]

STATE_COLOURS = {
    "None":"#2ecc71","Slight":"#f1c40f",
    "Moderate":"#e67e22","Extensive":"#e74c3c","Complete":"#8e44ad"
}

# =============================================================================
# SECTION 2 — DERIVED PROPERTIES
# (All computed automatically — do not edit)
# =============================================================================

def derive(b):
    """Compute all derived properties from the BUILDING dict."""
    n  = b['num_storeys']
    Ec = 0.043 * (2400**1.5) * (b['fc']**0.5)

    # Support variable storey heights — if scalar, broadcast to list
    if isinstance(b['storey_height'], (int, float)):
        heights = [b['storey_height']] * n
    else:
        heights = list(b['storey_height'])
        assert len(heights) == n, "len(storey_height list) must equal num_storeys"

    Hn = sum(heights)  # total building height

    # Floor-level cumulative heights (0 = ground)
    y_floors = [0.0]
    for h in heights:
        y_floors.append(y_floors[-1] + h)

    # Seismic weight per floor (AS1170.4: G + 0.3Q)
    floor_area = b['num_bays'] * b['bay_width'] * b['floor_width']
    W_floor    = (b['dead_load'] + 0.3 * b['live_load']) * floor_area
    W_total    = W_floor * n
    M_floor    = W_floor / G        # mass per floor (kN.s2/m = tonne)

    # Gravity loads at nodes
    w_frame  = (b['dead_load'] + b['live_load']) * b['floor_width'] / 2
    P_int    = w_frame * b['bay_width']
    P_ext    = P_int / 2

    # Per-storey section properties (allows setback buildings)
    storey_props = []
    for si in range(1, n + 1):
        over = PER_STOREY.get(si, {})
        col_b  = over.get('col_b',  b['col_b'])
        col_h  = over.get('col_h',  b['col_h'])
        beam_b = over.get('beam_b', b['beam_b'])
        beam_h = over.get('beam_h', b['beam_h'])
        storey_props.append(dict(
            col_b=col_b, col_h=col_h, beam_b=beam_b, beam_h=beam_h,
            Asc  = b['col_rho']    * col_b  * col_h,
            Ast  = b['beam_rho_t'] * beam_b * beam_h,
            Asc2 = b['beam_rho_c'] * beam_b * beam_h,
        ))

    return dict(
        n=n, Ec=Ec, heights=heights, y_floors=y_floors, Hn=Hn,
        floor_area=floor_area, W_floor=W_floor, W_total=W_total,
        M_floor=M_floor, P_int=P_int, P_ext=P_ext,
        storey_props=storey_props,
        fc_kN=b['fc']*1000, fy_kN=b['fy']*1000, Es_kN=b['Es']*1000,
    )

# =============================================================================
# SECTION 3 — AS1170.4 STATIC ANALYSIS
# =============================================================================

def static_analysis(b, d):
    """
    AS1170.4:2007 Cl 6.2 equivalent static base shear.
    Lateral force distribution: Cl 6.3 — proportional to Wi * hi.
    Returns V_static, Ch, T1_code, floor_forces[1..n] (kN).
    """
    T1 = 0.075 * d['Hn'] ** 0.75   # Appendix B — RC frames

    if   T1 <= 0.10: Ch = 2.35
    elif T1 <  1.50: Ch = 1.65 * (0.1/T1)**0.85
    else:            Ch = 1.10 * (1.5/T1)**2.0

    V = max((b['Z']/b['mu']) * b['Sp'] * Ch * d['W_total'], 0.01*d['W_total'])

    # Floor-level lateral forces: Fi = V * (Wi*hi) / sum(Wj*hj)
    # AS1170.4 Cl 6.3 — equal floor masses so Wi cancels
    hi_sum = sum(d['y_floors'][fi] for fi in range(1, d['n']+1))
    floor_forces = {}
    for fi in range(1, d['n']+1):
        floor_forces[fi] = V * d['y_floors'][fi] / hi_sum if hi_sum > 0 else V/d['n']

    return dict(V_static=V, Ch=Ch, T1_code=T1, floor_forces=floor_forces)

# =============================================================================
# SECTION 4 — OPENSEESPY MODEL BUILDER
# =============================================================================

def build_model(b, d):
    """
    Build 2D nonlinear RC plane frame for N storeys.

    NODE NUMBERING — critical fix for N > 9 storeys:
        nid = (floor_idx + 1) * 1000 + (col_idx + 1)
        e.g. floor 0 col 0 → 1001,  floor 10 col 3 → 11004
        Works for up to 999 floors × 999 bays (more than enough).
        Previous scheme (fi+1)*10 + (ci+1) broke at floor 10+.

    MASS ASSIGNMENT — master nodes only:
        Slave nodes in equalDOF get zero mass.
        Assign M_floor ONLY to node_id[fi][0] for fi >= 1.
        This is required for fullGenLapack eigenvalue to work.
    """
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # ── Nodes ─────────────────────────────────────────────────────────────
    x_coords = [j * b['bay_width'] for j in range(b['num_bays'] + 1)]

    node_id = []
    for fi, y in enumerate(d['y_floors']):
        row = []
        for ci, x in enumerate(x_coords):
            nid = (fi + 1) * 1000 + (ci + 1)   # ← GENERALISED (was *10)
            ops.node(nid, x, y)
            row.append(nid)
        node_id.append(row)

    # ── Boundary conditions ────────────────────────────────────────────────
    for nid in node_id[0]:
        ops.fix(nid, 1, 1, 1)

    # ── Rigid diaphragm — DOF 1 (X) at each floor ─────────────────────────
    for fi in range(1, d['n'] + 1):
        master = node_id[fi][0]
        for slave in node_id[fi][1:]:
            ops.equalDOF(master, slave, 1)

    # ── Materials (shared for all storeys) ────────────────────────────────
    ops.uniaxialMaterial('Concrete01', 1,
        -d['fc_kN'], b['epsc0_core'], -0.2*d['fc_kN'], b['epsU_core'])
    ops.uniaxialMaterial('Concrete01', 2,
        -d['fc_kN'], -0.002, 0.0, -0.004)
    ops.uniaxialMaterial('Steel01', 3,
        d['fy_kN'], d['Es_kN'], 0.01)

    # ── Fibre sections — one per storey (allows setback/taper) ────────────
    # Section tag = storey index (1-based)
    # Column sections: tags  1  to  n
    # Beam sections:   tags n+1 to 2n
    for si, sp in enumerate(d['storey_props'], 1):
        col_b, col_h = sp['col_b'], sp['col_h']
        cy = col_h/2 - COVER;  cz = col_b/2 - COVER
        As_bar = max(sp['Asc']/6, 1e-5)

        ops.section('Fiber', si)           # column section tag = si
        ops.patch('rect', 1, 10, 10, -cy, -cz,  cy,  cz)
        ops.patch('rect', 2, 10,  2,  cy, -col_b/2, col_h/2, col_b/2)
        ops.patch('rect', 2, 10,  2, -col_h/2, -col_b/2, -cy, col_b/2)
        ops.patch('rect', 2,  2, 10, -cy, -col_b/2, cy, -cz)
        ops.patch('rect', 2,  2, 10, -cy,  cz,      cy,  col_b/2)
        ops.layer('straight', 3, 3, As_bar, -cy, -cz, -cy, cz)
        ops.layer('straight', 3, 3, As_bar,  cy, -cz,  cy, cz)

        beam_b, beam_h = sp['beam_b'], sp['beam_h']
        by = beam_h/2 - COVER;  bz = beam_b/2 - COVER
        beam_tag = d['n'] + si  # beam section tag = n+si

        ops.section('Fiber', beam_tag)
        ops.patch('rect', 1, 10, 10, -by, -bz, by, bz)
        ops.patch('rect', 2, 10,  2,  by, -beam_b/2, beam_h/2, beam_b/2)
        ops.patch('rect', 2, 10,  2, -beam_h/2, -beam_b/2, -by, beam_b/2)
        ops.layer('straight', 3, 3, sp['Ast']/3,  -by, -bz, -by, bz)
        ops.layer('straight', 3, 3, sp['Asc2']/3,  by, -bz,  by, bz)

    # ── Geometric transformations ──────────────────────────────────────────
    # One PDelta transform per storey (tag = si), one Linear per storey (tag = n+si)
    for si in range(1, d['n']+1):
        ops.geomTransf('PDelta', si)          # for columns
        ops.geomTransf('Linear', d['n']+si)  # for beams

    # ── Elements ──────────────────────────────────────────────────────────
    eid = 10000
    for fi in range(d['n']):            # storey fi connects floor fi to fi+1
        si = fi + 1                     # storey index (1-based)
        for ci in range(b['num_bays'] + 1):
            ops.element('nonlinearBeamColumn', eid,
                        node_id[fi][ci], node_id[fi+1][ci],
                        5, si, si)      # col section and transform = si
            eid += 1
    for fi in range(1, d['n'] + 1):    # beams at each floor above ground
        si   = fi
        beam_tag   = d['n'] + si
        for ci in range(b['num_bays']):
            ops.element('nonlinearBeamColumn', eid,
                        node_id[fi][ci], node_id[fi][ci+1],
                        5, beam_tag, beam_tag)
            eid += 1

    n_col  = (b['num_bays']+1) * d['n']
    n_beam = b['num_bays']     * d['n']
    print("  Model: {} floors, {} nodes, {} col + {} beam elements".format(
          d['n'], len(d['y_floors'])*(b['num_bays']+1), n_col, n_beam))

    # ── Gravity analysis ──────────────────────────────────────────────────
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)
    for fi in range(1, d['n'] + 1):
        for ci, nid in enumerate(node_id[fi]):
            P = -d['P_ext'] if (ci == 0 or ci == b['num_bays']) else -d['P_int']
            ops.load(nid, 0.0, P, 0.0)
    ops.system('BandGeneral');  ops.numberer('RCM')
    ops.constraints('Plain');   ops.integrator('LoadControl', 0.1)
    ops.algorithm('Newton');    ops.analysis('Static')
    ok = ops.analyze(10)
    print("  Gravity: {}".format("CONVERGED" if ok==0 else "WARNING"))
    ops.loadConst('-time', 0.0)

    # ── Mass assignment — master nodes ONLY ───────────────────────────────
    # CRITICAL: slave nodes must get zero mass with equalDOF
    # fullGenLapack solver handles this; ARPACK fails
    for fi in range(1, d['n'] + 1):
        ops.mass(node_id[fi][0], d['M_floor'], d['M_floor'], 0.0)
    print("  Mass: {:.3f} kN.s2/m per floor x {} floors = {:.1f} kN.s2/m total".format(
          d['M_floor'], d['n'], d['M_floor']*d['n']))

    return node_id

# =============================================================================
# SECTION 5 — EIGENVALUE ANALYSIS
# =============================================================================

def eigenvalue_analysis(b, d):
    """
    fullGenLapack required — ARPACK fails with equalDOF + master-node mass.
    Request min(n, 3) modes — enough for Rayleigh damping and reporting.
    """
    n_modes = min(d['n'], 3)
    eigs    = ops.eigen('-fullGenLapack', n_modes)

    omega  = [abs(eigs[i])**0.5 for i in range(n_modes)]
    T      = [2*np.pi/w for w in omega]

    print("  Eigenvalue (fullGenLapack, {} modes):".format(n_modes))
    for i in range(n_modes):
        print("    Mode {}: T = {:.3f} s  (omega = {:.3f} rad/s)".format(
              i+1, T[i], omega[i]))

    return dict(omega=omega, T=T, eigs=eigs, n_modes=n_modes)

# =============================================================================
# SECTION 6 — PUSHOVER ANALYSIS
# =============================================================================

def run_pushover(b, d, node_id, sa):
    """
    Displacement-controlled pushover to 3% roof drift.
    Load pattern: AS1170.4 Cl 6.3 floor-level distribution
    (proportional to Wi * hi = proportional to hi for equal floor masses).
    """
    target  = 0.03 * d['Hn']
    n_steps = 300

    # Lateral load proportional to floor height (AS1170.4 Cl 6.3)
    hi_sum = sum(d['y_floors'][fi] for fi in range(1, d['n']+1))
    fracs  = [d['y_floors'][fi]/hi_sum for fi in range(1, d['n']+1)]

    ops.timeSeries('Linear', 10)
    ops.pattern('Plain', 10, 10)
    for fi in range(1, d['n']+1):
        ops.load(node_id[fi][0], fracs[fi-1], 0.0, 0.0)

    ops.system('UmfPack');  ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.test('NormDispIncr', 1e-6, 150, 0)
    ops.algorithm('Newton')
    ops.integrator('DisplacementControl',
                   node_id[-1][0], 1, target/n_steps)
    ops.analysis('Static')

    roof_disp = [0.0];  base_shear = [0.0]
    n_fail    = 0

    for _ in range(n_steps):
        ok = ops.analyze(1)
        if ok != 0:
            n_fail += 1
            ops.algorithm('KrylovNewton')
            ok = ops.analyze(1)
            ops.algorithm('Newton')
            if ok != 0: break

        rd = ops.nodeDisp(node_id[-1][0], 1)
        V  = sum(-ops.nodeReaction(nid, 1) for nid in node_id[0])
        roof_disp.append(rd)
        base_shear.append(V)

    rd = np.array(roof_disp)
    bs = np.array(base_shear)
    V_max    = float(np.max(bs)) if len(bs) > 1 else sa['V_static']
    d_yield  = next((rd[i] for i,v in enumerate(bs) if v >= 0.60*V_max), None)
    ductility = float(rd[-1]/d_yield) if (d_yield and d_yield > 0) else None
    T_eff    = 2*np.pi*np.sqrt(d['W_total']/(G*V_max/rd[-1])) if rd[-1]>0 else None

    print("  Pushover: V_max={:.1f}kN  ductility={}  steps={}  fail={}".format(
          V_max,
          "{:.2f}".format(ductility) if ductility else "N/A",
          len(rd)-1, n_fail))

    return dict(
        roof_disp=rd, base_shear=bs,
        roof_drift=rd/d['Hn']*100,
        V_max=V_max, d_yield=d_yield,
        ductility=ductility, T_eff=T_eff, n_fail=n_fail,
    )

# =============================================================================
# SECTION 7 — TIME-HISTORY ANALYSIS
# =============================================================================

def run_time_history(b, d, node_id, ev):
    """
    Nonlinear transient analysis.
    Records displacement at EVERY floor master node.
    Returns arrays: time_h, disp_all[floor 0..n], drift_all[storey 1..n]
    """
    omega = ev['omega']
    xi = 0.05
    # Rayleigh damping between modes 1 and 2 (or mode 1 and 3x if N=1)
    w1 = omega[0]
    w2 = omega[1] if len(omega) >= 2 else omega[0] * 3.0
    a0 = xi * 2*w1*w2 / (w1+w2)
    a1 = xi * 2 / (w1+w2)
    ops.rayleigh(a0, 0.0, 0.0, a1)

    # Synthetic ground motion — tuned to fundamental period
    T1   = ev['T'][0]
    dt   = 0.01; dur = 20.0
    t    = np.arange(0, dur, dt)
    freq = min(1.0/T1, 4.0)
    env  = np.sin(np.pi*t/dur)
    acc  = b['Z'] * G * env * np.sin(2*np.pi*freq*t)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    np.savetxt(tmp.name, acc, fmt='%.8f'); tmp.close()

    ops.timeSeries('Path', 2, '-dt', dt, '-filePath', tmp.name, '-factor', 1.0)
    ops.pattern('UniformExcitation', 2, 1, '-accel', 2)
    ops.system('UmfPack');  ops.numberer('RCM')
    ops.constraints('Transformation')   # REQUIRED with equalDOF
    ops.test('NormDispIncr', 1e-8, 10, 0)
    ops.integrator('Newmark', 0.5, 0.25)
    ops.algorithm('Newton')
    ops.analysis('Transient')

    dt_sub  = dt/2
    n_steps = int(len(t)*dt/dt_sub)
    n_fail  = 0; n_krylov = 0; n_modified = 0

    # Pre-allocate recording lists for ALL floors (index 0 = ground)
    time_h   = []
    disp_all = [[] for _ in range(d['n'] + 1)]   # disp_all[fi] for fi=0..n

    print("  Time-history: {} steps, dt={:.4f}s...".format(n_steps, dt_sub))

    for _ in range(n_steps):
        ok = ops.analyze(1, dt_sub)
        if ok != 0:
            n_fail += 1
            ops.algorithm('KrylovNewton'); n_krylov += 1
            ok = ops.analyze(1, dt_sub/5)
            if ok != 0:
                ops.test('NormDispIncr', 1e-6, 100, 0)
                ops.algorithm('ModifiedNewton','-initial'); n_modified += 1
                ops.analyze(1, dt_sub/10)
            ops.algorithm('Newton')
            ops.test('NormDispIncr', 1e-8, 10, 0)

        time_h.append(ops.getTime())
        for fi in range(d['n'] + 1):
            disp_all[fi].append(ops.nodeDisp(node_id[fi][0], 1))

    try: os.remove(tmp.name)
    except: pass

    th     = np.array(time_h)
    d_all  = [np.array(disp_all[fi]) for fi in range(d['n']+1)]

    # Compute inter-storey drifts for ALL storeys
    drift_all = []
    for fi in range(1, d['n']+1):
        h_storey = d['heights'][fi-1]
        drift_all.append((d_all[fi] - d_all[fi-1]) / h_storey)

    conv_rate = (1 - n_fail/n_steps)*100
    print("  TH done: conv={:.1f}%  fail={}  krylov={}  modified={}".format(
          conv_rate, n_fail, n_krylov, n_modified))

    return dict(
        time_h    = th,
        disp_all  = d_all,      # list of n+1 arrays (floor 0 to n)
        drift_all = drift_all,  # list of n arrays (storey 1 to n)
        conv_rate = conv_rate,
        n_fail    = n_fail,
    )

# =============================================================================
# SECTION 8 — POST-PROCESSING (EDPs)
# =============================================================================

def classify_damage(pidr):
    for name,lo,hi,_ in DAMAGE_STATES:
        if lo <= pidr < hi: return name
    return "Complete"


def compute_edps(b, d, th, sa, ev):
    """
    Compute Engineering Demand Parameters for ALL storeys.
    Returns per-storey PIDR and PFA arrays, governing values, damage state.
    """
    T1     = ev['T'][0]
    omega1 = ev['omega'][0]

    # Per-storey PIDR
    PIDR_storey = [float(np.max(np.abs(th['drift_all'][fi-1])))
                   for fi in range(1, d['n']+1)]
    PIDR_max    = max(PIDR_storey)
    gov_storey  = PIDR_storey.index(PIDR_max) + 1  # 1-based

    # Per-floor PFA (omega^2 * u approximation)
    PFA_storey = [b['Z']*G]  # ground = PGA
    for fi in range(1, d['n']+1):
        PFA = omega1**2 * float(np.max(np.abs(th['disp_all'][fi])))
        PFA_storey.append(PFA)

    # Dynamic base shear
    V_dyn = sum(d['M_floor'] * PFA_storey[fi] for fi in range(1, d['n']+1))

    # Spectral displacement (roof mode 1 proxy)
    Sd = float(np.max(np.abs(th['disp_all'][d['n']])))

    damage_state = classify_damage(PIDR_max)

    return dict(
        PIDR_storey  = PIDR_storey,       # list [s1, s2, ..., sn]
        PIDR_max     = PIDR_max,
        gov_storey   = gov_storey,
        PFA_storey   = PFA_storey,        # list [ground, fl1, fl2, ..., roof]
        amp_roof     = PFA_storey[-1] / PFA_storey[0],
        V_static     = sa['V_static'],
        V_dynamic    = V_dyn,
        W_total      = d['W_total'],
        Sd_m         = Sd,
        max_roof_mm  = Sd*1000,
        damage_state = damage_state,
        drift_pass   = PIDR_max <= DRIFT_LIMIT,
        compliant    = PIDR_max <= DRIFT_LIMIT,
    )

# =============================================================================
# SECTION 9 — PLOTS (fully dynamic — adapts to any N)
# =============================================================================

def plot_results(b, d, th, sa, ev, push, edp):
    """
    Generates two figures:
    Figure 1: Time-history results — adapts to N storeys
    Figure 2: Storey profile charts — drift profile, PFA profile
    """
    n     = d['n']
    col   = '#d9534f' if b['era']=='pre-1990' else \
            '#f0ad4e' if b['era']=='post-1990' else '#5cb85c'
    lim   = DRIFT_LIMIT * 100
    title = b['name']

    # ── Figure 1: Time-history overview ──────────────────────────────────
    # Rows: always 3 (roof disp, max drift vs time, pushover)
    # Plus one row per every 4 storeys for individual drift traces
    n_drift_rows = max(1, (n + 3)//4)
    fig1, axes = plt.subplots(3 + n_drift_rows, 1,
                              figsize=(13, 4*(3+n_drift_rows)))
    fig1.suptitle(title, fontsize=13, fontweight='bold')

    th_t  = th['time_h']
    disp_r = th['disp_all'][n]

    # Panel 1 — Roof displacement
    ax = axes[0]
    ax.plot(th_t, disp_r*1000, color='steelblue', lw=1)
    ax.fill_between(th_t, disp_r*1000, 0, alpha=0.12, color='steelblue')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('Roof Disp. (mm)')
    ax.set_title('Roof Displacement Time History')
    ax.text(0.99, 0.95, 'max={:.1f}mm'.format(edp['max_roof_mm']),
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(fc='white',alpha=0.7,ec='none'))
    ax.grid(True, alpha=0.3)

    # Panel 2 — Governing storey drift
    gov = edp['gov_storey'] - 1
    ax = axes[1]
    ax.plot(th_t, th['drift_all'][gov]*100, color=col, lw=1,
            label='Storey {} (governing)'.format(edp['gov_storey']))
    ax.axhline( lim, color='red', ls='--', lw=1.5, label='1.5% limit')
    ax.axhline(-lim, color='red', ls='--', lw=1.5)
    ax.set_ylabel('Drift (%)')
    ax.set_title('Governing Storey Drift (Storey {})'.format(edp['gov_storey']))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 3 — Pushover
    ax = axes[2]
    ax.plot(push['roof_drift'], push['base_shear'], color=col, lw=2)
    ax.axhline(sa['V_static'], color='navy', ls='--', lw=1.5,
               label='V_stat={:.0f}kN'.format(sa['V_static']))
    ax.axhline(push['V_max'], color='gray', ls=':', lw=1.2,
               label='V_max={:.0f}kN'.format(push['V_max']))
    ax.set_xlabel('Roof Drift (%)'); ax.set_ylabel('Base Shear (kN)')
    ax.set_title('Pushover Capacity Curve')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panels 4+ — All storey drifts, 4 per row
    for batch in range(n_drift_rows):
        ax = axes[3 + batch]
        storeys_in_batch = range(batch*4+1, min((batch+1)*4+1, n+1))
        for si in storeys_in_batch:
            ax.plot(th_t, th['drift_all'][si-1]*100, lw=0.8,
                    label='Storey {}'.format(si))
        ax.axhline( lim, color='red', ls='--', lw=1.2)
        ax.axhline(-lim, color='red', ls='--', lw=1.2)
        ax.set_ylabel('Drift (%)')
        ax.set_title('Inter-Storey Drift — Storeys {}-{}'.format(
            batch*4+1, min((batch+1)*4, n)))
        ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    fn1 = 'results_time_history_{}.png'.format(b['era'].replace('-',''))
    plt.savefig(fn1, dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print("  Saved: {}".format(fn1))

    # ── Figure 2: Storey profiles ─────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, max(5, n*0.7+2)))
    fig2.suptitle(title + ' -- Storey Profiles', fontsize=12, fontweight='bold')

    floors    = list(range(n+1))
    storeys   = list(range(1, n+1))
    y_heights = [d['y_floors'][fi] for fi in floors]

    # Panel A — PIDR profile (horizontal bar, colour-coded by damage state)
    ax = axes2[0]
    pidr_vals = edp['PIDR_storey']
    bar_cols  = [STATE_COLOURS.get(classify_damage(v),'gray') for v in pidr_vals]
    bars = ax.barh(storeys, [v*100 for v in pidr_vals],
                   color=bar_cols, edgecolor='white', height=0.6)
    ax.axvline(lim, color='red', ls='--', lw=2, label='1.5% limit')
    ax.set_xlabel('Peak PIDR (%)'); ax.set_ylabel('Storey')
    ax.set_title('Peak Inter-Storey Drift')
    ax.set_yticks(storeys); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, pidr_vals):
        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                '{:.3f}%'.format(val*100), va='center', fontsize=8)

    # Panel B — PFA profile (line plot height vs acceleration)
    ax = axes2[1]
    pfa_g = [v/G for v in edp['PFA_storey']]
    ax.plot(pfa_g, y_heights, color='steelblue', lw=2, marker='o', ms=6)
    for fi, (g, y) in enumerate(zip(pfa_g, y_heights)):
        label = 'Ground' if fi==0 else ('Roof' if fi==n else 'Fl.{}'.format(fi))
        ax.text(g+0.003, y, '{:.3f}g\n{}'.format(g,label), fontsize=7.5, va='center')
    ax.set_xlabel('PFA (g)'); ax.set_ylabel('Height (m)')
    ax.set_title('Peak Floor Acceleration Profile')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, d['Hn']+0.5)

    # Panel C — Summary text
    ax = axes2[2]
    ax.axis('off')
    duct_str = '{:.2f}'.format(push['ductility']) if push['ductility'] else 'N/A'
    lines = [
        '='*42,
        ' {} STOREYS -- {}'.format(n, b['era'].upper()),
        '='*42,
        " f'c={} MPa  fy={} MPa".format(b['fc'],b['fy']),
        ' mu={}  Sp={}  Z={}'.format(b['mu'],b['Sp'],b['Z']),
        '-'*42,
        ' T1 (FEM)  = {:.3f} s'.format(ev['T'][0]),
        ' T1 (code) = {:.3f} s  (ratio={:.2f}x)'.format(
            sa['T1_code'], ev['T'][0]/sa['T1_code']),
    ]
    if len(ev['T']) >= 2:
        lines.append(' T2 (FEM)  = {:.3f} s'.format(ev['T'][1]))
    lines += [
        '-'*42,
        ' PIDR max  = {:.3f}% (S{})'.format(edp['PIDR_max']*100, edp['gov_storey']),
        ' Limit     = {:.1f}%'.format(lim),
        ' Damage    = {}'.format(edp['damage_state']),
        '-'*42,
        ' V_static  = {:.1f} kN'.format(edp['V_static']),
        ' V_max     = {:.1f} kN'.format(push['V_max']),
        ' Ductility = {}'.format(duct_str),
        ' Roof disp = {:.1f} mm'.format(edp['max_roof_mm']),
        '='*42,
        ' AS1170.4: {}'.format('COMPLIANT' if edp['compliant'] else 'NON-COMPLIANT'),
    ]
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

    plt.tight_layout()
    fn2 = 'results_storey_profile_{}.png'.format(b['era'].replace('-',''))
    plt.savefig(fn2, dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print("  Saved: {}".format(fn2))

    return fn1, fn2

# =============================================================================
# SECTION 10 — JSON REPORT
# =============================================================================

def save_report(b, d, sa, ev, push, edp):
    report = {
        "project":    "UTS Engineering Graduate Project PG (42003)",
        "student":    "Kabish Jung Thapa (25631413)",
        "generated":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "standard":   "AS1170.4:2007",
        "building": {
            "name":         b['name'],
            "era":          b['era'],
            "num_storeys":  d['n'],
            "storey_heights_m": d['heights'],
            "total_height_m":   d['Hn'],
            "num_bays":     b['num_bays'],
            "bay_width_m":  b['bay_width'],
            "floor_width_m":b['floor_width'],
            "fc_MPa":       b['fc'],
            "fy_MPa":       b['fy'],
            "mu":           b['mu'],
            "Sp":           b['Sp'],
            "Z":            b['Z'],
            "site_class":   b['site_class'],
        },
        "loads": {
            "W_total_kN":       round(d['W_total'],1),
            "W_floor_kN":       round(d['W_floor'],1),
            "M_floor_kNs2m":    round(d['M_floor'],3),
        },
        "static_analysis": {
            "T1_code_s":    round(sa['T1_code'],4),
            "Ch":           round(sa['Ch'],4),
            "V_static_kN":  round(sa['V_static'],2),
            "V_over_W":     round(sa['V_static']/d['W_total'],4),
            "floor_forces_kN": {
                "floor_{}".format(fi): round(sa['floor_forces'][fi],2)
                for fi in range(1, d['n']+1)
            },
        },
        "eigenvalue": {
            "T1_FEM_s":     round(ev['T'][0],4),
            "T2_FEM_s":     round(ev['T'][1],4) if len(ev['T'])>=2 else None,
            "T1_ratio":     round(ev['T'][0]/sa['T1_code'],3),
        },
        "pushover": {
            "V_max_kN":     round(push['V_max'],2),
            "ductility":    round(push['ductility'],3) if push['ductility'] else None,
            "T_eff_s":      round(push['T_eff'],4) if push['T_eff'] else None,
        },
        "time_history_edps": {
            "per_storey": {
                "storey_{}".format(si): {
                    "PIDR_%":    round(edp['PIDR_storey'][si-1]*100,4),
                    "PFA_g":     round(edp['PFA_storey'][si]/G,4),
                    "drift_pass":edp['PIDR_storey'][si-1] <= DRIFT_LIMIT,
                }
                for si in range(1, d['n']+1)
            },
            "PIDR_max_%":       round(edp['PIDR_max']*100,4),
            "governing_storey": edp['gov_storey'],
            "drift_limit_%":    1.5,
            "drift_pass":       edp['drift_pass'],
            "PFA_ground_g":     round(edp['PFA_storey'][0]/G,4),
            "PFA_roof_g":       round(edp['PFA_storey'][-1]/G,4),
            "amp_roof_x":       round(edp['amp_roof'],3),
            "max_roof_disp_mm": round(edp['max_roof_mm'],3),
            "V_dynamic_kN":     round(edp['V_dynamic'],2),
            "V_dyn_stat_ratio": round(edp['V_dynamic']/edp['V_static'],3),
        },
        "damage_assessment": {
            "hazus_damage_state": edp['damage_state'],
            "as1170_4_result":    "COMPLIANT" if edp['compliant'] else "NON-COMPLIANT",
        },
    }

    fn = 'seismic_report_{}storeys.json'.format(d['n'])
    with open(fn,'w') as f:
        json.dump(report, f, indent=2)
    print("  JSON saved: {}".format(fn))
    return fn

# =============================================================================
# SECTION 11 — CONSOLE SUMMARY
# =============================================================================

def print_summary(b, d, sa, ev, push, edp):
    n = d['n']
    print()
    print('='*65)
    print('  RESULT SUMMARY — {} STOREYS'.format(n))
    print('='*65)
    print('  Building  : {}'.format(b['name']))
    print('  Era       : {}  |  f\'c={} MPa  fy={} MPa'.format(
          b['era'], b['fc'], b['fy']))
    print('  Z={}  Site={}  mu={}  Sp={}'.format(
          b['Z'], b['site_class'], b['mu'], b['Sp']))
    print()
    print('  Geometry  : {} storeys x {:.1f} m = {:.1f} m total'.format(
          n, b['storey_height'] if isinstance(b['storey_height'],float) else '(var)', d['Hn']))
    print('  W_total   : {:.1f} kN  ({:.1f} kN/floor)'.format(d['W_total'],d['W_floor']))
    print()
    print('  Periods:')
    print('    T1 FEM  = {:.3f} s  (code {:.3f} s, ratio={:.2f}x)'.format(
          ev['T'][0], sa['T1_code'], ev['T'][0]/sa['T1_code']))
    if len(ev['T']) >= 2:
        print('    T2 FEM  = {:.3f} s'.format(ev['T'][1]))
    print()
    print('  Static V  = {:.1f} kN  (V/W={:.4f})'.format(
          sa['V_static'], sa['V_static']/d['W_total']))
    print()
    print('  Per-storey PIDR (limit {:.1f}%):'.format(DRIFT_LIMIT*100))
    for si in range(1, n+1):
        pidr = edp['PIDR_storey'][si-1]*100
        flag = '<-- GOVERNING' if si == edp['gov_storey'] else ''
        mark = 'PASS' if pidr<=DRIFT_LIMIT*100 else 'FAIL'
        print('    Storey {:2d}: {:6.3f}%  [{}]  {}'.format(si, pidr, mark, flag))
    print()
    print('  Governing PIDR = {:.3f}%  (Storey {})'.format(
          edp['PIDR_max']*100, edp['gov_storey']))
    print('  Damage state   = {}'.format(edp['damage_state']))
    print()
    duct = '{:.2f}'.format(push['ductility']) if push['ductility'] else 'N/A'
    print('  Pushover V_max = {:.1f} kN  ductility = {}'.format(push['V_max'], duct))
    print('  V_dynamic      = {:.1f} kN  ({:.2f}x static)'.format(
          edp['V_dynamic'], edp['V_dynamic']/edp['V_static']))
    print()
    comply = 'COMPLIANT' if edp['compliant'] else 'NON-COMPLIANT'
    print('  AS1170.4 RESULT: {}'.format(comply))
    print('='*65)

# =============================================================================
# SECTION 12 — MAIN (run everything)
# =============================================================================

print("="*65)
print("  SEISMIC VULNERABILITY ASSESSMENT — N-STOREY GENERALISED")
print("  UTS Engineering Graduate Project PG (42003)")
print("  Kabish Jung Thapa | Supervisor: Prof. Jianchun Li")
print("="*65)
print()
print("  Building    : {}".format(BUILDING['name']))
print("  Storeys     : {}".format(BUILDING['num_storeys']))
print("  Era         : {}".format(BUILDING['era']))
print()

t0 = time.time()

# Step 1 — Derived properties
d = derive(BUILDING)
print("  Height      : {:.1f} m  ({} x {:.1f} m)".format(
      d['Hn'], d['n'],
      BUILDING['storey_height'] if isinstance(BUILDING['storey_height'],float)
      else 'variable'))
print("  W_total     : {:.1f} kN".format(d['W_total']))
print()

# Step 2 — Static analysis
print("--- Static Analysis ---")
sa = static_analysis(BUILDING, d)
print("  T1_code = {:.3f}s  Ch = {:.3f}  V = {:.1f}kN  V/W = {:.4f}".format(
      sa['T1_code'], sa['Ch'], sa['V_static'], sa['V_static']/d['W_total']))
print()

# Step 3 — Build model
print("--- Building OpenSeesPy Model ---")
node_id = build_model(BUILDING, d)
print()

# Step 4 — Eigenvalue
print("--- Eigenvalue Analysis ---")
ev = eigenvalue_analysis(BUILDING, d)
print()

# Step 5 — Pushover
print("--- Pushover Analysis ---")
push = run_pushover(BUILDING, d, node_id, sa)
print()

# Step 6 — Rebuild model (pushover changes state) + time-history
print("--- Rebuilding Model for Time-History ---")
node_id = build_model(BUILDING, d)
print()
print("--- Nonlinear Time-History Analysis ---")
th = run_time_history(BUILDING, d, node_id, ev)
print()

# Step 7 — EDPs
print("--- Computing EDPs ---")
edp = compute_edps(BUILDING, d, th, sa, ev)
print()

# Step 8 — Print summary
print_summary(BUILDING, d, sa, ev, push, edp)
print()

# Step 9 — Plots
print("--- Generating Plots ---")
plot_results(BUILDING, d, th, sa, ev, push, edp)
print()

# Step 10 — JSON report
print("--- Saving Report ---")
save_report(BUILDING, d, sa, ev, push, edp)

print()
print("Total elapsed: {:.1f}s".format(time.time()-t0))
print("Done. Download files from Colab Files panel (left sidebar).")
