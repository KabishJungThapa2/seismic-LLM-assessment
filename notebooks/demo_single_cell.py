# =============================================================================
# SEISMIC VULNERABILITY ASSESSMENT — COMPREHENSIVE DEMO
# LLM-Orchestrated Workflow | UTS Engineering Graduate Project PG (42003)
# Kabish Jung Thapa (25631413) | Supervisor: Prof. Jianchun Li
#
# HOW TO RUN IN GOOGLE COLAB:
#   Paste this entire script into ONE cell → Shift+Enter
#   No API key needed. No file uploads. Everything self-contained.
#
# WHAT THIS COVERS:
#   Mode 1–3  : Verified case study buildings (pre/post-1990/2010)
#   Mode 4    : Run ALL THREE buildings and generate comparison report
#   Mode 5–6  : Describe your own building (free text or guided Q&A)
#
#   Per building:
#     ✓ AS1170.4:2007 equivalent static base shear
#     ✓ Eigenvalue analysis (T1, T2) via fullGenLapack
#     ✓ Pushover analysis  — capacity curve, V_max, ductility
#     ✓ Nonlinear time-history analysis (4000 steps, Newmark)
#     ✓ EDPs: PIDR, PFA, roof displacement, base shear
#     ✓ HAZUS damage state classification
#     ✓ 8-panel results figure
#
#   Multi-building:
#     ✓ Comparison chart (drift, period, shear, pushover, PFA, table)
#     ✓ Comprehensive JSON report
#     ✓ Console summary table
# =============================================================================

# ── INSTALL ───────────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run(['pip', 'install', 'openseespy', 'numpy', 'matplotlib', '-q'],
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
import tempfile, os, json, re, textwrap, time
from datetime import datetime

print("✓ All packages ready\n")

# =============================================================================
# ── CONSTANTS AND LOOKUP TABLES ───────────────────────────────────────────────
# =============================================================================

G           = 9.81
COVER       = 0.040
DRIFT_LIMIT = 0.015

# HAZUS-MH (FEMA 2003) damage thresholds adapted for low-rise RC frames
# Key: (lower_PIDR, upper_PIDR)
DAMAGE_STATES = [
    ("None",      0.000, 0.005),
    ("Slight",    0.005, 0.010),
    ("Moderate",  0.010, 0.020),
    ("Extensive", 0.020, 0.040),
    ("Complete",  0.040, 9.999),
]

ERA_COLOURS = {'pre-1990':'#d9534f', 'post-1990':'#f0ad4e', 'post-2010':'#5cb85c'}

ERA_DEFAULTS = {
    "pre-1990":  dict(fc=20.0, fy=250.0, col_b=0.30, col_h=0.30, beam_b=0.30,
                      beam_h=0.45, col_rho=0.015, beam_rho_t=0.008,
                      beam_rho_c=0.004, mu=2.0, Sp=0.77,
                      epsc0_core=-0.004, epsU_core=-0.012),
    "post-1990": dict(fc=32.0, fy=500.0, col_b=0.35, col_h=0.35, beam_b=0.30,
                      beam_h=0.50, col_rho=0.020, beam_rho_t=0.012,
                      beam_rho_c=0.006, mu=3.0, Sp=0.67,
                      epsc0_core=-0.005, epsU_core=-0.020),
    "post-2010": dict(fc=40.0, fy=500.0, col_b=0.40, col_h=0.40, beam_b=0.35,
                      beam_h=0.55, col_rho=0.025, beam_rho_t=0.015,
                      beam_rho_c=0.0075, mu=4.0, Sp=0.67,
                      epsc0_core=-0.006, epsU_core=-0.030),
}

HAZARD_FACTORS = dict(
    newcastle=0.11, sydney=0.08, wollongong=0.08, nsw=0.11,
    melbourne=0.08, victoria=0.08, brisbane=0.05, queensland=0.05,
    adelaide=0.10, perth=0.09, canberra=0.08, hobart=0.05, darwin=0.09,
)

PARAM_BOUNDS = dict(
    num_storeys=(1,4), storey_height=(2.4,4.5), num_bays=(1,6),
    bay_width=(2.5,8.0), fc=(15.0,65.0), fy=(200.0,600.0),
    col_b=(0.20,0.80), col_h=(0.20,0.80), beam_b=(0.20,0.60),
    beam_h=(0.25,0.80), col_rho=(0.01,0.04), mu=(1.5,6.0), Z=(0.03,0.45),
)

# Three fully-verified case study buildings for Newcastle/Sydney region
def _case(name, era, fc, fy, col_b, col_h, beam_b, beam_h,
          col_rho, beam_rho_t, beam_rho_c, mu, Sp,
          epsc0, epsU):
    return dict(
        building_name=name, num_storeys=2, storey_height=3.0,
        num_bays=3, bay_width=4.0, floor_width=8.0,
        fc=fc, fy=fy, col_b=col_b, col_h=col_h,
        beam_b=beam_b, beam_h=beam_h, col_rho=col_rho,
        beam_rho_t=beam_rho_t, beam_rho_c=beam_rho_c,
        mu=mu, Sp=Sp, Z=0.11, site_class="De",
        dead_load=5.0, live_load=2.0, era=era, confidence=1.0,
        epsc0_core=epsc0, epsU_core=epsU,
        assumptions=["Verified case study — all parameters from thesis Table 2"],
    )

CASE_STUDIES = {
    "1": _case("Building 1 — Pre-1990 Non-Ductile RC Frame",
               "pre-1990",  20.0, 250.0,
               0.30, 0.30, 0.30, 0.45, 0.015, 0.008, 0.004,
               2.0, 0.77, -0.004, -0.012),
    "2": _case("Building 2 — Post-1990 Ductile RC Frame",
               "post-1990", 32.0, 500.0,
               0.35, 0.35, 0.30, 0.50, 0.020, 0.012, 0.006,
               3.0, 0.67, -0.005, -0.020),
    "3": _case("Building 3 — Post-2010 Code-Compliant RC Frame",
               "post-2010", 40.0, 500.0,
               0.40, 0.40, 0.35, 0.55, 0.025, 0.015, 0.0075,
               4.0, 0.67, -0.006, -0.030),
}

# =============================================================================
# ── STEP 1: USER INPUT ────────────────────────────────────────────────────────
# =============================================================================

def get_input_mode():
    print("=" * 65)
    print("  SEISMIC VULNERABILITY ASSESSMENT — COMPREHENSIVE DEMO")
    print("  UTS Engineering Graduate Project PG (42003)")
    print("=" * 65)
    print()
    print("  Choose input mode:")
    print()
    print("  1) Verified Building 1  (Pre-1990,  Newcastle)")
    print("  2) Verified Building 2  (Post-1990, Newcastle)")
    print("  3) Verified Building 3  (Post-2010, Newcastle)")
    print("  4) Run ALL THREE buildings and compare [recommended]")
    print("  5) Describe your own building — free text")
    print("  6) Describe your own building — guided questions")
    print()
    while True:
        c = input("  Enter 1–6: ").strip()
        if c in ('1','2','3','4','5','6'):
            return c
        print("  Please enter a number from 1 to 6.")


def get_param_sets(mode):
    if mode in ('1','2','3'):
        return [CASE_STUDIES[mode]]
    if mode == '4':
        return [CASE_STUDIES['1'], CASE_STUDIES['2'], CASE_STUDIES['3']]
    desc = _free_text() if mode == '5' else _guided_qa()
    return [extract_parameters(desc)]


def _free_text():
    print()
    print("  Describe your building below.")
    print("  Include: location, year, storeys, floor size, construction.")
    print("  Press Enter twice when done.")
    print()
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    desc = " ".join(l for l in lines if l.strip())
    print(f"\n  ✓ Description received ({len(desc)} characters)\n")
    return desc


def _guided_qa():
    print()
    print("  Answer each question. Press Enter to use the default [value].")
    print()
    qs = [
        ("location",  "City or suburb",                        "Newcastle, NSW"),
        ("year",      "Year built (approximate)",               "1985"),
        ("storeys",   "Number of storeys",                      "2"),
        ("length",    "Building length in metres (long side)",  "12"),
        ("width",     "Building width in metres (short side)",  "8"),
        ("condition", "Condition (good / fair / poor)",         "fair"),
        ("features",  "Special features (or press Enter)",      "none"),
    ]
    ans = {}
    for key, q, default in qs:
        v = input(f"  {q} [{default}]: ").strip()
        ans[key] = v if v else default

    desc = (
        "A {}-storey reinforced concrete residential building in {}, "
        "built approximately {}. "
        "Floor plan approximately {} metres by {} metres. "
        "Condition: {}.".format(
            ans['storeys'], ans['location'], ans['year'],
            ans['length'], ans['width'], ans['condition'])
    )
    if ans['features'].lower() not in ('none', ''):
        desc += " Features: {}.".format(ans['features'])
    desc += " Assess under AS1170.4."
    print("\n  ✓ Description: '{}...'\n".format(desc[:80]))
    return desc

# =============================================================================
# ── STEP 2: PARAMETER EXTRACTION (DEMO / KEYWORD MODE) ───────────────────────
# =============================================================================

def extract_parameters(description):
    d = description.lower()
    assumptions = []

    # Era detection
    pre_kws  = [str(y) for y in range(1960, 1990)] + [
        'pre-1990','pre 1990','old','heritage','brick veneer',
        'unreinforced','non-ductile','fibrous cement','asbestos']
    post10   = [str(y) for y in range(2011, 2026)] + [
        'post-2010','post 2010','modern','new build','recently built',
        'fully ductile','contemporary']
    post90   = [str(y) for y in range(1990, 2011)] + ['post-1990','post 1990']

    if   any(k in d for k in pre_kws):  era = "pre-1990"
    elif any(k in d for k in post10):   era = "post-2010"
    elif any(k in d for k in post90):   era = "post-1990"
    else:
        era = "pre-1990"
        assumptions.append("Era not stated — defaulted to pre-1990 (conservative)")

    props = dict(ERA_DEFAULTS[era])
    assumptions.append("Material/section defaults from {} era".format(era))

    # Storeys
    storey_kw = {
        1: ['one storey','1 storey','1-storey','single storey','bungalow'],
        2: ['two storey','2 storey','2-storey','double storey'],
        3: ['three storey','3 storey','3-storey'],
        4: ['four storey','4 storey','4-storey'],
    }
    num_storeys = 2
    for n, kws in storey_kw.items():
        if any(k in d for k in kws):
            num_storeys = n
            break
    m = re.search(r'(\d)\s*(?:floor|level)', d)
    if m:
        num_storeys = int(m.group(1))
    if not any(k in d for k in ['storey','story','floor','level']):
        assumptions.append("Storeys not stated — defaulted to 2")

    # Floor geometry
    floor_length, floor_width, num_bays, bay_width = 12.0, 8.0, 3, 4.0
    found_geom = False
    for pat in [
        r'(\d+(?:\.\d+)?)\s*(?:m|metres?)?\s*(?:x|by|×)\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*×\s*(\d+(?:\.\d+)?)',
    ]:
        m = re.search(pat, d)
        if m:
            l, w = float(m.group(1)), float(m.group(2))
            if w > l: l, w = w, l
            floor_length = l
            floor_width  = w
            num_bays     = max(2, round(l / 4))
            bay_width    = l / num_bays
            found_geom   = True
            break
    if not found_geom:
        assumptions.append("Floor plan not stated — defaulted to 12m x 8m, 3 bays x 4m")

    # Seismic zone
    Z, site_class = 0.11, 'De'
    city_map = {
        'newcastle': ('newcastle','De'), 'sydney': ('sydney','De'),
        'nsw': ('sydney','De'),          'wollongong': ('sydney','De'),
        'melbourne': ('melbourne','Ce'), 'victoria': ('melbourne','Ce'),
        'brisbane': ('brisbane','Ce'),   'queensland': ('brisbane','Ce'),
        'adelaide': ('adelaide','Ce'),   'perth': ('perth','Ce'),
        'canberra': ('canberra','Ce'),   'hobart': ('hobart','Ce'),
        'darwin': ('darwin','Ce'),
    }
    found_city = False
    for kw, (city, site) in city_map.items():
        if kw in d:
            Z = HAZARD_FACTORS[city]
            site_class = site
            found_city = True
            break
    if not found_city:
        assumptions.append("Location not identified — defaulted to Newcastle Z=0.11, Site De")

    return dict(
        building_name="{} RC Frame ({}-storey)".format(era.capitalize(), num_storeys),
        num_storeys=num_storeys, storey_height=3.0,
        num_bays=num_bays, bay_width=round(bay_width, 2),
        floor_width=floor_width,
        fc=props['fc'], fy=props['fy'],
        col_b=props['col_b'], col_h=props['col_h'],
        beam_b=props['beam_b'], beam_h=props['beam_h'],
        col_rho=props['col_rho'], beam_rho_t=props['beam_rho_t'],
        beam_rho_c=props['beam_rho_c'],
        epsc0_core=props['epsc0_core'], epsU_core=props['epsU_core'],
        mu=props['mu'], Sp=props['Sp'], Z=Z, site_class=site_class,
        dead_load=5.0, live_load=2.0,
        era=era, confidence=0.75, assumptions=assumptions,
    )

# =============================================================================
# ── STEP 3: HUMAN VERIFICATION CHECKPOINT ────────────────────────────────────
# =============================================================================

def verification_checkpoint(params, auto_approve=False):
    p = params
    print()
    print("=" * 65)
    print("  HUMAN VERIFICATION CHECKPOINT")
    print("=" * 65)
    print("  Building   : {}".format(p['building_name']))
    print("  Era        : {}  |  Confidence: {:.0%}".format(
        p['era'], p['confidence']))
    print("  Geometry   : {} storeys, {}x{}m bays, {}m wide".format(
        p['num_storeys'], p['num_bays'], p['bay_width'], p['floor_width']))
    print("  Materials  : f'c={} MPa, fy={} MPa".format(p['fc'], p['fy']))
    print("  Columns    : {}x{} mm,  rho={:.1f}%".format(
        int(p['col_b']*1000), int(p['col_h']*1000), p['col_rho']*100))
    print("  Beams      : {}x{} mm".format(
        int(p['beam_b']*1000), int(p['beam_h']*1000)))
    print("  Seismic    : Z={}, Site={}, mu={}, Sp={}".format(
        p['Z'], p['site_class'], p['mu'], p['Sp']))

    if p.get('assumptions'):
        print()
        for a in p['assumptions']:
            print("    [!] {}".format(a))

    # Bounds check
    warn = []
    for key, (lo, hi) in PARAM_BOUNDS.items():
        val = params.get(key)
        if val is not None and not (lo <= float(val) <= hi):
            warn.append("    OUT OF RANGE: {}={} (expected {}-{})".format(key, val, lo, hi))
    if warn:
        print()
        for w in warn: print(w)

    if auto_approve:
        print("\n  [auto-approved — preset case study]\n")
        return params, True

    print()
    print("  [Y] Approve  [E] Edit a value  [N] Reject")
    while True:
        c = input("  Choice: ").strip().upper()
        if c == 'Y':
            print("  Approved.\n")
            return params, True
        elif c == 'E':
            params = _edit_param(params)
            return verification_checkpoint(params, False)
        elif c == 'N':
            return params, False
        print("  Enter Y, E or N.")


def _edit_param(params):
    print("  Editable: " + ", ".join(PARAM_BOUNDS.keys()))
    key = input("  Parameter to edit: ").strip()
    if key not in params:
        print("  '{}' not found.".format(key))
        return params
    current = params[key]
    raw = input("  {} [{}] -> ".format(key, current)).strip()
    try:
        params[key] = int(raw) if isinstance(current, int) else float(raw)
        params.setdefault('assumptions', []).append(
            "User manually set {}={}".format(key, params[key]))
        print("  Updated: {}={}".format(key, params[key]))
    except ValueError:
        print("  Invalid input — keeping {}={}".format(key, current))
    return params

# =============================================================================
# ── STEP 4A: AS1170.4 STATIC ANALYSIS ────────────────────────────────────────
# =============================================================================

def static_analysis(params, W_total):
    p  = params
    Hn = p['num_storeys'] * p['storey_height']
    T1 = 0.075 * Hn ** 0.75

    if   T1 <= 0.10: Ch = 2.35
    elif T1 <  1.50: Ch = 1.65 * (0.1 / T1) ** 0.85
    else:            Ch = 1.10 * (1.5 / T1) ** 2.0

    V = max((p['Z'] / p['mu']) * p['Sp'] * Ch * W_total, 0.01 * W_total)
    return V, Ch, T1

# =============================================================================
# ── STEP 4B: BUILD OPENSEESPY MODEL ──────────────────────────────────────────
# =============================================================================

def build_model(p):
    """
    Build 2D nonlinear RC fibre frame and run gravity analysis.
    Returns (node_id, M_floor, W_total).

    Technical notes:
      - equalDOF rigid diaphragm on DOF 1 (X) at each floor
      - Masses assigned to master nodes only (avoids ARPACK failure)
      - Use ops.eigen('-fullGenLapack', n) with equalDOF
      - Gravity uses constraints('Plain')
      - Transient must use constraints('Transformation')
    """
    fc_kN = p['fc']  * 1000
    fy_kN = p['fy']  * 1000
    Es_kN = 200000.0 * 1000
    Ac    = p['col_b']  * p['col_h'];   Asc  = p['col_rho']    * Ac
    Ab    = p['beam_b'] * p['beam_h'];  Ast  = p['beam_rho_t'] * Ab
    Asc2  = p['beam_rho_c'] * Ab

    floor_area = p['num_bays'] * p['bay_width'] * p['floor_width']
    W_floor    = (p['dead_load'] + 0.3 * p['live_load']) * floor_area
    W_total    = W_floor * p['num_storeys']
    M_floor    = W_floor / G
    P_int = (p['dead_load'] + p['live_load']) * p['floor_width']/2 * p['bay_width']
    P_ext = P_int / 2

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    xs = [j * p['bay_width']     for j in range(p['num_bays']+1)]
    ys = [i * p['storey_height'] for i in range(p['num_storeys']+1)]

    node_id = []
    for fi, y in enumerate(ys):
        row = []
        for ci, x in enumerate(xs):
            nid = (fi+1)*10 + (ci+1)
            ops.node(nid, x, y)
            row.append(nid)
        node_id.append(row)

    for nid in node_id[0]:
        ops.fix(nid, 1, 1, 1)

    for fi in range(1, len(ys)):
        master = node_id[fi][0]
        for slave in node_id[fi][1:]:
            ops.equalDOF(master, slave, 1)

    epsc0 = p.get('epsc0_core', -0.004)
    epsU  = p.get('epsU_core',  -0.012)
    ops.uniaxialMaterial('Concrete01', 1, -fc_kN, epsc0, -0.2*fc_kN, epsU)
    ops.uniaxialMaterial('Concrete01', 2, -fc_kN, -0.002, 0.0, -0.004)
    ops.uniaxialMaterial('Steel01',    3,  fy_kN, Es_kN, 0.01)

    cy = p['col_h']/2 - COVER;  cz = p['col_b']/2 - COVER
    ops.section('Fiber', 1)
    ops.patch('rect', 1, 10, 10, -cy, -cz,  cy,  cz)
    ops.patch('rect', 2, 10,  2,  cy, -p['col_b']/2,  p['col_h']/2, p['col_b']/2)
    ops.patch('rect', 2, 10,  2, -p['col_h']/2, -p['col_b']/2, -cy, p['col_b']/2)
    ops.patch('rect', 2,  2, 10, -cy, -p['col_b']/2,  cy, -cz)
    ops.patch('rect', 2,  2, 10, -cy,  cz,             cy,  p['col_b']/2)
    As_bar = max(Asc/6, 1e-5)
    ops.layer('straight', 3, 3, As_bar, -cy, -cz, -cy, cz)
    ops.layer('straight', 3, 3, As_bar,  cy, -cz,  cy, cz)

    by = p['beam_h']/2 - COVER;  bz = p['beam_b']/2 - COVER
    ops.section('Fiber', 2)
    ops.patch('rect', 1, 10, 10, -by, -bz,  by,  bz)
    ops.patch('rect', 2, 10,  2,  by, -p['beam_b']/2,  p['beam_h']/2, p['beam_b']/2)
    ops.patch('rect', 2, 10,  2, -p['beam_h']/2, -p['beam_b']/2, -by, p['beam_b']/2)
    ops.layer('straight', 3, 3, Ast/3,  -by, -bz, -by, bz)
    ops.layer('straight', 3, 3, Asc2/3,  by, -bz,  by, bz)

    ops.geomTransf('PDelta',  1)
    ops.geomTransf('Linear', 2)

    eid = 100
    for fi in range(p['num_storeys']):
        for ci in range(p['num_bays']+1):
            ops.element('nonlinearBeamColumn', eid,
                        node_id[fi][ci], node_id[fi+1][ci], 5, 1, 1)
            eid += 1
    for fi in range(1, p['num_storeys']+1):
        for ci in range(p['num_bays']):
            ops.element('nonlinearBeamColumn', eid,
                        node_id[fi][ci], node_id[fi][ci+1], 5, 2, 2)
            eid += 1

    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)
    for fi in range(1, p['num_storeys']+1):
        for ci, nid in enumerate(node_id[fi]):
            P = -P_ext if (ci == 0 or ci == p['num_bays']) else -P_int
            ops.load(nid, 0.0, P, 0.0)
    ops.system('BandGeneral');  ops.numberer('RCM')
    ops.constraints('Plain');   ops.integrator('LoadControl', 0.1)
    ops.algorithm('Newton');    ops.analysis('Static')
    ops.analyze(10);            ops.loadConst('-time', 0.0)

    for fi in range(1, p['num_storeys']+1):
        ops.mass(node_id[fi][0], M_floor, M_floor, 0.0)

    return node_id, M_floor, W_total

# =============================================================================
# ── STEP 4C: EIGENVALUE ANALYSIS ─────────────────────────────────────────────
# =============================================================================

def eigenvalue_analysis(p):
    """
    fullGenLapack required — ARPACK fails with equalDOF + master-node mass.
    See docs/KNOWN_ISSUES.md for full explanation.
    """
    eigs   = ops.eigen('-fullGenLapack', p['num_storeys'])
    omega1 = abs(eigs[0]) ** 0.5
    omega2 = abs(eigs[1]) ** 0.5 if p['num_storeys'] >= 2 else omega1 * 3.0
    T1 = 2.0 * np.pi / omega1
    T2 = 2.0 * np.pi / omega2
    return T1, T2, omega1, omega2, eigs

# =============================================================================
# ── STEP 4D: PUSHOVER ANALYSIS ────────────────────────────────────────────────
# =============================================================================

def run_pushover(node_id, p, W_total, V_static):
    """
    Displacement-controlled pushover to 3% roof drift.
    Inverted-triangle lateral load distribution.
    Returns capacity curve arrays and key performance points.
    """
    Hn          = p['num_storeys'] * p['storey_height']
    target_disp = 0.03 * Hn
    n_steps     = 200

    heights     = [(i+1) * p['storey_height'] for i in range(p['num_storeys'])]
    total_h     = sum(heights)
    fractions   = [h / total_h for h in heights]

    ops.timeSeries('Linear', 10)
    ops.pattern('Plain', 10, 10)
    for fi, frac in enumerate(fractions, 1):
        ops.load(node_id[fi][0], frac, 0.0, 0.0)

    ops.system('UmfPack');          ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.test('NormDispIncr', 1e-6, 100, 0)
    ops.algorithm('Newton')
    ops.integrator('DisplacementControl',
                   node_id[-1][0], 1, target_disp / n_steps)
    ops.analysis('Static')

    roof_disp  = [0.0]
    base_shear = [0.0]

    for _ in range(n_steps):
        ok = ops.analyze(1)
        if ok != 0:
            ops.algorithm('KrylovNewton')
            ok = ops.analyze(1)
            ops.algorithm('Newton')
            if ok != 0:
                break

        d = ops.nodeDisp(node_id[-1][0], 1)
        V = sum(-ops.nodeReaction(nid, 1) for nid in node_id[0])
        roof_disp.append(d)
        base_shear.append(V)

    rd = np.array(roof_disp)
    bs = np.array(base_shear)

    V_max = float(np.max(bs)) if len(bs) > 1 else V_static
    # Yield displacement at 60% of V_max (bilinear idealisation)
    V_yield  = 0.60 * V_max
    d_yield  = next((rd[i] for i, v in enumerate(bs) if v >= V_yield), None)
    ductility = (rd[-1] / d_yield) if (d_yield and d_yield > 0) else None

    return dict(
        roof_disp   = rd,
        base_shear  = bs,
        roof_drift  = rd / Hn * 100,   # %
        V_max       = V_max,
        d_yield     = d_yield,
        ductility   = ductility,
    )

# =============================================================================
# ── STEP 4E: TIME-HISTORY ANALYSIS ───────────────────────────────────────────
# =============================================================================

def run_time_history(node_id, p, omega1, omega2):
    """
    Nonlinear transient analysis under synthetic sine-wave ground motion.
    Rayleigh damping 5% at modes 1 and 2.
    Convergence fallback: Newton -> KrylovNewton -> ModifiedNewton.
    """
    xi = 0.05
    a0 = xi * 2.0 * omega1 * omega2 / (omega1 + omega2)
    a1 = xi * 2.0 / (omega1 + omega2)
    ops.rayleigh(a0, 0.0, 0.0, a1)

    dt = 0.01; duration = 20.0
    t  = np.arange(0, duration, dt)
    freq  = min(1.0 / (2.0 * np.pi / omega1), 4.0)
    env   = np.sin(np.pi * t / duration)
    accel = p['Z'] * G * env * np.sin(2.0 * np.pi * freq * t)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    np.savetxt(tmp.name, accel, fmt='%.8f')
    tmp.close()

    ops.timeSeries('Path', 2, '-dt', dt, '-filePath', tmp.name, '-factor', 1.0)
    ops.pattern('UniformExcitation', 2, 1, '-accel', 2)
    ops.system('UmfPack');  ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.test('NormDispIncr', 1.0e-8, 10, 0)
    ops.integrator('Newmark', 0.5, 0.25)
    ops.algorithm('Newton')
    ops.analysis('Transient')

    dt_sub  = dt / 2.0
    n_steps = int(len(t) * dt / dt_sub)
    time_h, disp_g, disp_f, disp_r = [], [], [], []
    n_fail = 0

    for _ in range(n_steps):
        ok = ops.analyze(1, dt_sub)
        if ok != 0:
            n_fail += 1
            ops.algorithm('KrylovNewton')
            ok = ops.analyze(1, dt_sub / 5.0)
            if ok != 0:
                ops.test('NormDispIncr', 1.0e-6, 100, 0)
                ops.algorithm('ModifiedNewton', '-initial')
                ops.analyze(1, dt_sub / 10.0)
            ops.algorithm('Newton')
            ops.test('NormDispIncr', 1.0e-8, 10, 0)

        time_h.append(ops.getTime())
        disp_g.append(ops.nodeDisp(node_id[0][0],  1))
        disp_f.append(ops.nodeDisp(node_id[1][0],  1))
        disp_r.append(ops.nodeDisp(node_id[-1][0], 1))

    try:
        os.remove(tmp.name)
    except Exception:
        pass

    dg = np.array(disp_g)
    df = np.array(disp_f)
    dr = np.array(disp_r)
    h  = p['storey_height']

    return dict(
        time_h    = np.array(time_h),
        disp_g=dg, disp_f=df, disp_r=dr,
        drift_s1  = (df - dg) / h,
        drift_s2  = (dr - df) / h,
        n_fail    = n_fail,
    )

# =============================================================================
# ── STEP 4F: POST-PROCESSING ──────────────────────────────────────────────────
# =============================================================================

def classify_damage(pidr):
    for name, lo, hi in DAMAGE_STATES:
        if lo <= pidr < hi:
            return name
    return "Complete"


def compute_edps(th, p, M_floor, W_total, V_static, T1):
    PIDR1 = float(np.max(np.abs(th['drift_s1'])))
    PIDR2 = float(np.max(np.abs(th['drift_s2'])))
    PIDR  = max(PIDR1, PIDR2)

    omega1      = 2.0 * np.pi / T1
    PFA_ground  = p['Z'] * G
    PFA_f1      = omega1**2 * float(np.max(np.abs(th['disp_f'])))
    PFA_roof    = omega1**2 * float(np.max(np.abs(th['disp_r'])))
    V_dyn       = M_floor * PFA_f1 + M_floor * PFA_roof
    max_roof_mm = float(np.max(np.abs(th['disp_r']))) * 1000

    return dict(
        PIDR1       = PIDR1,
        PIDR2       = PIDR2,
        PIDR_max    = PIDR,
        PFA_ground  = PFA_ground,
        PFA_f1      = PFA_f1,
        PFA_roof    = PFA_roof,
        amp_f1      = PFA_f1  / PFA_ground,
        amp_roof    = PFA_roof / PFA_ground,
        V_static    = V_static,
        V_dynamic   = V_dyn,
        W_total     = W_total,
        max_roof_mm = max_roof_mm,
        damage_state= classify_damage(PIDR),
        drift_pass  = PIDR <= DRIFT_LIMIT,
        compliant   = PIDR <= DRIFT_LIMIT,
    )

# =============================================================================
# ── STEP 4: ORCHESTRATE FULL ASSESSMENT PER BUILDING ─────────────────────────
# =============================================================================

def assess_building(params, auto_approve=False):
    p = params
    print()
    print("=" * 65)
    print("  ASSESSING: {}".format(p['building_name']))
    print("=" * 65)

    p, approved = verification_checkpoint(p, auto_approve=auto_approve)
    if not approved:
        print("  Skipped (rejected).")
        return None

    t0 = time.time()

    # --- Model + static ---
    node_id, M_floor, W_total = build_model(p)
    n_col = (p['num_bays']+1) * p['num_storeys']
    n_bm  = p['num_bays']     * p['num_storeys']
    print("  Model: {} nodes, {} elements ({} col + {} beam)".format(
        (p['num_bays']+1)*(p['num_storeys']+1), n_col+n_bm, n_col, n_bm))

    V_static, Ch, T1_code = static_analysis(p, W_total)
    print("  Static: W={:.1f} kN, T1_code={:.3f} s, V={:.1f} kN (V/W={:.4f})".format(
        W_total, T1_code, V_static, V_static/W_total))

    # --- Eigenvalue ---
    T1, T2, omega1, omega2, eigs = eigenvalue_analysis(p)
    print("  Eigenvalue: T1={:.3f} s  T2={:.3f} s  (T1 ratio={:.2f}x code)".format(
        T1, T2, T1/T1_code))

    # --- Pushover ---
    print("  Pushover: running...")
    try:
        push = run_pushover(node_id, p, W_total, V_static)
        duct_str = "{:.2f}".format(push['ductility']) if push['ductility'] else "N/A"
        print("  Pushover: V_max={:.1f} kN  ductility={}".format(
            push['V_max'], duct_str))
    except Exception as ex:
        print("  Pushover failed ({}) — substituting zeros".format(ex))
        push = dict(roof_disp=np.array([0.0]), base_shear=np.array([0.0]),
                    roof_drift=np.array([0.0]), V_max=V_static,
                    d_yield=None, ductility=None)

    # Rebuild after pushover modifies the model state
    node_id, M_floor, W_total = build_model(p)

    # --- Time-history ---
    print("  Time-history: running 4000 steps...")
    th = run_time_history(node_id, p, omega1, omega2)
    print("  Time-history: done  (fallback steps={})".format(th['n_fail']))

    # --- EDPs ---
    edp = compute_edps(th, p, M_floor, W_total, V_static, T1)

    elapsed = time.time() - t0
    print()
    print("  PIDR max   : {:.3f}%  (limit 1.5%)  → {}".format(
        edp['PIDR_max']*100,
        "PASS" if edp['drift_pass'] else "FAIL"))
    print("  Damage     : {}".format(edp['damage_state']))
    print("  PFA roof   : {:.3f} m/s2 ({:.3f}g)".format(
        edp['PFA_roof'], edp['PFA_roof']/G))
    print("  V dynamic  : {:.1f} kN ({:.2f}x static)".format(
        edp['V_dynamic'], edp['V_dynamic']/V_static))
    print("  Elapsed    : {:.1f} s".format(elapsed))
    print("  RESULT     : {}".format(
        "COMPLIANT" if edp['compliant'] else "NON-COMPLIANT"))

    return dict(
        params    = p,
        T1        = T1,  T2        = T2,
        T1_code   = T1_code,
        V_static  = V_static,
        W_total   = W_total,
        M_floor   = M_floor,
        time_h    = th['time_h'],
        disp_f    = th['disp_f'],
        disp_r    = th['disp_r'],
        drift_s1  = th['drift_s1'],
        drift_s2  = th['drift_s2'],
        pushover  = push,
        elapsed   = elapsed,
        **edp,
    )

# =============================================================================
# ── STEP 5: PLOTS ─────────────────────────────────────────────────────────────
# =============================================================================

def plot_single(r, filename=None):
    """8-panel detailed results plot for one building."""
    p   = r['params']
    th  = r['time_h']
    lim = DRIFT_LIMIT * 100
    col = ERA_COLOURS.get(p['era'], 'steelblue')

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(p['building_name'], fontsize=13, fontweight='bold', y=0.99)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.38)

    # 1 — Roof displacement
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(th, r['disp_r']*1000, color='steelblue', lw=1)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('Time (s)');  ax.set_ylabel('Displacement (mm)')
    ax.set_title('Roof Displacement');  ax.grid(True, alpha=0.3)

    # 2 — Drift time histories
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(th, r['drift_s1']*100, color='tomato',     lw=1, label='Storey 1')
    ax.plot(th, r['drift_s2']*100, color='darkorange', lw=1, label='Storey 2')
    ax.axhline( lim, color='red', ls='--', lw=1.5, label='{:.0f}% limit'.format(lim))
    ax.axhline(-lim, color='red', ls='--', lw=1.5)
    ax.set_xlabel('Time (s)');  ax.set_ylabel('Drift (%)')
    ax.set_title('Inter-Storey Drift');  ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # 3 — Pushover
    ax = fig.add_subplot(gs[0, 2])
    push = r['pushover']
    ax.plot(push['roof_drift'], push['base_shear'], color=col, lw=2, label='Capacity')
    ax.axhline(r['V_static'], color='navy', ls='--', lw=1.5,
               label='V_stat={:.0f}kN'.format(r['V_static']))
    if push['V_max'] > 0:
        ax.axhline(push['V_max'], color='gray', ls=':', lw=1,
                   label='V_max={:.0f}kN'.format(push['V_max']))
    ax.set_xlabel('Roof Drift (%)');  ax.set_ylabel('Base Shear (kN)')
    ax.set_title('Pushover Capacity Curve');  ax.legend(fontsize=7);  ax.grid(True, alpha=0.3)

    # 4 — Peak drift profile
    ax = fig.add_subplot(gs[1, 0])
    dvals  = [r['PIDR1']*100, r['PIDR2']*100]
    bcolors = ['#d9534f' if v > DRIFT_LIMIT else '#5cb85c'
               for v in (r['PIDR1'], r['PIDR2'])]
    bars = ax.barh([1, 2], dvals, color=bcolors, edgecolor='white')
    ax.axvline(lim, color='red', ls='--', lw=1.5, label='Limit')
    ax.set_xlabel('Peak Drift (%)');  ax.set_ylabel('Storey')
    ax.set_title('Peak Drift Profile');  ax.legend(fontsize=8);  ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, dvals):
        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                '{:.3f}%'.format(val), va='center', fontsize=9)

    # 5 — Floor acceleration profile
    ax = fig.add_subplot(gs[1, 1])
    pfas   = [r['PFA_ground']/G, r['PFA_f1']/G, r['PFA_roof']/G]
    floors = [0, 1, 2]
    ax.barh(floors, pfas, color=['navy','steelblue','cornflowerblue'])
    ax.set_xlabel('PFA (g)');  ax.set_title('Peak Floor Accelerations')
    ax.set_yticks([0,1,2]);  ax.set_yticklabels(['Ground','Floor 1','Roof'])
    ax.grid(True, alpha=0.3, axis='x')
    for f, v in zip(floors, pfas):
        ax.text(v+0.002, f, '{:.3f}g'.format(v), va='center', fontsize=9)

    # 6 — Damage state indicator
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    ds_colours = {'None':'#5cb85c','Slight':'#a8d08d',
                  'Moderate':'#f0ad4e','Extensive':'#e07b39','Complete':'#d9534f'}
    ds_col = ds_colours.get(r['damage_state'], 'gray')
    ax.add_patch(plt.Circle((0.5, 0.5), 0.38, color=ds_col, alpha=0.85))
    ax.text(0.5, 0.53, r['damage_state'], ha='center', va='center',
            fontsize=15, fontweight='bold', color='white')
    ax.text(0.5, 0.18, 'Damage State',             ha='center', fontsize=10, color='#555')
    ax.text(0.5, 0.10, 'PIDR={:.3f}%'.format(r['PIDR_max']*100),
            ha='center', fontsize=9, color='#555')
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)

    # 7 — Storey 1 hysteresis loop
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(r['drift_s1']*100, r['disp_f']*1000, color=col, lw=0.7, alpha=0.85)
    ax.axvline(0, color='k', lw=0.4);  ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('Storey 1 Drift (%)');  ax.set_ylabel('Floor 1 Disp. (mm)')
    ax.set_title('Hysteresis — Storey 1');  ax.grid(True, alpha=0.3)

    # 8 — Text summary
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis('off')
    duct_val = '{:.2f}'.format(push['ductility']) if push['ductility'] else 'N/A'
    comply   = 'COMPLIANT' if r['compliant'] else 'NON-COMPLIANT'
    lines = [
        '=' * 50,
        '  RESULT SUMMARY — {}'.format(p['era'].upper()),
        '=' * 50,
        "  f'c={} MPa  fy={} MPa  Z={}  Site={}".format(
            p['fc'], p['fy'], p['Z'], p['site_class']),
        "  Columns {0}x{0} mm  col_rho={1:.1f}%  mu={2}".format(
            int(p['col_b']*1000), p['col_rho']*100, p['mu']),
        '-' * 50,
        '  T1 (FEM)  = {:.3f} s   T1 (code) = {:.3f} s'.format(r['T1'], r['T1_code']),
        '  PIDR S1   = {:.3f}%   PIDR S2   = {:.3f}%'.format(
            r['PIDR1']*100, r['PIDR2']*100),
        '  Max PIDR  = {:.3f}%   Limit = 1.5%'.format(r['PIDR_max']*100),
        '  Damage    = {}'.format(r['damage_state']),
        '-' * 50,
        '  V_max pushover = {:.1f} kN  Ductility = {}'.format(
            push['V_max'], duct_val),
        '  V_static  = {:.1f} kN  V_dynamic = {:.1f} kN'.format(
            r['V_static'], r['V_dynamic']),
        '  Roof disp = {:.1f} mm'.format(r['max_roof_mm']),
        '  Amp (roof/ground) = {:.2f}x'.format(r['amp_roof']),
        '=' * 50,
        '  AS1170.4 RESULT: {}'.format(comply),
    ]
    ax.text(0.02, 0.97, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

    fn = filename or 'results_{}.png'.format(p['era'].replace('-','').replace(' ','_'))
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.show();  plt.close()
    print("  Plot saved: {}".format(fn))
    return fn


def plot_comparison(results_list):
    """6-panel comparison chart for multiple buildings."""
    if len(results_list) < 2:
        return
    print()
    print("  Generating comparison chart...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        'Seismic Vulnerability Comparison — Newcastle RC Residential Buildings\n'
        'AS1170.4:2007  |  UTS Engineering Graduate Project PG (42003)',
        fontsize=12, fontweight='bold')

    n      = len(results_list)
    names  = ['B{}'.format(i+1) for i in range(n)]
    eras   = [r['params']['era'] for r in results_list]
    colors = [ERA_COLOURS.get(e, 'gray') for e in eras]

    # 1 — PIDR bar chart
    ax = axes[0, 0]
    bars = ax.bar(names, [r['PIDR_max']*100 for r in results_list], color=colors)
    ax.axhline(DRIFT_LIMIT*100, color='red', ls='--', lw=2, label='1.5% limit')
    ax.set_ylabel('Peak PIDR (%)');  ax.set_title('Peak Inter-Storey Drift')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3, axis='y')
    for bar, r in zip(bars, results_list):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                '{:.3f}%'.format(r['PIDR_max']*100), ha='center', fontsize=9)

    # 2 — Period comparison
    ax = axes[0, 1]
    xpos = np.arange(n);  w = 0.35
    ax.bar(xpos-w/2, [r['T1']      for r in results_list], w, color=colors,    label='T1 FEM')
    ax.bar(xpos+w/2, [r['T1_code'] for r in results_list], w, color='lightgray',
           edgecolor='gray', label='T1 code')
    ax.set_xticks(xpos);  ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Period (s)');  ax.set_title('Fundamental Period')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3, axis='y')

    # 3 — Base shear comparison
    ax = axes[0, 2]
    xpos = np.arange(n)
    ax.bar(xpos-w/2, [r['V_static']  for r in results_list], w,
           color=colors,               alpha=1.0, label='V static')
    ax.bar(xpos+w/2, [r['V_dynamic'] for r in results_list], w,
           color=colors,               alpha=0.4, label='V dynamic')
    ax.set_xticks(xpos);  ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Base Shear (kN)');  ax.set_title('Base Shear')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3, axis='y')

    # 4 — Pushover capacity curves
    ax = axes[1, 0]
    for r, col in zip(results_list, colors):
        push = r['pushover']
        ax.plot(push['roof_drift'], push['base_shear'],
                color=col, lw=2.5, label=r['params']['era'])
    ax.set_xlabel('Roof Drift (%)');  ax.set_ylabel('Base Shear (kN)')
    ax.set_title('Pushover Capacity Curves')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # 5 — Floor acceleration amplification
    ax = axes[1, 1]
    xpos = np.arange(3);  w2 = 1.0 / (n+1)
    floor_labels = ['Ground', 'Floor 1', 'Roof']
    for i, (r, col) in enumerate(zip(results_list, colors)):
        pfas = [r['PFA_ground']/G, r['PFA_f1']/G, r['PFA_roof']/G]
        ax.bar(xpos + i*w2, pfas, w2, color=col, alpha=0.85,
               label=r['params']['era'])
    ax.set_xticks(xpos + w2);  ax.set_xticklabels(floor_labels)
    ax.set_ylabel('PFA (g)');  ax.set_title('Peak Floor Accelerations')
    ax.legend(fontsize=7);  ax.grid(True, alpha=0.3, axis='y')

    # 6 — Summary table
    ax = axes[1, 2]
    ax.axis('off')
    col_labels = ['Era', 'PIDR', 'T1 FEM', 'Damage', 'Result']
    table_data = []
    for r in results_list:
        table_data.append([
            r['params']['era'],
            '{:.3f}%'.format(r['PIDR_max']*100),
            '{:.3f}s'.format(r['T1']),
            r['damage_state'],
            'PASS' if r['compliant'] else 'FAIL',
        ])
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc='center', loc='center',
                   bbox=[0, 0.1, 1, 0.85])
    tbl.auto_set_font_size(False);  tbl.set_fontsize(9)
    for i, r in enumerate(results_list):
        bg = '#d4edda' if r['compliant'] else '#f8d7da'
        tbl[i+1, 4].set_facecolor(bg)
        era_col = ERA_COLOURS.get(r['params']['era'], '#eee') + '44'
        tbl[i+1, 0].set_facecolor(era_col)
    ax.set_title('Summary', fontweight='bold')

    patches = [Patch(color=ERA_COLOURS[e], label=e) for e in ERA_DEFAULTS]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 0.005))

    fn = 'comparison_all_buildings.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.show();  plt.close()
    print("  Comparison chart saved: {}".format(fn))
    return fn

# =============================================================================
# ── STEP 6: CONSOLE SUMMARY TABLE ────────────────────────────────────────────
# =============================================================================

def print_summary(results_list):
    print()
    print('=' * 84)
    print('  FINAL SUMMARY TABLE — AS1170.4:2007 COMPLIANCE')
    print('=' * 84)
    hdr = '  {:<32}  {:>7}  {:>7}  {:>8}  {:<12}  {:>8}  {:>10}'
    print(hdr.format('Building', 'T1 FEM', 'PIDR', 'V_stat', 'Damage', 'Ductility', 'AS1170.4'))
    print('  ' + '-'*78)
    for r in results_list:
        duct_str = '{:.2f}'.format(r['pushover']['ductility']) \
                   if r['pushover']['ductility'] else 'N/A'
        row = '  {:<32}  {:>6.3f}s  {:>6.3f}%  {:>7.1f}kN  {:<12}  {:>8}  {:>10}'
        print(row.format(
            r['params']['building_name'][:32],
            r['T1'],
            r['PIDR_max'] * 100,
            r['V_static'],
            r['damage_state'],
            duct_str,
            'PASS' if r['compliant'] else '** FAIL **',
        ))
    print('  ' + '-'*78)
    print('  {:<32}  {:>7}  {:>6}%  {:>7}    {:<12}'.format(
        'AS1170.4 Limit', '—', '1.500', '—', '—'))
    print('=' * 84)

# =============================================================================
# ── STEP 7: JSON REPORT ───────────────────────────────────────────────────────
# =============================================================================

def save_report(results_list):
    report = {
        "project":    "UTS Engineering Graduate Project PG (42003)",
        "student":    "Kabish Jung Thapa (25631413)",
        "supervisor": "Prof. Jianchun Li",
        "generated":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "standard":   "AS1170.4:2007",
        "llm_mode":   "Demo mode — keyword-based extraction, no API key",
        "buildings":  [],
    }

    for r in results_list:
        p    = r['params']
        push = r['pushover']
        report['buildings'].append({
            "name":  p['building_name'],
            "era":   p['era'],
            "structural_parameters": {
                "fc_MPa":    p['fc'],   "fy_MPa":  p['fy'],
                "col_mm":    "{}x{}".format(int(p['col_b']*1000), int(p['col_h']*1000)),
                "beam_mm":   "{}x{}".format(int(p['beam_b']*1000),int(p['beam_h']*1000)),
                "col_rho_%": round(p['col_rho']*100, 2),
                "mu":        p['mu'],   "Sp":       p['Sp'],
                "Z":         p['Z'],    "site":     p['site_class'],
            },
            "loads": {
                "seismic_weight_kN": round(r['W_total'], 1),
                "dead_kPa": p['dead_load'], "live_kPa": p['live_load'],
            },
            "periods": {
                "T1_FEM_s":  round(r['T1'],     4),
                "T2_FEM_s":  round(r['T2'],     4),
                "T1_code_s": round(r['T1_code'],4),
                "T1_ratio":  round(r['T1']/r['T1_code'], 3),
            },
            "static_analysis": {
                "V_static_kN": round(r['V_static'], 2),
                "V_over_W":    round(r['V_static']/r['W_total'], 4),
            },
            "pushover": {
                "V_max_kN":  round(push['V_max'], 2),
                "ductility": round(push['ductility'], 3) if push['ductility'] else None,
            },
            "time_history_edps": {
                "PIDR_s1_%":       round(r['PIDR1']*100, 4),
                "PIDR_s2_%":       round(r['PIDR2']*100, 4),
                "PIDR_max_%":      round(r['PIDR_max']*100, 4),
                "drift_limit_%":   1.5,
                "drift_pass":      r['drift_pass'],
                "PFA_ground_g":    round(r['PFA_ground']/G, 4),
                "PFA_floor1_g":    round(r['PFA_f1']/G, 4),
                "PFA_roof_g":      round(r['PFA_roof']/G, 4),
                "amp_floor1x":     round(r['amp_f1'],   3),
                "amp_roofx":       round(r['amp_roof'],  3),
                "V_dynamic_kN":    round(r['V_dynamic'], 2),
                "V_dyn_stat_ratio":round(r['V_dynamic']/r['V_static'], 3),
                "max_roof_disp_mm":round(r['max_roof_mm'], 3),
            },
            "damage_assessment": {
                "hazus_damage_state": r['damage_state'],
                "as1170_4_result":    "COMPLIANT" if r['compliant'] else "NON-COMPLIANT",
            },
            "run_time_s":  round(r['elapsed'], 1),
            "assumptions": p.get('assumptions', []),
        })

    if len(results_list) > 1:
        report["cross_building_comparison"] = {
            "most_vulnerable_era":
                max(results_list, key=lambda x: x['PIDR_max'])['params']['era'],
            "strongest_era":
                max(results_list, key=lambda x: x['pushover']['V_max'])['params']['era'],
            "all_compliant":
                all(r['compliant'] for r in results_list),
            "pidr_range_%": [
                round(min(r['PIDR_max'] for r in results_list)*100, 3),
                round(max(r['PIDR_max'] for r in results_list)*100, 3),
            ],
            "V_static_range_kN": [
                round(min(r['V_static'] for r in results_list), 1),
                round(max(r['V_static'] for r in results_list), 1),
            ],
            "note": (
                "Pre-1990 buildings attract highest seismic demand (lowest mu) "
                "but have lowest structural capacity. Post-2010 buildings have "
                "lowest demand AND highest capacity."
            ),
        }

    fn = 'seismic_assessment_report.json'
    with open(fn, 'w') as f:
        json.dump(report, f, indent=2)
    print("  JSON report saved: {}".format(fn))
    return fn

# =============================================================================
# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
# =============================================================================

def run():
    print()
    print("╔" + "═"*63 + "╗")
    print("║  SEISMIC VULNERABILITY ASSESSMENT — COMPREHENSIVE DEMO   ║")
    print("║  LLM-Orchestrated Workflow  |  UTS EGP 42003             ║")
    print("║  No API key required  |  OpenSeesPy nonlinear analysis    ║")
    print("╚" + "═"*63 + "╝")
    print()

    mode       = get_input_mode()
    param_sets = get_param_sets(mode)
    auto       = mode in ('1', '2', '3', '4')

    results_list = []
    saved_files  = []

    for i, params in enumerate(param_sets, 1):
        result = assess_building(params, auto_approve=auto)
        if result is None:
            continue
        results_list.append(result)

        era = params['era'].replace('-','').replace(' ','_')
        fn  = 'results_building{}_{}.png'.format(i, era)
        plot_single(result, filename=fn)
        saved_files.append(fn)

    if not results_list:
        print("\n  No assessments completed.")
        return None

    if len(results_list) > 1:
        cf = plot_comparison(results_list)
        if cf:
            saved_files.append(cf)

    print_summary(results_list)
    jf = save_report(results_list)
    saved_files.append(jf)

    print()
    print("╔" + "═"*63 + "╗")
    print("║  COMPLETE                                                 ║")
    for r in results_list:
        name   = r['params']['building_name'][:42]
        status = 'PASS' if r['compliant'] else 'FAIL'
        print("║  [{:<4}]  {:<54}║".format(status, name))
    print("╚" + "═"*63 + "╝")
    print()
    print("  Files saved:")
    for f in saved_files:
        print("  |-- {}".format(f))
    print()
    print("  Download from Colab: Files panel (left sidebar) → right-click")

    return results_list


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    results = run()
