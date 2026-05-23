# =============================================================================
# compliance.py — AS1170.4 EDP computation and compliance checking
# seismic-llm-assessment | UTS Engineering Graduate Project PG (42003)
#
# IMPLEMENTS:
#   - AS 1170.4:2007 Cl 6.2 equivalent static base shear
#   - Amendment 2 (2018) Cl 3.3: minimum kpZ = 0.08
#   - Inter-storey drift, peak floor acceleration (absolute), storey shear
#   - P-Delta stability coefficient theta (Cl 6.5)
#   - ASCE 41-17 performance level (IO / LS / CP)
#   - HAZUS-MH damage state (None / Slight / Moderate / Extensive / Complete)
# =============================================================================

import numpy as np
from config import G, DRIFT_LIMIT, MIN_BASE_SHEAR, SPECTRAL_SHAPE


# ============================================================================
# Equivalent static analysis (AS 1170.4 Cl 6.2 + Amendment 2 Cl 3.3)
# ============================================================================

def static_base_shear(params: dict, W_total: float) -> tuple:
    """
    Compute AS1170.4:2007 + Amendment 2 (2018) equivalent static base shear.

    V = (kpZ / mu) * Sp * Ch(T1) * W_total
    where kpZ = max(kp * Z, 0.08)   per AS 1170.4 Amdt 2 Cl 3.3 Table 3.3
    and   V   >= 0.01 * W_total      per AS 1170.4 Cl 6.2.3

    Returns:
        (V_static, Ch, T1_approx, kpZ)
    """
    p         = params
    Hn        = p['num_storeys'] * p['storey_height']
    T1_approx = 0.075 * Hn ** 0.75   # AS1170.4 Appendix B

    ch_func   = SPECTRAL_SHAPE.get(p['site_class'], SPECTRAL_SHAPE['De'])
    Ch        = ch_func(T1_approx)
    kp        = 1.0
    kpZ       = max(kp * p['Z'], 0.08)   # Amendment 2 Cl 3.3 — CRITICAL
    V         = (kpZ / p['mu']) * p['Sp'] * Ch * W_total
    V         = max(V, MIN_BASE_SHEAR * W_total)

    return V, Ch, T1_approx, kpZ


# ============================================================================
# Engineering Demand Parameters
# ============================================================================

def compute_edps(th_results: dict, params: dict, T1: float,
                 M_floor: float, W_floor: float, V_static: float,
                 W_total: float, gm_accel: np.ndarray = None,
                 gm_dt: float = 0.01) -> dict:
    """
    Compute Engineering Demand Parameters from time-history results.

    EDPs computed:
      1.  PIDR per storey + governing (AS 1170.4 Cl 6.7)
      2.  Peak floor accelerations (ABSOLUTE, from numerical 2nd derivative)
      3.  Storey shears from inertial forces (V_i = sum_{j>=i} m_j*a_abs_j)
      4.  Dynamic vs static base shear ratio
      5.  P-Delta stability coefficient theta per storey (Cl 6.5)
      6.  ASCE 41-17 performance level (IO / LS / CP / Beyond CP)
      7.  HAZUS-MH damage state (None / Slight / Moderate / Extensive / Complete)
      8.  AS 1170.4 compliance (PIDR <= 1.5%)
    """
    p = params
    n = p['num_storeys']
    h = p['storey_height']

    # ---- 1. INTER-STOREY DRIFT ----------------------------------------------
    dg = th_results['disp_g']
    df = th_results['disp_f']
    dr = th_results['disp_r']
    time_h = th_results['time_h']

    drift1 = (df - dg) / h
    drift2 = (dr - df) / h
    PIDR1 = float(np.max(np.abs(drift1)))
    PIDR2 = float(np.max(np.abs(drift2)))
    PIDR  = max(PIDR1, PIDR2)
    govern_storey = 1 if PIDR1 >= PIDR2 else 2

    # ---- 2. ABSOLUTE FLOOR ACCELERATIONS (proper numerical method) ----------
    # a_abs(t) = a_ground(t) + d2/dt2[ u_rel(t) ]
    # We need the ground motion array. If not supplied, fall back to a
    # conservative estimate from displacement time-history alone.
    if gm_accel is not None and len(gm_accel) > 1:
        gm_time = np.arange(len(gm_accel)) * gm_dt
        gm_interp = np.interp(time_h, gm_time, gm_accel)
    else:
        # Fallback: assume ground acceleration is small relative to
        # relative-floor acceleration (only valid at low PGA).
        gm_interp = np.zeros_like(time_h)

    dt_arr = np.diff(time_h)
    dt_avg = float(np.mean(dt_arr[dt_arr > 0])) if len(dt_arr) > 0 else gm_dt

    # Relative acceleration of each floor from second derivative of displacement
    rel_acc_f = np.gradient(np.gradient(df - dg, dt_avg), dt_avg)
    rel_acc_r = np.gradient(np.gradient(dr - dg, dt_avg), dt_avg)

    abs_acc_g = np.abs(gm_interp)
    abs_acc_f = np.abs(gm_interp + rel_acc_f)
    abs_acc_r = np.abs(gm_interp + rel_acc_r)

    PFA_ground = float(np.max(abs_acc_g)) if gm_accel is not None else p['Z'] * G
    PFA_f1     = float(np.max(abs_acc_f))
    PFA_roof   = float(np.max(abs_acc_r))

    # ---- 3. STOREY SHEAR FROM INERTIAL FORCES -------------------------------
    # F_i(t) = m_i * a_abs_i(t).  V_storey_i(t) = sum_{j=i..n} F_j(t)
    # For 2-storey: V_storey1 = F1 + F2;  V_storey2 = F2
    F_f1 = M_floor * (gm_interp + rel_acc_f)
    F_r  = M_floor * (gm_interp + rel_acc_r)
    V_s1_t = F_f1 + F_r
    V_s2_t = F_r
    V_storey_peak = [float(np.max(np.abs(V_s1_t))),
                     float(np.max(np.abs(V_s2_t)))]
    V_dynamic = V_storey_peak[0]   # base shear = storey-1 shear
    V_dyn_static_ratio = V_dynamic / V_static if V_static > 0 else 0.0

    # ---- 4. STOREY LATERAL STIFFNESS (secant) -------------------------------
    delta_1 = PIDR1 * h
    delta_2 = PIDR2 * h
    k_eff_1 = V_storey_peak[0] / max(delta_1, 1e-6)
    k_eff_2 = V_storey_peak[1] / max(delta_2, 1e-6)

    # Soft-storey: k_i < 0.70 * k_{i+1}  (AS 1170.4 Cl 5.2)
    soft_storey = k_eff_1 < 0.70 * k_eff_2

    # ---- 5. P-DELTA STABILITY COEFFICIENT (AS 1170.4 Cl 6.5) ---------------
    # theta_i = (P_i * delta_i) / (V_i * h_i)
    # P_i = cumulative gravity ABOVE storey i
    P_above_1 = W_floor * n         # all floor weights above storey 1
    P_above_2 = W_floor             # only roof above storey 2
    theta_1 = (P_above_1 * delta_1) / max(V_storey_peak[0] * h, 1e-6)
    theta_2 = (P_above_2 * delta_2) / max(V_storey_peak[1] * h, 1e-6)
    theta_max = max(theta_1, theta_2)
    pdelta_critical = theta_max > 0.10

    # ---- 6. ASCE 41-17 PERFORMANCE LEVEL ------------------------------------
    if PIDR < 0.005:
        performance = 'IO'    # Immediate Occupancy
    elif PIDR < 0.015:
        performance = 'LS'    # Life Safety
    elif PIDR < 0.025:
        performance = 'CP'    # Collapse Prevention
    else:
        performance = 'Beyond CP'

    # ---- 7. HAZUS-MH DAMAGE STATE (FEMA, 2003) ------------------------------
    # PIDR thresholds for low-rise RC frame (HAZUS, simplified):
    #   Slight     : PIDR >= 0.4%
    #   Moderate   : PIDR >= 0.8%
    #   Extensive  : PIDR >= 2.0%
    #   Complete   : PIDR >= 5.0%
    if PIDR < 0.004:
        hazus = 'None'
    elif PIDR < 0.008:
        hazus = 'Slight'
    elif PIDR < 0.020:
        hazus = 'Moderate'
    elif PIDR < 0.050:
        hazus = 'Extensive'
    else:
        hazus = 'Complete'

    # ---- 8. COMPLIANCE (AS 1170.4 + Amendment 2) ----------------------------
    drift_pass    = PIDR <= DRIFT_LIMIT
    pdelta_pass   = theta_max <= 0.25     # AS 1170.4 absolute upper bound
    compliant     = drift_pass and pdelta_pass

    return {
        # Drift
        'drift_s1': drift1, 'drift_s2': drift2,
        'PIDR1': PIDR1, 'PIDR2': PIDR2, 'PIDR_max': PIDR,
        'govern_storey': govern_storey,
        'drift_limit': DRIFT_LIMIT, 'drift_pass': drift_pass,
        # Acceleration
        'PFA_ground': PFA_ground, 'PFA_f1': PFA_f1, 'PFA_roof': PFA_roof,
        'amp_f1': PFA_f1 / max(PFA_ground, 1e-6),
        'amp_roof': PFA_roof / max(PFA_ground, 1e-6),
        # Shear
        'V_static': V_static, 'V_dynamic': V_dynamic,
        'V_dyn_static_ratio': V_dyn_static_ratio,
        'V_storey_peak': V_storey_peak,
        # Stiffness
        'k_eff_1': k_eff_1, 'k_eff_2': k_eff_2,
        'soft_storey': soft_storey,
        # P-Delta
        'theta_1': theta_1, 'theta_2': theta_2, 'theta_max': theta_max,
        'pdelta_critical': pdelta_critical, 'pdelta_pass': pdelta_pass,
        # Performance
        'performance_level': performance,
        'hazus_damage': hazus,
        # Overall
        'W_total': W_total,
        'compliant': compliant,
    }


# ============================================================================
# Reporting
# ============================================================================

def print_compliance_report(edp: dict, params: dict,
                             T1: float, T1_approx: float,
                             kpZ: float = None):
    """Print formatted AS1170.4 + Amendment 2 compliance report."""
    p = params
    print()
    print("=" * 65)
    print("  ENGINEERING DEMAND PARAMETERS & AS1170.4 COMPLIANCE")
    print("=" * 65)
    print(f"  Building  : {p.get('building_name','—')}")
    print(f"  Era       : {p.get('era','—')}")
    if kpZ is not None:
        print(f"  Z={p['Z']}, kpZ={kpZ:.3f} (Amdt 2), "
              f"Site {p['site_class']}, μ={p['mu']}, Sp={p['Sp']}")
    else:
        print(f"  Z={p['Z']}, Site {p['site_class']}, μ={p['mu']}, Sp={p['Sp']}")
    print()
    print(f"  Periods:")
    print(f"    T1 (FEM)   : {T1:.3f} s")
    print(f"    T1 (code)  : {T1_approx:.3f} s")
    print(f"    Ratio      : {T1/T1_approx:.2f}× (>1 expected — cracked sections)")
    print()
    lim = edp['drift_limit'] * 100
    print(f"  Inter-Storey Drift (limit {lim:.1f}%):")
    print(f"    Storey 1   : {edp['PIDR1']*100:.3f}%")
    print(f"    Storey 2   : {edp['PIDR2']*100:.3f}%")
    print(f"    Governing  : {edp['PIDR_max']*100:.3f}% (Storey {edp['govern_storey']})")
    print(f"    CHECK      : {'✓ PASS' if edp['drift_pass'] else '✗ FAIL'}")
    print()
    print(f"  Peak Floor Accelerations (absolute):")
    print(f"    Ground PGA : {edp['PFA_ground']:.3f} m/s²  ({edp['PFA_ground']/G:.3f}g)")
    print(f"    Floor 1    : {edp['PFA_f1']:.3f} m/s²  "
          f"({edp['PFA_f1']/G:.3f}g)  amp = {edp['amp_f1']:.2f}×")
    print(f"    Roof       : {edp['PFA_roof']:.3f} m/s²  "
          f"({edp['PFA_roof']/G:.3f}g)  amp = {edp['amp_roof']:.2f}×")
    print()
    print(f"  Base Shear:")
    print(f"    Static     : {edp['V_static']:.1f} kN  "
          f"(V/W = {edp['V_static']/edp['W_total']:.4f})")
    print(f"    Dynamic    : {edp['V_dynamic']:.1f} kN  "
          f"(ratio = {edp['V_dyn_static_ratio']:.2f}×)")
    print()
    print(f"  P-Delta Stability (AS 1170.4 Cl 6.5):")
    print(f"    θ₁         : {edp['theta_1']:.4f}")
    print(f"    θ₂         : {edp['theta_2']:.4f}")
    print(f"    θ_max      : {edp['theta_max']:.4f}  "
          f"(limit 0.10 for P-Δ consideration, 0.25 instability)")
    print(f"    CHECK      : {'✓ PASS' if edp['pdelta_pass'] else '✗ FAIL'}")
    print()
    print(f"  Performance:")
    print(f"    ASCE 41-17 : {edp['performance_level']}")
    print(f"    HAZUS      : {edp['hazus_damage']}")
    print()
    print(f"  OVERALL    : {'✓ COMPLIANT' if edp['compliant'] else '✗ NON-COMPLIANT'}")
    print("=" * 65)
