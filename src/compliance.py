# =============================================================================
# compliance.py — AS1170.4 EDP computation and compliance checking
# seismic-llm-assessment | UTS Engineering Graduate Project PG (42003)
# =============================================================================

import numpy as np
from config import G, DRIFT_LIMIT, MIN_BASE_SHEAR, SPECTRAL_SHAPE


def static_base_shear(params: dict, W_total: float) -> tuple:
    """
    Compute AS1170.4:2007 equivalent static base shear.
    V = (Z * kp / mu) * Sp * Ch(T1) * W

    Returns:
        (V_static, Ch, T1_approx)
    """
    p         = params
    Hn        = p['num_storeys'] * p['storey_height']
    T1_approx = 0.075 * Hn ** 0.75   # AS1170.4 Appendix B

    ch_func   = SPECTRAL_SHAPE.get(p['site_class'], SPECTRAL_SHAPE['De'])
    Ch        = ch_func(T1_approx)
    kp        = 1.0
    V         = (p['Z'] * kp / p['mu']) * p['Sp'] * Ch * W_total
    V         = max(V, MIN_BASE_SHEAR * W_total)

    return V, Ch, T1_approx


def compute_edps(th_results: dict, params: dict, T1: float,
                 M_floor: float, V_static: float, W_total: float) -> dict:
    """Compute Engineering Demand Parameters from time-history results."""
    dg = th_results['disp_g']
    df = th_results['disp_f']
    dr = th_results['disp_r']
    h  = params['storey_height']

    drift1 = (df - dg) / h
    drift2 = (dr - df) / h

    PIDR1  = float(np.max(np.abs(drift1)))
    PIDR2  = float(np.max(np.abs(drift2)))
    PIDR   = max(PIDR1, PIDR2)

    omega1     = 2 * np.pi / T1
    PFA_ground = params['Z'] * G
    PFA_f1     = omega1**2 * float(np.max(np.abs(df)))
    PFA_roof   = omega1**2 * float(np.max(np.abs(dr)))
    V_dyn      = M_floor * PFA_f1 + M_floor * PFA_roof

    drift_pass = PIDR <= DRIFT_LIMIT

    return {
        'drift_s1':   drift1,
        'drift_s2':   drift2,
        'PIDR1':      PIDR1,
        'PIDR2':      PIDR2,
        'PIDR_max':   PIDR,
        'PFA_ground': PFA_ground,
        'PFA_f1':     PFA_f1,
        'PFA_roof':   PFA_roof,
        'V_static':   V_static,
        'V_dynamic':  V_dyn,
        'W_total':    W_total,
        'drift_limit':DRIFT_LIMIT,
        'drift_pass': drift_pass,
        'compliant':  drift_pass,
    }


def print_compliance_report(edp: dict, params: dict,
                             T1: float, T1_approx: float):
    """Print formatted AS1170.4 compliance report."""
    p = params
    print()
    print("=" * 65)
    print("  ENGINEERING DEMAND PARAMETERS & AS1170.4 COMPLIANCE")
    print("=" * 65)
    print(f"  Building  : {p.get('building_name','—')}")
    print(f"  Era       : {p.get('era','—')}")
    print(f"  Z={p['Z']}, Site {p['site_class']}, μ={p['mu']}, Sp={p['Sp']}")
    print()
    print(f"  Periods:")
    print(f"    T1 (FEM)   : {T1:.3f} s")
    print(f"    T1 (code)  : {T1_approx:.3f} s")
    print(f"    Ratio      : {T1/T1_approx:.2f}× (>1 expected — cracked sections)")
    print()
    lim = edp['drift_limit']*100
    print(f"  Inter-Storey Drift (limit {lim:.1f}%):")
    print(f"    Storey 1   : {edp['PIDR1']*100:.3f}%")
    print(f"    Storey 2   : {edp['PIDR2']*100:.3f}%")
    print(f"    Governing  : {edp['PIDR_max']*100:.3f}%")
    print(f"    CHECK      : {'✓ PASS' if edp['drift_pass'] else '✗ FAIL'}")
    print()
    print(f"  Peak Floor Accelerations:")
    print(f"    Ground PGA : {edp['PFA_ground']:.3f} m/s²  ({edp['PFA_ground']/G:.3f}g)")
    print(f"    Floor 1    : {edp['PFA_f1']:.3f} m/s²  ({edp['PFA_f1']/G:.3f}g)")
    print(f"    Roof       : {edp['PFA_roof']:.3f} m/s²  ({edp['PFA_roof']/G:.3f}g)")
    print()
    print(f"  Base Shear:")
    print(f"    Static     : {edp['V_static']:.1f} kN  "
          f"(V/W={edp['V_static']/edp['W_total']:.4f})")
    print(f"    Dynamic    : {edp['V_dynamic']:.1f} kN  "
          f"(ratio={edp['V_dynamic']/edp['V_static']:.2f}×)")
    print()
    print(f"  OVERALL: {'✓ COMPLIANT' if edp['compliant'] else '✗ NON-COMPLIANT'}")
    print("=" * 65)
