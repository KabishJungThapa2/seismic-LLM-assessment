# =============================================================================
# analysis.py — Time-history analysis engine
# seismic-llm-assessment | UTS Engineering Graduate Project PG (42003)
#
# Handles ground motion generation and nonlinear transient analysis.
# Uses Newmark average acceleration with Rayleigh damping at 5% critical.
# Convergence fallback chain: Newton → KrylovNewton → ModifiedNewton
# =============================================================================

import numpy as np
import tempfile
import os
import openseespy.opensees as ops
from config import G


def generate_synthetic_gm(Z: float, T1: float,
                           dt: float = 0.01, duration: float = 20.0) -> tuple:
    """
    Generate a modulated sine wave ground motion.
    Frequency is tuned to half the building period to avoid exact resonance
    while still exciting the structure.

    Args:
        Z:        Hazard factor (PGA = Z*g in m/s²)
        T1:       Fundamental period (s) — used to tune frequency
        dt:       Time step (s)
        duration: Duration (s)

    Returns:
        (filepath, dt, npts) — file contains acceleration in m/s²

    NOTE: This is a simplified synthetic motion for proof-of-concept.
    For final analysis, replace with recorded or spectrum-compatible motions
    scaled to AS1170.4 design spectrum using SeismoMatch or similar.
    """
    t    = np.arange(0, duration, dt)
    freq = min(1.0 / T1, 4.0)      # dominant frequency near building period
    env  = np.sin(np.pi * t / duration)   # Hanning envelope
    accel = Z * G * env * np.sin(2 * np.pi * freq * t)

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False)
    np.savetxt(tmp.name, accel, fmt='%.8f')
    tmp.close()

    print(f"  Ground motion: PGA={Z}g ({Z*G:.2f} m/s²), "
          f"f_dom={freq:.2f} Hz, dt={dt}s, npts={len(t)}")
    return tmp.name, dt, len(t)


def run_time_history(model, gm_file: str, dt: float, npts: int,
                     T1: float, eigs: list) -> dict:
    """
    Run nonlinear time-history analysis.

    Args:
        model:   RCFrameModel instance (already built + gravity run)
        gm_file: Path to ground motion acceleration file (m/s²)
        dt:      Time step of ground motion file
        npts:    Number of points in ground motion
        T1:      Fundamental period (s) from eigenvalue analysis
        eigs:    Eigenvalue list from eigenvalue analysis

    Returns:
        Dictionary with time history arrays and peak EDPs.
    """
    p      = model.p
    nid    = model.node_id

    # ── Rayleigh damping — 5% at modes 1 and 2 ───────────────────────
    omega1 = abs(eigs[0]) ** 0.5
    omega2 = abs(eigs[1]) ** 0.5 if len(eigs) >= 2 else omega1 * 3
    xi     = 0.05
    a0     = xi * 2 * omega1 * omega2 / (omega1 + omega2)
    a1     = xi * 2 / (omega1 + omega2)
    ops.rayleigh(a0, 0.0, 0.0, a1)
    print(f"  Rayleigh: a0={a0:.5f}, a1={a1:.6f}  (5% at T1,T2)")

    # ── Ground motion loading ─────────────────────────────────────────
    ops.timeSeries('Path', 2, '-dt', dt, '-filePath', gm_file, '-factor', 1.0)
    ops.pattern('UniformExcitation', 2, 1, '-accel', 2)

    # ── Analysis setup ────────────────────────────────────────────────
    ops.system('UmfPack')
    ops.numberer('RCM')
    ops.constraints('Transformation')      # REQUIRED for equalDOF
    ops.test('NormDispIncr', 1.0e-8, 10, 0)
    ops.integrator('Newmark', 0.5, 0.25)   # average acceleration
    ops.algorithm('Newton')
    ops.analysis('Transient')

    # ── Time stepping ─────────────────────────────────────────────────
    dt_sub  = dt / 2           # sub-step for stability
    n_steps = int(npts * dt / dt_sub)

    time_h  = []
    disp_g  = []   # ground (fixed = 0, reference)
    disp_f  = []   # first floor master node
    disp_r  = []   # roof master node

    print(f"  Running: {n_steps} steps × dt={dt_sub:.4f}s")

    n_fail = 0
    for step in range(n_steps):
        ok = ops.analyze(1, dt_sub)

        if ok != 0:
            n_fail += 1
            # Fallback 1: KrylovNewton with smaller step
            ops.algorithm('KrylovNewton')
            ok = ops.analyze(1, dt_sub / 5)
            if ok != 0:
                # Fallback 2: ModifiedNewton with initial stiffness
                ops.test('NormDispIncr', 1.0e-6, 100, 0)
                ops.algorithm('ModifiedNewton', '-initial')
                ops.analyze(1, dt_sub / 10)
            # Reset to standard settings
            ops.algorithm('Newton')
            ops.test('NormDispIncr', 1.0e-8, 10, 0)

        time_h.append(ops.getTime())
        disp_g.append(ops.nodeDisp(nid[0][0], 1))
        disp_f.append(ops.nodeDisp(nid[1][0], 1))
        disp_r.append(ops.nodeDisp(nid[-1][0], 1))

    if n_fail > 0:
        print(f"  TH: COMPLETE ({n_fail} fallback steps used)")
    else:
        print("  TH: COMPLETE (all steps converged)")

    # Cleanup temp file
    try:
        os.remove(gm_file)
    except Exception:
        pass

    return {
        'time_h': np.array(time_h),
        'disp_g': np.array(disp_g),
        'disp_f': np.array(disp_f),
        'disp_r': np.array(disp_r),
    }
