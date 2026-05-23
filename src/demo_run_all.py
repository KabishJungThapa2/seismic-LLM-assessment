"""
demo_run_all.py — Run all three case study buildings non-interactively.

This script demonstrates the fixed v2.0.0 pipeline on Building 1, 2, 3.
No API key required; uses demo mode keyword extraction.

Usage:
    cd src
    python demo_run_all.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractor import extract
from opensees_model import RCFrameModel
from analysis import generate_synthetic_gm, run_time_history, cleanup_gm_file
from compliance import static_base_shear, compute_edps, print_compliance_report


def assess(label: str, description: str) -> dict:
    print()
    print("=" * 65)
    print(f"  {label}")
    print("=" * 65)
    params = extract(description)
    model = RCFrameModel(params)
    model.build()
    model.run_gravity()
    model.assign_masses()
    T1, eigs = model.eigenvalue_analysis()
    V_static, Ch, T1_code, kpZ = static_base_shear(params, model.W_total)
    gm_file, dt, npts, gm_accel = generate_synthetic_gm(params['Z'], T1)
    th = run_time_history(model, gm_file, dt, npts, T1, eigs)
    edp = compute_edps(th, params, T1, model.M_floor, model.W_floor,
                       V_static, model.W_total, gm_accel=gm_accel, gm_dt=dt)
    cleanup_gm_file(gm_file)
    print_compliance_report(edp, params, T1, T1_code, kpZ)
    return {
        'label': label, 'T1': T1, 'T1_code': T1_code, 'kpZ': kpZ,
        'V_static': V_static, **edp,
    }


def main():
    descriptions = [
        ("Building 1 (Pre-1990)",
         "2-storey reinforced concrete frame, 12m x 8m, Newcastle, built 1985"),
        ("Building 2 (Post-1990)",
         "2-storey reinforced concrete frame, 12m x 8m, Newcastle, built 2000"),
        ("Building 3 (Post-2010)",
         "2-storey reinforced concrete frame, 12m x 8m, Newcastle, modern build 2015"),
    ]
    results = [assess(label, desc) for label, desc in descriptions]

    # Print summary table
    print("\n\n" + "=" * 86)
    print("  SUMMARY — Three Case Study Buildings")
    print("=" * 86)
    print(f"{'Building':<22} {'T1_FEM':>8} {'PIDR_max':>10} "
          f"{'θ_max':>8} {'V_dyn/V_st':>10} {'Perf':>5} {'HAZUS':>10}")
    print("-" * 86)
    for r in results:
        print(f"{r['label']:<22} {r['T1']:>8.3f} "
              f"{r['PIDR_max']*100:>9.3f}% {r['theta_max']:>8.4f} "
              f"{r['V_dyn_static_ratio']:>9.2f}x {r['performance_level']:>5} "
              f"{r['hazus_damage']:>10}")

    print()
    print("  All buildings: AS 1170.4:2007 + Amendment No. 2 (2018)")
    print("  Site Class De, Newcastle (Z=0.11, kpZ=0.11)")
    print("  Ground motion: synthetic sine-wave with Hanning envelope")
    print("  Standard reference: Standards Australia (2018)")
    print()
    print("  See CHANGELOG.md for the v2.0.0 bug fixes that produced these results.")


if __name__ == "__main__":
    main()
