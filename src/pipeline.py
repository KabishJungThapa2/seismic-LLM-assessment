# =============================================================================
# pipeline.py — LLM-Orchestrated Seismic Assessment Pipeline
# seismic-llm-assessment | UTS Engineering Graduate Project PG (42003)
# Kabish Jung Thapa (25631413) | Supervisor: Prof. Jianchun Li
#
# HOW TO RUN:
#   Google Colab:  paste notebooks/colab_pipeline.py into a cell
#   Local:         python src/pipeline.py  (Python 3.10, venv activated)
#
# PIPELINE STAGES:
#   1. User inputs building description (free text or guided Q&A)
#   2. LLM (GPT-4o or demo mode) extracts structural parameters as JSON
#   3. Human verification checkpoint — approve, edit, or reject
#   4. OpenSeesPy model built and analysed
#   5. AS1170.4 compliance checked
#   6. Results printed, plotted, saved as JSON
# =============================================================================

import sys
import os
import json
import textwrap
import re

# Add src directory to path when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import PARAM_BOUNDS
from extractor import extract, validate
from opensees_model import RCFrameModel
from analysis import generate_synthetic_gm, run_time_history
from compliance import static_base_shear, compute_edps, print_compliance_report


# =============================================================================
# STAGE 1: User Input
# =============================================================================

def get_user_input() -> str:
    """Prompt user for building description via free text or guided Q&A."""
    print()
    print("=" * 65)
    print("  SEISMIC VULNERABILITY ASSESSMENT")
    print("  LLM-Orchestrated Workflow | UTS EGP 42003")
    print("=" * 65)
    print()
    print("  How would you like to describe your building?")
    print()
    print("  A) Free-text description")
    print("     Example: '2-storey brick and concrete home in Newcastle,")
    print("     built around 1975, floor plan 12m × 8m'")
    print()
    print("  B) Guided Q&A (recommended for real buildings)")
    print()

    while True:
        mode = input("  Enter A or B: ").strip().upper()
        if mode in ('A', 'B'):
            break
        print("  Please enter A or B.")

    return _free_text() if mode == 'A' else _guided_qa()


def _free_text() -> str:
    print()
    print("  Describe your building. Include location, year built,")
    print("  storeys, approximate size, construction type.")
    print("  Press Enter twice when done.")
    print()
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    desc = " ".join(l for l in lines if l.strip())
    print(f"\n  ✓ Description received ({len(desc)} characters)")
    return desc


def _guided_qa() -> str:
    print()
    print("  Answer the questions below. Press Enter for default [value].")
    print()
    questions = [
        ("location",  "Location (city/suburb)",                   "Newcastle, NSW"),
        ("year",      "Year built (approximate)",                  "1985"),
        ("storeys",   "Number of storeys",                         "2"),
        ("system",    "Structural system (RC frame/timber/masonry)","RC frame"),
        ("length",    "Building length in metres (long side)",     "12"),
        ("width",     "Building width in metres (short side)",     "8"),
        ("use",       "Use (residential/commercial)",              "residential"),
        ("condition", "Condition (good/fair/poor)",                "fair"),
        ("features",  "Special features (or none)",               "none"),
    ]
    answers = {}
    for key, q, default in questions:
        ans = input(f"  {q} [{default}]: ").strip()
        answers[key] = ans if ans else default

    desc = (
        f"A {answers['storeys']}-storey {answers['system']} {answers['use']} "
        f"building in {answers['location']}, built approximately {answers['year']}. "
        f"Floor plan approximately {answers['length']}m × {answers['width']}m. "
        f"Condition: {answers['condition']}."
    )
    if answers['features'].lower() not in ('none', ''):
        desc += f" Features: {answers['features']}."
    desc += " Australian seismic zone — assess under AS1170.4."

    print()
    print("  ✓ Description:")
    print(textwrap.fill(f"  '{desc}'", width=65))
    return desc


# =============================================================================
# STAGE 3: Human Verification Checkpoint
# =============================================================================

def verification_checkpoint(params: dict) -> tuple[dict, bool]:
    """
    Display all extracted parameters and request human approval.
    Returns (params, approved).
    The engineer can approve, edit individual values, or reject entirely.
    """
    _print_params(params)
    warnings = validate(params)
    if warnings:
        print("  ── Validation Warnings ───────────────────────────────────")
        for w in warnings:
            print(f"  ⚠  {w}")
        print()

    print("  Options:")
    print("  [Y] Approve and run analysis")
    print("  [E] Edit a parameter")
    print("  [N] Reject — re-enter description")
    print()

    while True:
        choice = input("  Choice (Y/E/N): ").strip().upper()
        if choice == 'Y':
            print("\n  ✓ Approved. Starting analysis...\n")
            return params, True
        elif choice == 'E':
            params = _edit_param(params)
            return verification_checkpoint(params)
        elif choice == 'N':
            print("\n  ✗ Rejected. Please re-enter building description.\n")
            return params, False
        print("  Please enter Y, E, or N.")


def _print_params(params: dict):
    p = params
    print()
    print("=" * 65)
    print("  ⚠  HUMAN VERIFICATION CHECKPOINT")
    print("  Review all extracted parameters before analysis runs")
    print("=" * 65)
    print()
    print(f"  Building    : {p.get('building_name', '—')}")
    print(f"  Era         : {p.get('era', '—')}")
    print(f"  Confidence  : {p.get('confidence', 0):.0%}")
    print()
    print(f"  Geometry:")
    print(f"    Storeys       : {p['num_storeys']}")
    print(f"    Storey height : {p['storey_height']} m")
    print(f"    Bays × width  : {p['num_bays']} × {p['bay_width']} m")
    print(f"    Floor width   : {p['floor_width']} m")
    print()
    print(f"  Materials:")
    print(f"    Concrete f'c  : {p['fc']} MPa")
    print(f"    Steel fy      : {p['fy']} MPa")
    print()
    print(f"  Members:")
    print(f"    Columns       : {int(p['col_b']*1000)}×{int(p['col_h']*1000)} mm")
    print(f"    Beams         : {int(p['beam_b']*1000)}×{int(p['beam_h']*1000)} mm")
    print(f"    Column steel ρ: {p['col_rho']*100:.1f}%")
    print()
    print(f"  Seismic (AS1170.4):")
    print(f"    Z={p['Z']}, Site={p['site_class']}, μ={p['mu']}, Sp={p['Sp']}")
    print()
    if p.get('assumptions'):
        print("  Assumptions made:")
        for a in p['assumptions']:
            print(f"    ⚠  {a}")
        print()


def _edit_param(params: dict) -> dict:
    print()
    print("  Editable parameters: " + ", ".join(PARAM_BOUNDS.keys()))
    key = input("  Parameter name: ").strip()
    if key not in params:
        print(f"  '{key}' not found — no change made.")
        return params
    current = params[key]
    new_raw = input(f"  Current: {current}  →  New value: ").strip()
    try:
        params[key] = int(new_raw) if isinstance(current, int) else float(new_raw)
        params.setdefault('assumptions', []).append(
            f"User manually set {key} = {params[key]}")
        print(f"  ✓ {key} updated to {params[key]}")
    except ValueError:
        print(f"  Invalid value — keeping {key} = {current}")
    return params


# =============================================================================
# STAGE 4–5: Run analysis and check compliance
# =============================================================================

def run_analysis_pipeline(params: dict) -> dict:
    """Build model, run analysis, compute EDPs. Returns full results dict."""
    # Build model
    model = RCFrameModel(params)
    model.build()

    # Gravity + masses
    model.run_gravity()
    model.assign_masses()

    # Eigenvalue
    T1, eigs = model.eigenvalue_analysis()

    # Static benchmark
    V_static, Ch, T1_approx = static_base_shear(params, model.W_total)
    print(f"  Static V  = {V_static:.1f} kN  "
          f"(V/W = {V_static/model.W_total:.4f})")

    # Ground motion
    gm_file, dt, npts = generate_synthetic_gm(params['Z'], T1)

    # Time-history
    th = run_time_history(model, gm_file, dt, npts, T1, eigs)

    # EDPs and compliance
    edp = compute_edps(th, params, T1, model.M_floor,
                       V_static, model.W_total)

    return {
        'params':     params,
        'T1':         T1,
        'T1_approx':  T1_approx,
        'V_static':   V_static,
        'W_total':    model.W_total,
        'time_h':     th['time_h'],
        'disp_f':     th['disp_f'],
        'disp_r':     th['disp_r'],
        **edp,
    }


# =============================================================================
# STAGE 6: Output — plots and JSON report
# =============================================================================

def plot_results(r: dict, filename: str = 'seismic_results.png'):
    """Generate 4-panel results figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(r['params'].get('building_name', 'Seismic Assessment'),
                 fontsize=12, fontweight='bold')

    th  = r['time_h']
    lim = r['drift_limit'] * 100

    # Panel 1: Roof displacement
    ax = axes[0, 0]
    ax.plot(th, r['disp_r'] * 1000, color='steelblue', lw=1)
    ax.axhline(0, color='k', lw=0.4)
    ax.set(xlabel='Time (s)', ylabel='Roof Displacement (mm)',
           title='Roof Displacement Time History')
    ax.grid(True, alpha=0.3)

    # Panel 2: Drift time histories
    ax = axes[0, 1]
    ax.plot(th, r['drift_s1'] * 100, color='tomato',     lw=1, label='Storey 1')
    ax.plot(th, r['drift_s2'] * 100, color='darkorange', lw=1, label='Storey 2')
    ax.axhline( lim, color='red', ls='--', lw=1.5, label=f'Limit {lim:.0f}%')
    ax.axhline(-lim, color='red', ls='--', lw=1.5)
    ax.set(xlabel='Time (s)', ylabel='Inter-Storey Drift (%)',
           title='Inter-Storey Drift Time History')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Peak drift profile
    ax = axes[1, 0]
    colours = ['#d9534f' if v > r['drift_limit'] else '#5cb85c'
               for v in (r['PIDR1'], r['PIDR2'])]
    bars = ax.barh([1, 2], [r['PIDR1']*100, r['PIDR2']*100],
                   color=colours, edgecolor='white')
    ax.axvline(lim, color='red', ls='--', lw=1.5, label=f'Limit {lim:.0f}%')
    ax.set(xlabel='Peak Drift (%)', ylabel='Storey', title='Peak Drift Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, [r['PIDR1']*100, r['PIDR2']*100]):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}%", va='center', fontsize=9)

    # Panel 4: Summary text box
    ax = axes[1, 1]
    ax.axis('off')
    p   = r['params']
    txt = (
        f"RESULT SUMMARY\n{'─'*36}\n"
        f"Era:         {p.get('era','').capitalize()} RC Frame\n"
        f"f'c={p['fc']} MPa | fy={p['fy']} MPa\n"
        f"Z={r['params']['Z']} | Site {r['params']['site_class']}\n"
        f"{'─'*36}\n"
        f"T1 (FEM):    {r['T1']:.3f} s\n"
        f"T1 (code):   {r['T1_approx']:.3f} s\n"
        f"PIDR S1:     {r['PIDR1']*100:.3f}%\n"
        f"PIDR S2:     {r['PIDR2']*100:.3f}%\n"
        f"Limit:       {r['drift_limit']*100:.1f}%\n"
        f"{'─'*36}\n"
        f"V static:    {r['V_static']:.1f} kN\n"
        f"V dynamic:   {r['V_dynamic']:.1f} kN\n"
        f"{'─'*36}\n"
        f"Drift:  {'✓ PASS' if r['drift_pass'] else '✗ FAIL'}\n"
        f"RESULT: {'✓ COMPLIANT' if r['compliant'] else '✗ NON-COMPLIANT'}"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9.5,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Plot saved: {filename}")
    return filename


def save_json(r: dict, filename: str = 'seismic_report.json') -> str:
    """Save results as machine-readable JSON."""
    report = {
        "building_name":  r['params'].get('building_name'),
        "era":            r['params'].get('era'),
        "T1_FEM_s":       round(r['T1'], 4),
        "T1_code_s":      round(r['T1_approx'], 4),
        "PIDR_storey1_%": round(r['PIDR1'] * 100, 4),
        "PIDR_storey2_%": round(r['PIDR2'] * 100, 4),
        "PIDR_max_%":     round(r['PIDR_max'] * 100, 4),
        "drift_limit_%":  round(r['drift_limit'] * 100, 1),
        "drift_pass":     r['drift_pass'],
        "PFA_f1_g":       round(r['PFA_f1'] / 9.81, 4),
        "PFA_roof_g":     round(r['PFA_roof'] / 9.81, 4),
        "V_static_kN":    round(r['V_static'], 2),
        "V_dynamic_kN":   round(r['V_dynamic'], 2),
        "compliant":      r['compliant'],
        "parameters":     {k: v for k, v in r['params'].items()
                           if k not in ('assumptions', 'confidence')},
        "assumptions":    r['params'].get('assumptions', []),
    }
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  JSON saved: {filename}")
    return filename


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    """
    Full end-to-end pipeline.
    Stages: Input → Extract → Verify → Analyse → Report
    """
    print()
    print("╔" + "═" * 63 + "╗")
    print("║  SEISMIC VULNERABILITY ASSESSMENT — LLM-ORCHESTRATED     ║")
    print("║  UTS Engineering Graduate Project PG (42003)             ║")
    print("║  Kabish Jung Thapa | Supervisor: Prof. Jianchun Li       ║")
    print("╚" + "═" * 63 + "╝")

    # ── API key ────────────────────────────────────────────────────────
    print()
    print("  Optional: Enter your OpenAI API key to use GPT-4o.")
    print("  Press Enter to use demo mode (no API key needed).")
    print()
    api_key = input("  OpenAI API key: ").strip() or None
    if not api_key:
        print("  → Demo mode selected\n")

    # ── Main loop (retry up to 3 times on rejection) ───────────────────
    for attempt in range(3):
        if attempt:
            print(f"\n  Attempt {attempt + 1} of 3\n")

        # Stage 1: Input
        description = get_user_input()

        # Stage 2: LLM extraction
        print()
        print("  Extracting structural parameters...")
        params = extract(description, api_key)

        # Stage 3: Human verification
        params, approved = verification_checkpoint(params)
        if approved:
            break
        if attempt == 2:
            print("  Maximum attempts reached. Exiting.")
            return None
    else:
        return None

    # Stage 4–5: Analysis
    results = run_analysis_pipeline(params)

    # Stage 5: Print compliance report
    print_compliance_report(
        {k: results[k] for k in results if k not in
         ('params','time_h','disp_f','disp_r','drift_s1','drift_s2')},
        params, results['T1'], results['T1_approx']
    )

    # Stage 6: Save outputs
    plot_results(results)
    save_json(results)

    # Summary banner
    print()
    print("╔" + "═" * 63 + "╗")
    bname = results['params'].get('building_name', '')[:50]
    print(f"║  ASSESSMENT COMPLETE                                      ║")
    print(f"║  {bname:<62}║")
    status = ('✓ COMPLIANT' if results['compliant'] else '✗ NON-COMPLIANT')
    print(f"║  Result: {status:<54}║")
    pidr   = results['PIDR_max'] * 100
    print(f"║  Peak PIDR: {pidr:.3f}%  (limit 1.5%)                        ║")
    print("╚" + "═" * 63 + "╝")
    print()
    print("  Outputs saved:")
    print("  ├── seismic_results.png   (4-panel results plot)")
    print("  └── seismic_report.json   (machine-readable results)")

    return results


# =============================================================================

if __name__ == "__main__":
    run_pipeline()
