# =============================================================================
# opensees_model.py — OpenSeesPy model builder
# seismic-llm-assessment | UTS Engineering Graduate Project PG (42003)
#
# Builds a 2D nonlinear RC plane frame model from a parameter dictionary.
# Uses fibre sections with Concrete01 + Steel01 material models.
#
# CRITICAL TECHNICAL NOTES:
#   1. fullGenLapack solver required for eigenvalue analysis when equalDOF
#      constraints are present — ARPACK fails with "Starting vector is zero"
#   2. Masses assigned ONLY to master nodes — slave nodes get zero mass
#   3. constraints('Transformation') required for transient analysis
#      with multi-point constraints (equalDOF)
#
# Reference: OpenSeesPy RC Frame Earthquake example
#   openseespydoc.readthedocs.io/en/latest/src/RCFrameEarthquake.html
# =============================================================================

import openseespy.opensees as ops
import numpy as np
from config import G, COVER, ERA_DEFAULTS


class RCFrameModel:
    """
    2D nonlinear RC plane frame model.

    Node numbering: (floor_idx+1)*10 + (col_idx+1)
      Ground:  11, 12, 13, 14  (fixed)
      Floor 1: 21, 22, 23, 24  (master = 21)
      Roof:    31, 32, 33, 34  (master = 31)

    Elements:
      Columns: nonlinearBeamColumn, fibre section, PDelta transform
      Beams:   nonlinearBeamColumn, fibre section, Linear transform
    """

    def __init__(self, params: dict):
        self.p         = params
        self.node_id   = []    # node_id[floor][column]
        self.M_floor   = None  # seismic mass per floor (kN·s²/m)
        self.W_floor   = None  # seismic weight per floor (kN)
        self.W_total   = None  # total seismic weight (kN)
        self.T1_approx = None  # AS1170.4 approximate period

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self):
        """Build the complete OpenSees model. Call before any analysis."""
        p = self.p
        self._compute_derived()
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        self._build_nodes()
        self._apply_boundary_conditions()
        self._apply_diaphragm_constraints()
        self._define_materials()
        self._define_sections()
        self._define_elements()
        print(f"  Model built: {self._n_nodes()} nodes, "
              f"{self._n_columns() + self._n_beams()} elements "
              f"({self._n_columns()} col + {self._n_beams()} beam)")

    def run_gravity(self):
        """Apply gravity loads and run static analysis."""
        p          = self.p
        w_frame    = (p['dead_load'] + p['live_load']) * p['floor_width'] / 2
        P_interior = w_frame * p['bay_width']
        P_exterior = w_frame * p['bay_width'] / 2

        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)

        for fi in range(1, p['num_storeys'] + 1):
            for ci, nid in enumerate(self.node_id[fi]):
                P = -P_exterior if (ci == 0 or ci == p['num_bays']) \
                    else -P_interior
                ops.load(nid, 0.0, P, 0.0)

        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Plain')
        ops.integrator('LoadControl', 0.1)
        ops.algorithm('Newton')
        ops.analysis('Static')
        ok = ops.analyze(10)
        if ok != 0:
            print("  ⚠ Gravity analysis did not fully converge")
        else:
            print("  Gravity analysis: CONVERGED")
        ops.loadConst('-time', 0.0)
        return ok

    def assign_masses(self):
        """
        Assign lumped seismic masses to master nodes only.

        CRITICAL: Do NOT assign mass to slave nodes. With equalDOF
        constraints, mass on slave nodes creates a singular/inconsistent
        mass matrix that causes the ARPACK eigenvalue solver to fail
        with 'Starting vector is zero'. fullGenLapack handles this but
        master-node-only mass is still the correct physical model.
        """
        p = self.p
        for fi in range(1, p['num_storeys'] + 1):
            master = self.node_id[fi][0]
            ops.mass(master, self.M_floor, self.M_floor, 0.0)
        print(f"  Masses assigned: {self.M_floor:.3f} kN·s²/m per floor "
              f"(master nodes only)")

    def eigenvalue_analysis(self):
        """
        Extract natural frequencies using fullGenLapack solver.

        IMPORTANT: Use '-fullGenLapack' not the default ARPACK.
        ARPACK fails with equalDOF constraints (singular mass matrix).
        fullGenLapack handles multi-point constraints correctly.

        Returns:
            T1: fundamental period (s)
            eigs: list of eigenvalues [ω₁², ω₂²]
        """
        p    = self.p
        eigs = ops.eigen('-fullGenLapack', p['num_storeys'])

        omega1 = abs(eigs[0]) ** 0.5
        T1     = 2 * np.pi / omega1
        omega2 = abs(eigs[1]) ** 0.5 if p['num_storeys'] >= 2 else omega1 * 3
        T2     = 2 * np.pi / omega2

        print(f"  Eigenvalue analysis (fullGenLapack):")
        print(f"    ω₁ = {omega1:.3f} rad/s  →  T1 = {T1:.3f} s "
              f"(code approx: {self.T1_approx:.3f} s)")
        print(f"    ω₂ = {omega2:.3f} rad/s  →  T2 = {T2:.3f} s")
        return T1, eigs

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_derived(self):
        p           = self.p
        floor_area  = p['num_bays'] * p['bay_width'] * p['floor_width']
        self.W_floor = (p['dead_load'] + 0.3 * p['live_load']) * floor_area
        self.W_total = self.W_floor * p['num_storeys']
        self.M_floor = self.W_floor / G
        Hn           = p['num_storeys'] * p['storey_height']
        self.T1_approx = 0.075 * Hn ** 0.75

    def _build_nodes(self):
        p        = self.p
        x_coords = [j * p['bay_width'] for j in range(p['num_bays'] + 1)]
        y_coords = [i * p['storey_height'] for i in range(p['num_storeys'] + 1)]
        self.node_id = []
        for fi, y in enumerate(y_coords):
            row = []
            for ci, x in enumerate(x_coords):
                nid = (fi + 1) * 10 + (ci + 1)
                ops.node(nid, x, y)
                row.append(nid)
            self.node_id.append(row)

    def _apply_boundary_conditions(self):
        for nid in self.node_id[0]:
            ops.fix(nid, 1, 1, 1)

    def _apply_diaphragm_constraints(self):
        """Rigid diaphragm: constrain slave nodes to master (X-DOF only)."""
        p = self.p
        for fi in range(1, p['num_storeys'] + 1):
            master = self.node_id[fi][0]
            for slave in self.node_id[fi][1:]:
                ops.equalDOF(master, slave, 1)

    def _define_materials(self):
        p     = self.p
        fc_kN = p['fc'] * 1000
        fy_kN = p['fy'] * 1000
        Es_kN = 200000.0 * 1000

        # Get confinement strains from era defaults
        era_key = p.get('era', 'pre-1990')
        era_d   = ERA_DEFAULTS.get(era_key, ERA_DEFAULTS['pre-1990'])
        epsc0   = era_d['epsc0_core']
        epsU    = era_d['epsU_core']

        # Core concrete (confined)
        ops.uniaxialMaterial('Concrete01', 1,
                             -fc_kN, epsc0, -0.2*fc_kN, epsU)
        # Cover concrete (unconfined — spalls)
        ops.uniaxialMaterial('Concrete01', 2,
                             -fc_kN, -0.002, 0.0, -0.004)
        # Steel (bilinear with 1% strain hardening)
        ops.uniaxialMaterial('Steel01', 3, fy_kN, Es_kN, 0.01)

    def _define_sections(self):
        p    = self.p
        Ac   = p['col_b'] * p['col_h']
        Asc  = p['col_rho'] * Ac
        Ab   = p['beam_b'] * p['beam_h']
        Ast  = p['beam_rho_t'] * Ab
        Asc2 = p['beam_rho_c'] * Ab

        # Column section (tag=1)
        cy = p['col_h'] / 2 - COVER
        cz = p['col_b'] / 2 - COVER
        ops.section('Fiber', 1)
        ops.patch('rect', 1, 10, 10, -cy, -cz, cy, cz)
        ops.patch('rect', 2, 10, 2,   cy,      -p['col_b']/2, p['col_h']/2, p['col_b']/2)
        ops.patch('rect', 2, 10, 2,  -p['col_h']/2, -p['col_b']/2, -cy, p['col_b']/2)
        ops.patch('rect', 2,  2, 10, -cy, -p['col_b']/2, cy, -cz)
        ops.patch('rect', 2,  2, 10, -cy,  cz,           cy,  p['col_b']/2)
        As_bar = max(Asc / 6, 1e-5)
        ops.layer('straight', 3, 3, As_bar, -cy, -cz, -cy, cz)
        ops.layer('straight', 3, 3, As_bar,  cy, -cz,  cy, cz)

        # Beam section (tag=2)
        by = p['beam_h'] / 2 - COVER
        bz = p['beam_b'] / 2 - COVER
        ops.section('Fiber', 2)
        ops.patch('rect', 1, 10, 10, -by, -bz, by, bz)
        ops.patch('rect', 2, 10, 2,   by,      -p['beam_b']/2, p['beam_h']/2, p['beam_b']/2)
        ops.patch('rect', 2, 10, 2,  -p['beam_h']/2, -p['beam_b']/2, -by, p['beam_b']/2)
        ops.layer('straight', 3, 3, Ast/3,  -by, -bz, -by, bz)
        ops.layer('straight', 3, 3, Asc2/3,  by, -bz,  by, bz)

    def _define_elements(self):
        p   = self.p
        ops.geomTransf('PDelta', 1)   # columns
        ops.geomTransf('Linear', 2)   # beams
        eid = 100
        for fi in range(p['num_storeys']):
            for ci in range(p['num_bays'] + 1):
                ops.element('nonlinearBeamColumn', eid,
                            self.node_id[fi][ci],
                            self.node_id[fi+1][ci], 5, 1, 1)
                eid += 1
        for fi in range(1, p['num_storeys'] + 1):
            for ci in range(p['num_bays']):
                ops.element('nonlinearBeamColumn', eid,
                            self.node_id[fi][ci],
                            self.node_id[fi][ci+1], 5, 2, 2)
                eid += 1

    def _n_nodes(self):
        p = self.p
        return (p['num_bays'] + 1) * (p['num_storeys'] + 1)

    def _n_columns(self):
        p = self.p
        return (p['num_bays'] + 1) * p['num_storeys']

    def _n_beams(self):
        p = self.p
        return p['num_bays'] * p['num_storeys']
