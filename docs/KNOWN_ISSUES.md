# Known Issues and Resolutions

## Issue 1: ARPACK Eigenvalue Solver Fails with equalDOF Constraints

**Severity:** Critical — prevents all eigenvalue analysis

**Symptom:**
```
ArpackSolver::Error with _saupd info = -9
Starting vector is zero.
WARNING StaticAnalysis::eigen() - EigenSOE failed in solve()
```

**When it occurs:** Any OpenSeesPy model that combines:
- Rigid diaphragm constraints via `ops.equalDOF(master, slave, 1)`
- Lumped mass assignment to master nodes only (correct implementation)

**Root cause:** ARPACK initialises with a zero starting vector. With equalDOF
constraints and master-node-only mass, the mass matrix is singular in the
slave DOFs. ARPACK's zero starting vector falls in the null space of this
singular mass matrix, causing immediate failure.

**Fix:**
```python
# WRONG — fails with equalDOF + master-node-only mass
eigenvalues = ops.eigen(num_modes)

# CORRECT — fullGenLapack handles singular mass matrices
eigenvalues = ops.eigen('-fullGenLapack', num_modes)

# Also required: wrap eigenvalues in abs() before sqrt
# (tiny negative values from numerical rounding cause domain errors)
omega1 = abs(eigenvalues[0]) ** 0.5
```

**Mass assignment (must be master nodes only):**
```python
for fi in range(1, num_storeys + 1):
    master_node = node_id[fi][0]
    ops.mass(master_node, M_floor, M_floor, 0.0)
    # Do NOT assign mass to slave nodes
```

**Why undocumented:** The OpenSeesPy official documentation does not mention
this incompatibility. It was discovered through systematic debugging in this
project and is expected to affect any researcher implementing rigid diaphragm
models with the default eigenvalue solver.

---

## Issue 2: constraints('Plain') Fails in Transient Analysis with equalDOF

**Symptom:** Analysis runs but produces incorrect results or warnings about
constraint enforcement.

**Fix:** Use `'Transformation'` for transient analysis with multi-point
constraints:
```python
# Gravity analysis (no transient DOF coupling) — Plain is OK
ops.constraints('Plain')

# Transient analysis — MUST use Transformation with equalDOF
ops.constraints('Transformation')
```

---

## Issue 3: OpenSeesPy Not Available on macOS Apple Silicon (M1/M2/M3)

**Symptom:** Architecture mismatch error when importing openseespy on
macOS with Apple Silicon chip.

**Fix:** Use Google Colab instead. The pre-compiled binaries available on
PyPI were built for x86_64 and are incompatible with ARM64 (Apple Silicon).
A community-built ARM64 package (`openseespy-mac-arm`) exists but has
compatibility issues on macOS 26.x.

**Workaround for local development:** Build OpenSeesPy from source (see
INSTALL.md for full instructions — approximately 30 minutes).

---

## Issue 4: fullGenLapack Solver Speed Warning

**Symptom:**
```
WARNING - the 'fullGenLapack' eigen solver is VERY SLOW.
Consider using the default eigen solver.
```

**Context:** This warning appears because fullGenLapack is O(n³) while
ARPACK is O(n·k) for k eigenvalues from an n-DOF system. For large models
the speed difference is significant.

**Impact for this project:** The buildings analysed have 12 nodes and 3 DOF
each = 36 DOF total. At this scale the speed difference is negligible
(milliseconds). The warning can be safely ignored.

For models with hundreds of nodes, consider alternative solutions:
- Use ARPACK with a non-zero starting vector (undocumented parameter)
- Remove equalDOF constraints and use 3D mass distribution instead
