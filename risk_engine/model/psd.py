"""Projection to the nearest positive-definite correlation matrix (§2.1).

§2.1 asks for a PSD projection and then relies on Cholesky factorising every
slice. Those two are not compatible: a positive *semi*-definite matrix may
have a zero eigenvalue, and its slices then fail Cholesky. The projection
here therefore targets positive *definiteness* with a strictly positive
eigenvalue floor, which does make the slice argument in §2.1 valid — a
principal submatrix of a PD matrix is PD.

Every correction is recorded with its magnitude (§2.1, §7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risk_engine.observability.metrics import METRICS, Metrics

DEFAULT_EIGENVALUE_FLOOR = 1e-8


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    corr: np.ndarray
    corrected: bool
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    frobenius_correction: float
    #: True when eigenvalue clipping alone left the matrix under the floor and
    #: the closed-form shift in step 3 had to finish the job.
    shift_applied: bool


def is_positive_definite(m: np.ndarray, floor: float = 0.0) -> bool:
    try:
        np.linalg.cholesky(m)
    except np.linalg.LinAlgError:
        return False
    return float(np.linalg.eigvalsh(m).min()) > floor


def project_to_correlation(
    matrix: np.ndarray,
    eigenvalue_floor: float = DEFAULT_EIGENVALUE_FLOOR,
    metrics: Metrics | None = None,
    label: str = "global",
) -> ProjectionResult:
    """Three deterministic steps, no iteration and no convergence risk.

    1. clip the eigenvalues to the floor -- the standard projection, and the
       one that stays closest to the input;
    2. renormalise the diagonal back to 1, which is what makes it a
       correlation matrix again;
    3. if step 2 pushed the smallest eigenvalue back under the floor, finish
       with the closed-form shift `(C + sI) / (1 + s)`, `s = (floor - lmin) /
       (1 - floor)`. That leaves the diagonal at exactly 1 and the smallest
       eigenvalue at exactly the floor, algebraically.

    Step 3 exists because steps 1 and 2 fight each other on badly
    rank-deficient inputs: renormalising re-shrinks the eigenvalue that
    clipping just lifted, and alternating the two can oscillate around the
    floor instead of converging. An iterative version of this function did
    exactly that on a rank-4 matrix in 12 dimensions and hit its iteration
    cap. The shift terminates by construction.
    """
    metrics = metrics or METRICS
    original = np.asarray(matrix, dtype=np.float64)
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        raise ValueError(f"expected a square matrix, got {original.shape}")
    if eigenvalue_floor <= 0 or eigenvalue_floor >= 1:
        raise ValueError("the eigenvalue floor must be in (0, 1)")

    m = 0.5 * (original + original.T)
    before = float(np.linalg.eigvalsh(m).min())
    already_correlation = np.allclose(np.diag(m), 1.0, atol=1e-12)
    if before >= eigenvalue_floor and already_correlation:
        return ProjectionResult(m, False, before, before, 0.0, False)

    vals, vecs = np.linalg.eigh(m)
    m = (vecs * np.maximum(vals, eigenvalue_floor)) @ vecs.T
    d = np.sqrt(np.diag(m))
    m = m / np.outer(d, d)
    m = 0.5 * (m + m.T)
    np.fill_diagonal(m, 1.0)

    lmin = float(np.linalg.eigvalsh(m).min())
    shift_applied = lmin < eigenvalue_floor
    if shift_applied:
        s = (eigenvalue_floor - lmin) / (1.0 - eigenvalue_floor)
        m = (m + s * np.eye(m.shape[0])) / (1.0 + s)
        np.fill_diagonal(m, 1.0)

    after = float(np.linalg.eigvalsh(m).min())
    frob = float(np.linalg.norm(m - original, ord="fro"))
    metrics.incr("psd_projection_corrections")
    metrics.psd_corrections.append(
        {
            "label": label,
            "min_eigenvalue_before": before,
            "min_eigenvalue_after": after,
            "frobenius_correction": frob,
            "shift_applied": float(shift_applied),
        }
    )
    return ProjectionResult(
        corr=m,
        corrected=True,
        min_eigenvalue_before=before,
        min_eigenvalue_after=after,
        frobenius_correction=frob,
        shift_applied=shift_applied,
    )
