"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mzl_poly as P  # noqa: E402
import mzl_regions as R  # noqa: E402

A_EX = [[F(1), F(1)], [F(0), F(1)]]
B_EX = [F(0), F(1)]


@pytest.fixture(scope="session")
def reg_main():
    """Hauptmessung: |u| ≤ 1/2, |xᵢ| ≤ 5 (N̄ = 6)."""
    return R.regions(0.5, 5.0, 5.0)


@pytest.fixture(scope="session")
def lp_membership():
    """Zulässigkeit eines Zustands für das MPC-Problem ohne Terminalmenge über eine LP (anderer Rechenweg als die exakten Polygone): lp(x0, N, umax, s1, s2, term=None)."""
    import numpy as np
    from scipy.optimize import linprog

    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([0.0, 1.0])

    def lp(x0, N, umax, s1, s2, term=None):
        if N == 0:
            return abs(x0[0]) <= s1 and abs(x0[1]) <= s2
        rows, rhs = [], []
        for k in range(1, N + 1):
            # x_k = A^k x0 + sum_j A^(k-1-j) B u_j
            Ak = np.linalg.matrix_power(A, k)
            G = np.zeros((2, N))
            for j in range(k):
                G[:, j] = np.linalg.matrix_power(A, k - 1 - j) @ B
            base = Ak @ np.asarray(x0, dtype=float)
            for i, s in enumerate((s1, s2)):
                rows.append(G[i]); rhs.append(s - base[i])
                rows.append(-G[i]); rhs.append(s + base[i])
            if k == N and term is not None:
                for a1, a2, b in term:
                    rows.append(a1 * G[0] + a2 * G[1]); rhs.append(b - (a1 * base[0] + a2 * base[1]))
        r = linprog(np.zeros(N), A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(-umax, umax)] * N, method="highs")
        return r.status == 0

    return lp
