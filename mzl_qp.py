"""Kleiner QP-Löser und das MPC-Problem in verdichteter Form (nur numpy und scipy).

Löser: min ½ UᵀH U + fᵀU unter C U ≤ d mit H positiv definit, über die Dualform (Lawson und Hanson, „Least distance programming“ per NNLS):
H = L Lᵀ, M = C L⁻ᵀ, b = d + M L⁻¹ f; gesucht ist die Lösung z kleinster Norm von M z ≤ b, danach U = L⁻ᵀ (z − L⁻¹ f). Die Dualform löst
`scipy.optimize.nnls` (aktive Menge, endlich). Das ist ein eigener, kleiner Löser, kein Hochleistungslöser; er ist gegen die KKT-Bedingungen und gegen
Aufzählung der Aktivmengen geprüft (`tests/test_oracle_qp.py`). Neu geschrieben nach dem Muster der Vormessung `rg_v16_qp.py` (nur gelesen).

Das Problem: x_(k+1) = A x_k + B u_k, Kosten Σ_(k<N) (x_kᵀ Q x_k + R u_k²) + x_NᵀP_f x_N (+ linearer Anteil Σ_(k=1..N) wᵀ x_k),
Nebenbedingungen |u_k| ≤ u_max (k < N), |x_k,i| ≤ s_i für die vorhergesagten Zustände k = 1 … N (der Anfangszustand zählt nicht; in dieser Lesart
reproduziert die Vormessung die Quelle Borrelli/Bemporad/Morari, Beispiel 12.1) und optional x_N ∈ X_f (Ungleichungen a·x_N ≤ b)."""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
from scipy.optimize import nnls

A_CAR = np.array([[1.0, 1.0], [0.0, 1.0]])       # Wagen: Doppelintegrator, Ts = 1
B_CAR = np.array([[0.0], [1.0]])


class QP:
    """min ½ UᵀH U + fᵀU unter C U ≤ d für feste H und C; solve(f, d) für wechselnde f und d (der Zustand geht nur in f und d ein)."""

    def __init__(self, H, C):
        self.H = np.asarray(H, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.n = self.H.shape[0]
        self.m = self.C.shape[0]
        self.L = np.linalg.cholesky(self.H)
        self.Linv = la.solve_triangular(self.L, np.eye(self.n), lower=True)
        self.M = self.C @ self.Linv.T

    def solve(self, f, d):
        """Rückgabe (U, "ok") oder (None, "unzulaessig"). Unzulässig heißt: die Dualform hat kein Residuum 0 (Schwelle 1e-12), also gibt es kein U mit C U ≤ d."""
        n = self.n
        g = self.Linv @ f
        b = d + self.M @ g
        if np.all(b >= 0):                      # z = 0 ist zulässig: der Minimierer ist der unbeschränkte
            z = np.zeros(n)
        else:
            E = np.vstack((-self.M.T, -b[None, :]))
            rhs = np.zeros(n + 1)
            rhs[n] = 1.0
            lam, rn = nnls(E, rhs, maxiter=50 * (self.m + n))
            r = E @ lam - rhs
            if rn < 1e-12 or abs(r[n]) < 1e-12:
                return None, "unzulaessig"
            z = -r[:n] / r[n]
        return self.Linv.T @ (z - g), "ok"


@dataclass
class Problem:
    """Das verdichtete MPC-Problem eines Horizonts: x_k = Φ_k x0 + Γ_k U; f = Fm x0 + g und d = d0 − E x0."""

    N: int
    qp: QP
    Fm: np.ndarray
    g: np.ndarray
    d0: np.ndarray
    E: np.ndarray
    Phi: list
    Gam: list
    H: np.ndarray
    C: np.ndarray
    Qm: np.ndarray
    R: float
    Pf: np.ndarray
    lin: tuple | None

    def solve(self, x0, slack=1e-9):
        """Optimale Stellfolge ab x0 oder (None, "unzulaessig"). `slack` lockert die Nebenbedingungen um 1e-9 (Zustände auf dem Rand der Menge)."""
        x0 = np.asarray(x0, dtype=float)
        return self.qp.solve(self.Fm @ x0 + self.g, self.d0 - self.E @ x0 + slack)

    def cost(self, x0, U):
        """Wert der Kostenfunktion ohne den linearen Anteil (Stufenkosten plus Endkosten) für die Stellfolge U ab x0."""
        x = np.asarray(x0, dtype=float)
        A, B = self.Phi[1], self.Gam[1][:, 0]
        s = 0.0
        for k in range(self.N):
            s += float(x @ self.Qm @ x) + self.R * float(U[k]) ** 2
            x = A @ x + B * U[k]
        return s + float(x @ self.Pf @ x)


def build_problem(N, Qm, R, Pf, xmax, umax, term=None, lin=None, A=None, B=None):
    """MPC-Problem des Horizonts N. `xmax`: Betragsgrenzen je Zustand (Liste), `term`: Liste (a1, a2, b) für a·x_N ≤ b oder None, `lin`: Gewichte w
    des linearen Kostenanteils Σ_(k=1..N) wᵀx_k oder None."""
    A = A_CAR if A is None else np.asarray(A, dtype=float)
    B = B_CAR if B is None else np.asarray(B, dtype=float).reshape(-1, 1)
    nx = A.shape[0]
    Qm = np.asarray(Qm, dtype=float)
    Pf = np.asarray(Pf, dtype=float)
    xmax = np.broadcast_to(np.asarray(xmax, dtype=float), (nx,))
    Phi = [np.eye(nx)]
    Gam = [np.zeros((nx, N))]
    for k in range(1, N + 1):
        Phi.append(np.linalg.matrix_power(A, k))
        G = np.zeros((nx, N))
        for j in range(k):
            G[:, j:j + 1] = np.linalg.matrix_power(A, k - 1 - j) @ B
        Gam.append(G)
    Hc = R * np.eye(N)
    Fc = np.zeros((N, nx))
    for k in range(1, N + 1):
        W = Pf if k == N else Qm
        Hc = Hc + Gam[k].T @ W @ Gam[k]
        Fc = Fc + Gam[k].T @ W @ Phi[k]
    H, Fm = 2 * Hc, 2 * Fc
    rows, rhs, ex = [], [], []
    for k in range(N):
        e = np.zeros((1, N))
        e[0, k] = 1.0
        rows += [e, -e]
        rhs += [umax, umax]
        ex += [np.zeros((1, nx)), np.zeros((1, nx))]
    for k in range(1, N + 1):
        for s in (1.0, -1.0):
            for i in range(nx):
                rows.append(s * Gam[k][i:i + 1, :])
                rhs.append(float(xmax[i]))
                ex.append(s * Phi[k][i:i + 1, :])
    if term is not None:
        for t in term:
            a, b = np.array([t[:-1]], dtype=float), float(t[-1])
            rows.append(a @ Gam[N])
            rhs.append(b)
            ex.append(a @ Phi[N])
    if N == 0:                                  # kein Optimierungsproblem: F_0 ist der Kasten selbst
        raise ValueError("Horizont N muss mindestens 1 sein")
    C = np.vstack(rows)
    g = np.zeros(N) if lin is None else sum(G.T @ np.asarray(lin, dtype=float) for G in Gam[1:])
    return Problem(N, QP(H, C), Fm, g, np.array(rhs, dtype=float), np.vstack(ex), Phi, Gam, H, C, Qm, float(R), Pf, None if lin is None else tuple(lin))
