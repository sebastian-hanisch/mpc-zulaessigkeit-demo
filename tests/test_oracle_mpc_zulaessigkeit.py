"""Orakel-Tests des Kerns (ORACLE-PLAYBOOK §2, §5; DEMO-PLAYBOOK §12): die exakten Gebiete und der Regelkreis gegen unabhängige Rechenwege.

1. Exakte Polygone (Fourier-Motzkin, Brüche) gegen LP-Zulässigkeit (scipy HiGHS) an Zufallspunkten, für F_N und für K_N mit Terminalmenge; Randpunkte ausgenommen.
2. C ist steuerinvariant (exakt: C ⊆ Vorbild von C) und die größte solche Menge: eine kleine Vergrößerung ist es nicht mehr (Stichprobe an den Ecken).
3. Der Skalar: Formel f_N = 1 + 3/2^N gegen LP (tests/test_scalar.py) und gegen die Gebiete des allgemeinen Polygon-Codes in einer Dimension (Wagen mit u_max = 0 ist ein anderes System; hier der Vergleich Skalar-Formel gegen Schleife).
4. Die Bahn aus BBM Beispiel 12.1 (N = 3, Q = P = I, R = 10, |u| ≤ 1/2, |xᵢ| ≤ 5) gegen eine SLSQP-Bahn mit einer eigenen, getrennt aufgebauten Kostenfunktion (Band 1e-4; SLSQP ist die ungenauere Seite).
5. Der LQR-Gewinn: die gerundeten Brüche der Konstanten gegen die DARE (scipy); die Endkosten aus der Lyapunov-Gleichung gegen die DARE-Lösung (Band).
Zufall nur über SplitMix64 (Zufalls-Seeds fest)."""

from fractions import Fraction as F

import numpy as np
import pytest
import scipy.linalg as la
from scipy.optimize import minimize

import mzl_constants as C
import mzl_loop as L
import mzl_poly as P
import mzl_qp as Q
import mzl_regions as R
from conftest import A_EX, B_EX
from mzl_rng import SplitMix64


def _points(n, seed, s1=5.0, s2=5.0):
    rng = SplitMix64(seed)
    return [np.array([(2 * rng.random() - 1) * s1, (2 * rng.random() - 1) * s2]) for _ in range(n)]


@pytest.mark.parametrize("N", [0, 1, 2, 3, 4, 5, 6, 7])
def test_exact_F_N_against_lp_at_random_points(N, lp_membership):
    reg = R.regions(0.5, 5.0, 5.0)
    a, b = P.to_float(reg.F(N))
    n_checked = 0
    for x in _points(150, 500 + N):
        if np.min(np.abs(b - a @ x)) < 1e-7:
            continue                                                         # Randpunkte ausnehmen
        assert bool(np.all(a @ x <= b)) == lp_membership(x, N, 0.5, 5.0, 5.0), (N, x)
        n_checked += 1
    assert n_checked >= 100


@pytest.mark.parametrize("umax,s1,s2", [(0.25, 5.0, 5.0), (2.0, 10.0, 5.0), (1.0, 10.0, 10.0)])
def test_other_limits_against_lp(umax, s1, s2, lp_membership):
    reg = R.regions(umax, s1, s2)
    for N in (1, 3, 5):
        a, b = P.to_float(reg.F(N))
        for x in _points(80, 900 + N, s1, s2):
            if np.min(np.abs(b - a @ x)) < 1e-7:
                continue
            assert bool(np.all(a @ x <= b)) == lp_membership(x, N, umax, s1, s2), (umax, s1, s2, N, x)


@pytest.mark.parametrize("key,Ns", [("lqr1", (1, 2, 4)), ("lqr001", (1, 3)), ("null", (2, 3, 5))])
def test_exact_K_N_with_terminal_set_against_lp(key, Ns, lp_membership):
    T = R.terminal(key, 0.5, 5.0, 5.0)
    term = [(float(a1), float(a2), float(b)) for a1, a2, b in T.Xf]
    for N in Ns:
        a, b = P.to_float(T.K(N))
        n_in = 0
        for x in _points(250, 1300 + N):
            if np.min(np.abs(b - a @ x)) < 1e-7:
                continue
            exakt = bool(np.all(a @ x <= b))
            n_in += exakt
            assert exakt == lp_membership(x, N, 0.5, 5.0, 5.0, term=term), (key, N, x)
        assert n_in >= 1


def test_c_is_invariant_and_the_largest_such_set(reg_main, lp_membership):
    """Steuerinvarianz exakt (C ⊆ Vorbild von C). Maximalität mit der LP bei N = 12 (> N̄ = 6): ein Punkt kurz innerhalb jeder Ecke ist lösbar, ein Punkt kurz außerhalb
    (Ecke um 1/100 weiter vom Ursprung, nur wenn noch im Kasten) nicht."""
    assert P.subset(reg_main.Cset, P.step_back(reg_main.Cset, reg_main.X, A_EX, B_EX, F(1, 2)))
    n_out = 0
    for v in P.vertices(reg_main.Cset):
        x, y = float(v[0]), float(v[1])
        assert lp_membership((x * 0.99, y * 0.99), 12, 0.5, 5.0, 5.0)
        out = (x * 1.01, y * 1.01)
        if abs(out[0]) <= 5.0 and abs(out[1]) <= 5.0:
            assert not lp_membership(out, 12, 0.5, 5.0, 5.0), out
            n_out += 1
    assert n_out >= 10                                           # die meisten Ecken liegen nicht auf dem Kastenrand


def _slsqp_step(x0, N, Q_, R_, P_, umax, xmax):
    """Ein MPC-Schritt mit SLSQP: eigene Kostenfunktion durch Vorwärtsrechnung (kein Code aus mzl_qp); Rückgabe u₀ oder None."""
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([0.0, 1.0])

    def states(U):
        x = np.array(x0, dtype=float)
        out = []
        for u in U:
            x = A @ x + B * u
            out.append(x)
        return out

    def cost(U):
        x = np.array(x0, dtype=float)
        s = 0.0
        for u in U:
            s += x @ Q_ @ x + R_ * u * u
            x = A @ x + B * u
        return s + x @ P_ @ x

    def cons(U):
        return np.concatenate([xmax - np.abs(x) for x in states(U)])

    best = None
    for start in (np.zeros(N), np.full(N, -umax), np.full(N, umax)):
        r = minimize(cost, start, constraints=[{"type": "ineq", "fun": cons}], bounds=[(-umax, umax)] * N, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
        if r.success and np.all(cons(r.x) > -1e-7) and (best is None or r.fun < best.fun):
            best = r
    return None if best is None else best.x[0]


def test_bbm_example_12_1_trajectories_against_slsqp(reg_main):
    """Beispiel 12.1: aus (−4,5; 3) fährt der Regler (−1,5; 2,5) → (1; 2) und scheitert; aus (−4,5; 2) konvergiert die Bahn. Je Takt gegen die SLSQP-Bahn (Band 1e-4)."""
    I2 = np.eye(2)
    prob = L.cost_problem(3, "q_i_r10", 0.5, 5.0, 5.0)
    zul = P.to_float(reg_main.F(3))
    for x0, steps in (((-4.5, 3.0), 2), ((-4.5, 2.0), 12)):
        run = L.run_loop(prob, zul, np.array(x0), steps)
        x = np.array(x0)
        for k in range(steps):
            u = _slsqp_step(x, 3, I2, 10.0, I2, 0.5, 5.0)
            if u is None:
                break
            x = np.array([[1.0, 1.0], [0.0, 1.0]]) @ x + np.array([0.0, 1.0]) * u
            assert run.traj[k + 1] == pytest.approx(x, abs=1e-4), (x0, k)
    run1 = L.run_loop(prob, zul, np.array([-4.5, 3.0]), 10)
    assert run1.status == "verloren" and run1.k_stop == 2
    assert run1.traj[1] == pytest.approx([-1.5, 2.5], abs=1e-6) and run1.traj[2] == pytest.approx([1.0, 2.0], abs=1e-6)
    # kein zulässiger Plan von (1; 2) aus: SLSQP findet ebenfalls keinen (Handrechnung: x₁(3) ≥ 11/2)
    assert _slsqp_step([1.0, 2.0], 3, I2, 10.0, I2, 0.5, 5.0) is None
    run2 = L.run_loop(prob, zul, np.array([-4.5, 2.0]), 12)
    assert run2.status == "ende" and run2.traj[-1] == pytest.approx([-0.082, -0.040], abs=2e-3)         # Wert der Quelle: (−0,082; −0,040)


def test_rounded_lqr_gains_against_dare():
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.0], [1.0]])
    for key, r in C.LQR_R.items():
        Pm = la.solve_discrete_are(A, B, np.eye(2), np.array([[r]]))
        K = np.linalg.solve(np.array([[r]]) + B.T @ Pm @ B, B.T @ Pm @ A).ravel()
        gain = C.LQR_GAINS[key]
        assert [float(g) for g in gain] == pytest.approx([round(float(k), 3) for k in K], abs=1e-12)        # drei Nachkommastellen
        assert abs(float(gain[0]) - K[0]) < 5e-4 and abs(float(gain[1]) - K[1]) < 5e-4
        Pf = L.lyapunov_pf(gain, r)
        assert Pf == pytest.approx(Pm, abs=2e-3)                                                              # Endkosten aus dem gerundeten Gewinn ≈ DARE
    Pm1 = la.solve_discrete_are(A, B, np.eye(2), np.array([[1.0]]))
    assert L.lyapunov_pf(C.LQR_GAINS["1"], 1.0) == pytest.approx(Pm1, abs=5e-5)                                  # „auf 5 Stellen“ (README B8)
    assert L.lyapunov_pf(C.LQR_GAINS["1"], 1.0).ravel() == pytest.approx([2.94712, 2.36921, 2.36921, 4.61313], abs=5e-5)


def test_lqr_terminal_sets_are_invariant_under_the_closed_loop(reg_main):
    """O_LQR bleibt unter x⁺ = (A − BK)x in sich (exakt, Ecken) und erfüllt |Kx| ≤ u_max und den Kasten."""
    for key in ("lqr1", "lqr001"):
        T = R.terminal(key, 0.5, 5.0, 5.0)
        K = C.LQR_GAINS[C.TERM_R[key]]
        for v in P.vertices(T.Xf):
            assert abs(K[0] * v[0] + K[1] * v[1]) <= F(1, 2)
            nxt = (v[0] + v[1], v[1] - (K[0] * v[0] + K[1] * v[1]))
            assert P.contains(T.Xf, nxt)
