"""Orakel für den QP-Löser (mzl_qp): andere Rechenwege als der Dual-NNLS-Löser.

1. KKT-Bedingungen an 300 Zufallszuständen (Stationarität mit Multiplikatoren aus kleinsten Quadraten über der Aktivmenge, Vorzeichen, Zulässigkeit).
2. Aufzählung der 3^N Aktivmengen-Muster beim Kasten-QP (nur Stellgrenzen; N = 1 bis 4): die einzige Lösung mit gültigem Muster.
3. Zulässigkeit der Dualform gegen die exakten Polygone F_N (Zustände nahe dem Rand ausgenommen).
4. Zielwert gegen SLSQP (SLSQP ist die ungenauere Seite: Toleranz 1e-5, nicht die Parameter allein vergleichen; ORACLE-PLAYBOOK §2).
Zufall nur über SplitMix64."""

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize

import mzl_loop as L
import mzl_poly as P
import mzl_qp as Q
import mzl_regions as R
from mzl_rng import SplitMix64


def _uniform_points(n, seed, lim=5.0):
    rng = SplitMix64(seed)
    return [np.array([(2 * rng.random() - 1) * lim, (2 * rng.random() - 1) * lim]) for _ in range(n)]


def _kkt_ok(prob, x0, U, tol=1e-7):
    H, C = prob.H, prob.C
    f = prob.Fm @ x0 + prob.g
    d = prob.d0 - prob.E @ x0
    if np.max(C @ U - d) > 1e-8:                       # Zulässigkeit (die Lösung nutzt 1e-9 Lockerung)
        return False
    g = H @ U + f
    act = np.where(d - C @ U < 1e-8)[0]
    if len(act) == 0:
        return np.abs(g).max() < tol
    mu, *_ = np.linalg.lstsq(C[act].T, -g, rcond=None)  # Stationarität g + Cᵀμ = 0 mit μ ≥ 0
    return mu.min() > -tol and np.abs(C[act].T @ mu + g).max() < tol


@pytest.mark.parametrize("N,R_", [(1, 0.1), (2, 10.0), (3, 0.1), (5, 10.0), (7, 0.1)])
def test_kkt_conditions_hold_at_random_feasible_states(N, R_):
    reg = R.regions(0.5, 5.0, 5.0)
    zul = P.to_float(reg.F(N))
    prob = L.cost_problem(N, "q_i_r01" if R_ == 0.1 else "q_i_r10", 0.5, 5.0, 5.0)
    n_checked = 0
    for x0 in _uniform_points(60, 100 + N):
        a, b = zul
        if np.min(b - a @ x0) < 1e-6:                  # Randpunkte (innen und außen) ausnehmen
            continue
        U, st = prob.solve(x0)
        assert st == "ok"
        assert _kkt_ok(prob, x0, U), (N, x0)
        n_checked += 1
    assert n_checked >= 15


def _box_oracle_by_enumeration(H, f, um):
    """Kasten-QP min ½UᵀHU + fᵀU, |u_k| ≤ um: alle 3^N Muster (unten, frei, oben) durchprobieren; die Lösung ist das Muster mit gültigem KKT-Punkt."""
    n = len(f)
    best = None
    for pattern in itertools.product((-1, 0, 1), repeat=n):
        free = [k for k in range(n) if pattern[k] == 0]
        bound = [k for k in range(n) if pattern[k] != 0]
        u = np.zeros(n)
        for k in bound:
            u[k] = pattern[k] * um
        if free:
            rhs = -(f[free] + H[np.ix_(free, bound)] @ u[bound]) if bound else -f[free]
            u[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
            if np.abs(u[free]).max() > um + 1e-12:
                continue
        g = H @ u + f
        if all((g[k] >= -1e-10 if pattern[k] == -1 else g[k] <= 1e-10) for k in bound):
            best = u if best is None else best                    # strikt konvex: genau ein gültiges Muster (bei Gleichstand an der Grenze dieselbe Lösung)
    return best


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_box_qp_against_enumeration_of_active_sets(N):
    um = 0.5
    prob = Q.build_problem(N, np.eye(2), 1.0, np.eye(2), [1e6, 1e6], um)         # Zustandsgrenzen so weit, dass nur die Stellgrenze wirkt
    for x0 in _uniform_points(40, 7 * N, lim=8.0):
        U, st = prob.solve(x0, slack=0.0)
        assert st == "ok"
        ref = _box_oracle_by_enumeration(prob.H, prob.Fm @ x0, um)
        assert ref is not None
        assert U == pytest.approx(ref, abs=1e-9), (N, x0)


@pytest.mark.parametrize("N", [1, 3, 6])
def test_feasibility_of_dual_form_against_exact_polygons(N, lp_membership):
    reg = R.regions(0.5, 5.0, 5.0)
    zul = P.to_float(reg.F(N))
    prob = L.cost_problem(N, "q_i_r10", 0.5, 5.0, 5.0)
    n_checked = 0
    for x0 in _uniform_points(120, 31 + N, lim=5.0):
        a, b = zul
        if np.min(np.abs(b - a @ x0)) < 1e-6:           # Randpunkte ausnehmen
            continue
        exakt = bool(np.all(a @ x0 <= b))
        U, st = prob.solve(x0, slack=0.0)
        assert (st == "ok") == exakt == lp_membership(x0, N, 0.5, 5.0, 5.0), (N, x0)
        n_checked += 1
    assert n_checked >= 60


@pytest.mark.parametrize("N,cost", [(2, "q_i_r10"), (3, "q_i_r01"), (4, "pos"), (3, "grenze")])
def test_objective_value_against_slsqp(N, cost):
    reg = R.regions(0.5, 5.0, 5.0)
    zul = P.to_float(reg.F(N))
    prob = L.cost_problem(N, cost, 0.5, 5.0, 5.0)
    n_checked = 0
    for x0 in _uniform_points(30, 13 + N, lim=5.0):
        a, b = zul
        if np.min(b - a @ x0) < 1e-3:
            continue
        U, st = prob.solve(x0, slack=0.0)
        f = prob.Fm @ x0 + prob.g
        obj = lambda u: 0.5 * u @ prob.H @ u + f @ u                         # noqa: E731
        d = prob.d0 - prob.E @ x0
        res = minimize(obj, np.clip(U, -0.5, 0.5) * 0.0, jac=lambda u: prob.H @ u + f, constraints=[{"type": "ineq", "fun": lambda u: d - prob.C @ u}],
                       method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
        if not res.success:
            continue
        assert obj(U) <= res.fun + 1e-8                                       # keine bessere Lösung als unsere (SLSQP ist höchstens gleich gut)
        assert obj(U) == pytest.approx(res.fun, abs=1e-5)
        n_checked += 1
    assert n_checked >= 5
