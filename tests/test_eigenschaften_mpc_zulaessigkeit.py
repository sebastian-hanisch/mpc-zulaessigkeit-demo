"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2) und die Gegenprobe für Absolutaussagen (§6): Aussagen über Gebiete und Regelkreis, die ohne Kenntnis des richtigen Ergebnisses gelten müssen.

Gebiete exakt (Brüche), Regelkreise mit festem Zufalls-Seed (SplitMix64). Bedingungen stehen im Docstring jedes Tests."""

from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest

import mzl_constants as C
import mzl_loop as L
import mzl_poly as P
import mzl_qp as Q
import mzl_regions as R
from conftest import A_EX, B_EX
from mzl_rng import SplitMix64

COMBOS = [(u, s1, s2) for u in C.UMAX_OPTIONS for s1 in C.S_OPTIONS for s2 in C.S_OPTIONS]


@pytest.mark.parametrize("umax,s1,s2", COMBOS)
def test_F_N_is_nested_and_contains_C(umax, s1, s2):
    """F_(N+1) ⊆ F_N, und C ⊆ F_N für jedes N (alle 20 Kombinationen der Reglerstufen)."""
    reg = R.regions(umax, s1, s2)
    assert reg.n_bar is not None
    for n in range(len(reg.seq) - 1):
        assert P.subset(reg.seq[n + 1], reg.seq[n])
        assert P.subset(reg.Cset, reg.seq[n])
    assert P.same(P.step_back(reg.Cset, reg.X, A_EX, B_EX, reg.umax), reg.Cset)         # C ist ein Fixpunkt der Rückwärtsrechnung


@pytest.mark.parametrize("key", ["lqr1", "lqr001", "null"])
def test_K_N_grows_nested_and_stays_in_C(key):
    """Für steuerinvariantes X_f (O_LQR, {0}) wächst K_N mit N geschachtelt und liegt in C."""
    reg = R.regions(0.5, 5.0, 5.0)
    T = R.terminal(key, 0.5, 5.0, 5.0)
    for n in range(len(T.seq) - 1):
        assert P.subset(T.K(n), T.K(n + 1))
        assert P.subset(T.K(n), reg.Cset)


@pytest.mark.parametrize("c", [2, 3, F(1, 2)])
def test_scaling_all_limits_scales_the_regions(c):
    """F_N(c·Grenzen) = c·F_N(Grenzen) für c > 0: Gleichungen und Ungleichungen des Problems sind positiv homogen in (x, u, Grenzen)."""
    base = R.regions(0.5, 5.0, 5.0)
    big = R.regions(float(F(1, 2) * c), float(5 * c), float(5 * c))
    assert big.n_bar == base.n_bar
    for n in range(len(base.seq)):
        assert P.same(big.seq[n], P.scaled(base.seq[n], c))


def test_mirror_symmetry():
    """Spiegelung x → −x: die Gebiete sind punktsymmetrisch (die Dynamik ist linear, die Grenzen symmetrisch)."""
    for umax, s1, s2 in ((0.5, 5.0, 5.0), (0.25, 10.0, 5.0)):
        reg = R.regions(umax, s1, s2)
        for c in reg.seq:
            vs = P.vertices(c)
            assert sorted(vs) == sorted((-v[0], -v[1]) for v in vs)


def test_area_of_C_grows_with_the_limits():
    """Mit größerer Stellgrenze und größeren Zustandsgrenzen wird C größer (Mengeninklusion, exakt)."""
    for s1, s2 in product(C.S_OPTIONS, C.S_OPTIONS):
        regs = [R.regions(u, s1, s2) for u in C.UMAX_OPTIONS]
        for small, big in zip(regs, regs[1:]):
            assert P.subset(small.Cset, big.Cset) and P.area_of(small.Cset) < P.area_of(big.Cset)
    for u in C.UMAX_OPTIONS:
        assert P.subset(R.regions(u, 5.0, 5.0).Cset, R.regions(u, 10.0, 5.0).Cset)
        assert P.subset(R.regions(u, 5.0, 5.0).Cset, R.regions(u, 5.0, 10.0).Cset)


def test_n_bar_does_not_grow_with_the_input_limit():
    """Eine größere Stellgrenze braucht nicht mehr Takte (N̄ fällt oder bleibt): 9, 6, 4, 3, 1 bei s = 5; 13, 9, 5, 3, 1 bei s = 10."""
    for s in C.S_OPTIONS:
        nb = [R.regions(u, s, 5.0).n_bar for u in C.UMAX_OPTIONS]
        assert all(a >= b for a, b in zip(nb, nb[1:]))


def test_cost_scaling_does_not_change_the_control():
    """c·Q, c·R, c·P ändern die Optimierung nicht: gleiche erste Stellgröße (c = 3,7; 40 Startzustände)."""
    rng = SplitMix64(21)
    c = 3.7
    for N in (2, 4):
        p1 = Q.build_problem(N, np.eye(2), 1.0, np.eye(2), [5.0, 5.0], 0.5)
        p2 = Q.build_problem(N, c * np.eye(2), c * 1.0, c * np.eye(2), [5.0, 5.0], 0.5)
        reg = R.regions(0.5, 5.0, 5.0)
        a, b = P.to_float(reg.F(N))
        for _ in range(40):
            x = np.array([(2 * rng.random() - 1) * 5, (2 * rng.random() - 1) * 5])
            if np.min(b - a @ x) < 1e-6:
                continue
            u1, s1 = p1.solve(x)
            u2, s2 = p2.solve(x)
            assert s1 == s2 == "ok" and u1[0] == pytest.approx(u2[0], abs=1e-7)


def test_cost_choice_changes_the_trajectory_but_not_the_region():
    """Das Gebiet hängt nur an Grenzen und N, die Bahn an den Kosten (Zweig-Test: die Kosten unterscheiden sich auf mindestens einem Start)."""
    reg = R.regions(0.5, 5.0, 5.0)
    zul = P.to_float(reg.F(3))
    x0 = np.array([-3.0, 1.0])
    ends = {}
    for key in C.COST_KEYS:
        ends[key] = L.run_loop(L.cost_problem(3, key, 0.5, 5.0, 5.0), zul, x0, 6).traj[2]
    assert len({tuple(np.round(v, 6)) for v in ends.values()}) == len(C.COST_KEYS)


def test_larger_horizon_does_not_make_the_exit_from_C_more_frequent():
    """„Zur Grenze treibend“, 300 Starts je N in F_N (Zufalls-Seed 716): Anteil der Starts in C, deren Bahn C verlässt, fällt nicht, wenn N von 2 bis 7 wächst (Zählung)."""
    shares = []
    for N in range(2, 8):
        res = L.experiment_falls(N, "grenze", 300, 716, 0.5, 5.0, 5.0)
        c = res.counts()
        shares.append(c["austritt"] / c["in_c"])
    assert all(b <= a + 0.02 for a, b in zip(shares, shares[1:])), shares
    assert shares[0] > 0.5 and shares[-1] == 0.0


def test_with_terminal_set_every_run_stays_in_K_N():
    """Mit steuerinvarianter Terminalmenge bleibt jede Bahn aus K_N in K_N (nicht nur in F_N): geprüft an jedem Takt gegen die exakte Menge."""
    T = R.terminal("lqr1", 0.5, 5.0, 5.0)
    for N in (2, 5):
        from mzl_loop import term_setup

        prob, kfloat, _ = term_setup("lqr1", N, 0.5, 5.0, 5.0)
        pts = L.sample_starts(kfloat, (5.0, 5.0), 40, 9)
        for x0 in pts:
            run = L.run_loop(prob, kfloat, x0, 30, conv_tol=1e-7)
            assert run.status != "verloren"
            assert all(np.all(kfloat[0] @ x <= kfloat[1] + 1e-6) for x in run.traj)


# --- Gegenprobe für Absolutaussagen (ORACLE-PLAYBOOK §6) --------------------------------------------------------------------------------------

def _first_control_feasible(x, u0, N, umax=0.5, s=5.0):
    """Gibt es einen zulässigen Plan der Länge N mit erster Stellgröße u0? LP mit u₀ festgenagelt (anderer Rechenweg als die Gebiete)."""
    from scipy.optimize import linprog

    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([0.0, 1.0])
    rows, rhs = [], []
    for k in range(1, N + 1):
        G = np.zeros((2, N))
        for j in range(k):
            G[:, j] = np.linalg.matrix_power(A, k - 1 - j) @ B
        base = np.linalg.matrix_power(A, k) @ x
        for i in range(2):
            rows.append(G[i]); rhs.append(s - base[i] + 1e-9)
            rows.append(-G[i]); rhs.append(s + base[i] + 1e-9)
    bounds = [(u0, u0)] + [(-umax, umax)] * (N - 1)
    return linprog(np.zeros(N), A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds, method="highs").status == 0


def test_from_N_bar_plus_one_every_feasible_first_control_keeps_the_state_in_C():
    """„Ab N = 7 hält jede zulässige erste Stellgröße den Zustand in C“ (Voraussetzung: dieses Modell, |u| ≤ 1/2, |xᵢ| ≤ 5). 3 000 Zufallspaare (x in C, u₀ gleichverteilt in [−1/2; 1/2],
    nicht vom Optimierer gewählt): ist u₀ erste Stellgröße eines zulässigen Plans der Länge 7 (LP), liegt A x + B u₀ in C. Zusätzlich die exakte Rechnung: Austrittsfläche 0 ab N = 7."""
    reg = R.regions(0.5, 5.0, 5.0)
    cf = P.to_float(reg.Cset)
    rng = SplitMix64(2026)
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([0.0, 1.0])
    n_pairs = n_feasible = 0
    while n_pairs < 3000:
        x = np.array([(2 * rng.random() - 1) * 5, (2 * rng.random() - 1) * 5])
        if not L.inside(cf, x):
            continue
        u0 = (2 * rng.random() - 1) * 0.5
        n_pairs += 1
        if _first_control_feasible(x, u0, 7):
            n_feasible += 1
            assert L.inside(cf, A @ x + B * u0, 1e-7), (x, u0)
    assert n_feasible >= 500                                      # die Prüfung ist nicht leer: mindestens ein Sechstel der Paare ist zulässig
    ea = R.exit_areas(reg, 9)
    assert all(ea[n] == 0 for n in (7, 8, 9)) and all(ea[n] > 0 for n in range(1, 7))


def test_at_N_6_an_exit_from_C_exists_and_is_observed_with_adverse_costs():
    """„Garantie“ nie für N = 6 behaupten: exakt gibt es einen Austritt (Fläche 1/168), und mit dem Zufalls-Seed 716 und „zur Grenze treibenden“ Kosten beobachtet ihn der Regelkreis
    (in 500 Starts mindestens einmal; gemessen 2). Zeuge: Start in C, die erste Stellgröße des QP ist LP-zulässig (N = 6), der Folgezustand liegt um mehr als 1e-4 außerhalb C."""
    reg = R.regions(0.5, 5.0, 5.0)
    assert R.exit_areas(reg, 6)[6] == F(1, 168)
    res = L.experiment_falls(6, "grenze", 500, 716, 0.5, 5.0, 5.0)
    idx = [i for i in range(len(res.starts)) if res.exits_c[i] and res.in_c[i]]
    assert len(idx) >= 1
    prob = L.cost_problem(6, "grenze", 0.5, 5.0, 5.0)
    cf = P.to_float(reg.Cset)
    for i in idx:
        x = res.starts[i]
        U, st = prob.solve(x)
        assert st == "ok" and L.inside(cf, x)
        x1 = Q.A_CAR @ x + Q.B_CAR[:, 0] * U[0]
        assert _first_control_feasible(x, float(U[0]), 6)
        assert float(np.max(cf[0] @ x1 - cf[1])) > 1e-4


def test_with_Q_equal_identity_no_start_in_C_leaves_C_for_N_from_2():
    """Zählung (Zufalls-Seed 716, 500 Starts): bei Q = I, R = 0,1 oder 10 verlässt für N = 2 bis 8 keine Bahn aus C die Menge C; Ausnahme N = 1 (R = 0,1: wenige, R = 10: viele)."""
    for cost in ("q_i_r01", "q_i_r10"):
        for N in (2, 3, 5, 8):
            c = L.experiment_falls(N, cost, 500, 716, 0.5, 5.0, 5.0).counts()
            assert c["austritt"] == 0 and c["faelle_innen"] == 0 and c["faelle_aussen"] == c["nicht_in_c"], (cost, N)
