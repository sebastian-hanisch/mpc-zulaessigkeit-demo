"""Regelkreis (mzl_loop) und Zufall (mzl_rng): Zustände, Statusarten, Startzustände, Zählungen auf Mini-Instanzen von Hand."""

import numpy as np
import pytest

import mzl_constants as C
import mzl_loop as L
import mzl_poly as P
import mzl_regions as R
from mzl_rng import SplitMix64


def test_splitmix64_reference_values():
    """SplitMix64 (Vigna) mit Zufalls-Seed 0: erste Ausgaben 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F (Referenzimplementierung)."""
    rng = SplitMix64(0)
    assert [rng.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_random_is_in_unit_interval_and_reproducible():
    a, b = SplitMix64(716), SplitMix64(716)
    xs = [a.random() for _ in range(1000)]
    assert xs == [b.random() for _ in range(1000)]
    assert all(0.0 <= x < 1.0 for x in xs) and 0.45 < np.mean(xs) < 0.55


def test_bbm_trajectories_status_and_period():
    reg = R.regions(0.5, 5.0, 5.0)
    prob = L.cost_problem(3, "q_i_r10", 0.5, 5.0, 5.0)
    run = L.run_loop(prob, P.to_float(reg.F(3)), np.array([-4.5, 3.0]), 20)
    assert run.status == "verloren" and run.k_stop == 2 and len(run.traj) == 3
    assert run.traj[-1] == pytest.approx([1.0, 2.0], abs=1e-6)
    run = L.run_loop(prob, P.to_float(reg.F(3)), np.array([-4.5, 2.0]), 12)
    assert run.status == "ende" and len(run.traj) == 13 and run.k_stop is None


def test_converged_status_when_the_state_is_small():
    reg = R.regions(0.5, 5.0, 5.0)
    prob = L.cost_problem(4, "q_i_r01", 0.5, 5.0, 5.0)
    run = L.run_loop(prob, P.to_float(reg.F(4)), np.array([0.0008, 0.0]), 10, conv_tol=C.CONV_TOL)
    assert run.status == "konvergiert" and run.k_stop == 1 and np.abs(run.traj[-1]).max() < C.CONV_TOL


def test_randlauf_when_the_state_is_slightly_outside_the_exact_set():
    """Zustand 3e-8 hinter der Grenze von F_1: innerhalb der Toleranz 1e-7 der Menge, aber das QP (Lockerung 1e-9) meldet unlösbar → Randlauf, nicht Verlust."""
    reg = R.regions(0.5, 5.0, 5.0)
    prob = L.cost_problem(1, "q_i_r10", 0.5, 5.0, 5.0)
    run = L.run_loop(prob, P.to_float(reg.F(1)), np.array([5.0 + 3e-8, 0.0]), 5)
    assert run.status == "randlauf" and run.k_stop == 0
    run = L.run_loop(prob, P.to_float(reg.F(1)), np.array([5.0 + 3e-6, 0.0]), 5)             # deutlich außerhalb: Verlust
    assert run.status == "verloren" and run.k_stop == 0


def test_sample_starts_are_inside_the_region_and_reproducible():
    reg = R.regions(0.5, 5.0, 5.0)
    zul = P.to_float(reg.F(3))
    a = L.sample_starts(zul, (5.0, 5.0), 200, 716)
    b = L.sample_starts(zul, (5.0, 5.0), 200, 716)
    c = L.sample_starts(zul, (5.0, 5.0), 200, 717)
    assert a.shape == (200, 2) and np.array_equal(a, b) and not np.array_equal(a, c)
    assert L.inside_many(zul, a).all()
    assert abs(a[:, 0].mean()) < 0.5 and abs(a[:, 1].mean()) < 0.5               # Region ist symmetrisch: Mittel nahe 0


def test_sample_starts_cover_the_region_uniformly():
    """Anteil der Starts in C gleich Fläche C / Fläche F_N (Band ± 0,06 bei 2 000 Starts): N = 2: 37,29 / 54,75 = 0,681."""
    reg = R.regions(0.5, 5.0, 5.0)
    pts = L.sample_starts(P.to_float(reg.F(2)), (5.0, 5.0), 2000, 11)
    share = L.inside_many(P.to_float(reg.Cset), pts).mean()
    assert share == pytest.approx(float(reg.area(6) / reg.area(2)), abs=0.06)


def test_sample_starts_of_a_degenerate_set_are_few():
    """Eine Strecke hat Fläche 0: das Ablehnungsverfahren findet keine Startzustände (kein Absturz, kein Endlosschleifen)."""
    T = R.terminal("null", 0.5, 5.0, 5.0)
    pts = L.sample_starts(P.to_float(T.K(1)), (5.0, 5.0), 50, 3, max_draws=20000)
    assert len(pts) == 0


def test_fall_counts_are_consistent():
    res = L.experiment_falls(3, "q_i_r10", 200, 716, 0.5, 5.0, 5.0)
    c = res.counts()
    assert c["in_c"] + c["nicht_in_c"] == c["starts"] == 200
    assert c["faelle"] == c["faelle_aussen"] + c["faelle_innen"]
    assert c["faelle"] + c["konvergiert"] + c["randlauf"] + c["ende"] == c["starts"]
    assert c["faelle_aussen"] <= c["nicht_in_c"] and c["austritt"] >= c["faelle_innen"]


def test_terminal_runs_decrease_the_value_function_and_never_fall():
    res = L.experiment_terminal("lqr1", 3, 60, 716, 0.5, 5.0, 5.0)
    c = res.counts()
    assert c["starts"] == 60 and c["faelle"] == 0 and c["wert_verletzt"] == 0 and c["konvergiert"] == 60


def test_value_decrease_is_checked_not_vacuous():
    """Der Test der Abnahme meldet Verletzungen, wenn die Endkosten falsch sind: mit P_f = 0,1·I (zu klein, kein Lyapunov-Gewicht des Gewinns) nimmt die Wertfunktion nicht überall ab."""
    reg_t = R.terminal("lqr1", 0.5, 5.0, 5.0)
    import mzl_qp as Q

    Xf = [(float(a), float(b), float(c)) for a, b, c in reg_t.Xf]
    prob = Q.build_problem(2, np.eye(2), 1.0, 0.1 * np.eye(2), [5.0, 5.0], 0.5, term=Xf)
    kfloat = P.to_float(reg_t.K(2))
    pts = L.sample_starts(kfloat, (5.0, 5.0), 60, 5)
    viol = sum(L.run_loop(prob, kfloat, x0, 40, conv_tol=1e-7, track_value=True).value_violations for x0 in pts)
    assert viol > 0
