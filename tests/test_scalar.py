"""Handbeispiel H1: der Skalar x⁺ = 2x + u, |u| ≤ 1, |x| ≤ 4. Bruch gleich Bruch, die Gebiete zusätzlich gegen eine LP-Zulässigkeitsprüfung (anderer Rechenweg)."""

from fractions import Fraction as F

import numpy as np
import pytest
from scipy.optimize import linprog

import mzl_constants as C
import mzl_scalar as S


def test_halbbreiten_are_one_plus_three_over_two_to_the_n():
    f = S.f_seq(11)
    assert f[:6] == [F(4), F(5, 2), F(7, 4), F(11, 8), F(19, 16), F(35, 32)]
    assert all(v == 1 + F(3, 2 ** n) for n, v in enumerate(f))
    assert all(v > 1 for v in f)                                    # nie ein endliches N̄
    assert S.c_halb() == 1


def _lp_zulaessig(x0, N, a=2, xmax=4, umax=1, term=None):
    """x₀ ∈ F_N ⇔ es gibt u₀ … u_(N−1) ∈ [−umax, umax] mit |x_k| ≤ xmax (k = 1 … N), x_k = aᵏ x₀ + Σ a^(k−1−j) u_j (LP)."""
    rows, rhs = [], []
    for k in range(1, N + 1):
        zeile = [float(a ** (k - 1 - j)) if j < k else 0.0 for j in range(N)]
        lim = xmax if (term is None or k < N) else min(xmax, term)
        rows.append(zeile); rhs.append(lim - a ** k * x0)
        rows.append([-z for z in zeile]); rhs.append(lim + a ** k * x0)
    r = linprog(np.zeros(N), A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(-umax, umax)] * N, method="highs")
    return r.status == 0


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
def test_formula_against_lp_on_a_grid(N):
    grenze = float(S.f_seq(N)[N])
    for x0 in np.linspace(-4, 4, 161):
        if abs(abs(x0) - grenze) < 1e-9:
            continue                                                 # Randpunkte gehören nicht in die LP-Gegenprobe
        assert _lp_zulaessig(float(x0), N) == (abs(x0) < grenze), (N, x0)


def test_start_three_halves():
    """x₀ = 3/2: in F_2 (7/4 ≥ 3/2), nicht in F_3 (11/8 < 3/2); für N = 2 liegt jedes erlaubte x₁ = 3 + u₀ in [2; 5/2], über f₂ = 7/4."""
    f = S.f_seq(4)
    assert F(3, 2) <= f[2] and F(3, 2) > f[3]
    assert S.next_interval(F(3, 2), 2) == (F(2), F(5, 2))
    assert S.next_interval(F(3, 2), 2)[0] > f[2]
    assert S.classify(F(3, 2), 2) == "in_FN_nicht_C" and S.classify(F(3, 2), 3) == "aussen"


def test_start_in_c_can_still_lose_with_cost_driving_to_the_bound():
    """x₀ = 1 in C: x₁ = 5/2 ist zulässig (x₂ = 4), liegt aber über f₂ = 7/4. Ob der Regler es wählt, entscheiden die Kosten: „zur Grenze treibend“ tut es, „gutartig“ nicht."""
    x1 = 2 * F(1) + F(1, 2)
    assert x1 == F(5, 2) and 2 * x1 - 1 == 4 and x1 > S.f_seq(2)[2]
    treibend = S.run(1.0, 2, "grenze", False)
    assert treibend.status == "verloren" and treibend.k_stop == 1 and treibend.traj[1][0] == pytest.approx(2.5, abs=1e-6)
    gut = S.run(1.0, 2, "gut", False)
    assert gut.status == "ende" and np.allclose(gut.traj[:, 0], 1.0, atol=1e-6)


def test_terminal_set_gives_C_for_every_N_and_guarantees_the_solvability():
    for N in range(0, 6):
        assert S.f_seq(N, start=C.SK_TERM)[N] == 1
    for N in range(1, 6):
        for x0 in (-1.0, -0.5, 0.0, 0.75, 1.0):
            for cost in C.SK_COST_KEYS:
                run = S.run(x0, N, cost, True, T=12)
                assert run.status != "verloren", (N, x0, cost)
    assert S.classify(F(3, 2), 5, terminal=True) == "aussen"


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_terminal_region_against_lp(N):
    for x0 in np.linspace(-4, 4, 81):
        if abs(abs(x0) - 1.0) < 1e-9:
            continue
        assert _lp_zulaessig(float(x0), N, term=1.0) == (abs(x0) < 1.0), (N, x0)


def test_n_zero_has_no_controller():
    run = S.run(1.0, 0, "gut", False)
    assert run.status == "kein_regler"


def test_cost_driving_strength_decides_the_first_step():
    """x₀ = 1, N = 2, Q = 0, P_f = 0, dazu −x linear: bei R = 0,1 ist x₁ = 5/2 (Rand von F₁); bei größerem R wählt der Regler weniger (gemessen 2,2 bei R = 0,5; 2,07 bei R = 1,5)."""
    import mzl_qp as Q

    def x1(R):
        prob = Q.build_problem(2, [[0.0]], R, [[0.0]], [4.0], 1.0, lin=(-1.0,), A=[[2.0]], B=[[1.0]])
        U, st = prob.solve(np.array([1.0]), slack=0.0)
        assert st == "ok"
        return 2.0 + U[0]

    assert x1(0.1) == pytest.approx(2.5, abs=1e-6)
    assert x1(0.5) == pytest.approx(2.2, abs=1e-6) and x1(1.5) == pytest.approx(2.0 + 0.1 / 1.5, abs=1e-6)
