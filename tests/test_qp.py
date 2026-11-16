"""QP-Löser und MPC-Problem von Hand: Mini-Instanzen mit bekannter Lösung (Handrechnung im Docstring)."""

import numpy as np
import pytest

import mzl_qp as Q


def test_unconstrained_minimum_is_found():
    """min ½·2·u² + (−2)·u = u² − 2u → u = 1 (Handrechnung), Nebenbedingung u ≤ 5 inaktiv."""
    qp = Q.QP([[2.0]], [[1.0]])
    U, st = qp.solve(np.array([-2.0]), np.array([5.0]))
    assert st == "ok" and U[0] == pytest.approx(1.0, abs=1e-12)


def test_active_constraint_clips_the_minimum():
    """Dasselbe Problem mit u ≤ 1/2: Lösung u = 1/2 (Rand)."""
    qp = Q.QP([[2.0]], [[1.0]])
    U, st = qp.solve(np.array([-2.0]), np.array([0.5]))
    assert st == "ok" and U[0] == pytest.approx(0.5, abs=1e-12)


def test_infeasible_is_reported():
    """u ≤ −1 und −u ≤ −1 (also u ≥ 1): leer."""
    qp = Q.QP([[2.0]], [[1.0], [-1.0]])
    U, st = qp.solve(np.array([0.0]), np.array([-1.0, -1.0]))
    assert U is None and st == "unzulaessig"


def test_two_variable_box_problem_by_hand():
    """min u₁² + u₂² − 4u₁ − 4u₂ (H = 2I, f = (−4, −4)) mit |uᵢ| ≤ 1: Minimum (2; 2) liegt außerhalb, Lösung (1; 1)."""
    C = np.vstack([np.eye(2), -np.eye(2)])
    qp = Q.QP(2 * np.eye(2), C)
    U, st = qp.solve(np.array([-4.0, -4.0]), np.ones(4))
    assert st == "ok" and U == pytest.approx([1.0, 1.0], abs=1e-12)


def test_mpc_problem_n1_from_hand():
    """N = 1, Q = P = 1, R = 1, Wagen, x₀ = (0; 2): x₁ = (2; 2 + u), Kosten 0 + u² + 4 + (2 + u)² → u = −1 (Stellgrenze 1 inaktiv bei |u| ≤ 5), x₂ = 1 ≤ 5."""
    Qm = np.eye(2)
    prob = Q.build_problem(1, Qm, 1.0, Qm, [5.0, 5.0], 5.0)
    U, st = prob.solve(np.array([0.0, 2.0]))
    assert st == "ok" and U[0] == pytest.approx(-1.0, abs=1e-9)
    # Kostenwert: x₀'Qx₀ + R u² + x₁'P x₁ = 4 + 1 + (4 + 1) = 10
    assert prob.cost(np.array([0.0, 2.0]), U) == pytest.approx(10.0, abs=1e-8)


def test_mpc_problem_with_state_bound_n1():
    """N = 1, Position nach einem Schritt höchstens 1: x₀ = (0; 2) gibt x₁(1) = 2 > 1, unzulässig (die Position wächst unabhängig von u um x₂)."""
    prob = Q.build_problem(1, np.eye(2), 1.0, np.eye(2), [1.0, 5.0], 5.0)
    U, st = prob.solve(np.array([0.0, 2.0]), slack=0.0)
    assert U is None and st == "unzulaessig"


def test_builder_rejects_horizon_zero():
    with pytest.raises(ValueError):
        Q.build_problem(0, np.eye(2), 1.0, np.eye(2), [5.0, 5.0], 1.0)
