"""Exakte Polytope (mzl_poly) und Gebiete (mzl_regions) gegen Handrechnung: Handbeispiele H2 und H3 des Plans, Randfälle (N = 0, u_max = 0, entartete Terminalmenge).

Alles mit Brüchen: Bruch gleich Bruch, auf jeder Plattform gleich."""

from fractions import Fraction as F

import pytest

import mzl_constants as C
import mzl_poly as P
import mzl_regions as R
from conftest import A_EX, B_EX


def test_box_vertices_and_area():
    b = P.box(5)
    assert sorted(P.vertices(b)) == sorted([(F(5), F(5)), (F(5), F(-5)), (F(-5), F(5)), (F(-5), F(-5))])
    assert P.area_of(b) == 100
    assert P.area_of(P.box(2, 3)) == 24


def test_first_region_is_the_hexagon_of_the_hand_example():
    """F_1: |x₁|, |x₂|, |x₁ + x₂| ≤ 5; Ecken (5; 0), (5; −5), (0; −5), (−5; 0), (−5; 5), (0; 5); Fläche 100 − 2·12,5 = 75."""
    f1 = P.step_back(P.box(5), P.box(5), A_EX, B_EX, F(1, 2))
    assert set(P.vertices(f1)) == {(F(5), F(0)), (F(5), F(-5)), (F(0), F(-5)), (F(-5), F(0)), (F(-5), F(5)), (F(0), F(5))}
    assert P.area_of(f1) == 75
    # |x₁ + x₂| ≤ 5 ist die dritte Kante: (3; 2) liegt außerhalb, (2; 2) innerhalb
    assert not P.contains(f1, (F(3), F(3))) and P.contains(f1, (F(2), F(2)))


def test_areas_of_F_N_main_measurement():
    """Flächen der Hauptmessung (|u| ≤ 1/2, |xᵢ| ≤ 5), exakt: 100, 75, 219/4, 257/6, 155/4, 75/2, 895/24; N̄ = 6."""
    reg = R.regions(0.5, 5.0, 5.0)
    assert reg.n_bar == 6
    assert [reg.area(n) for n in range(7)] == [F(100), F(75), F(219, 4), F(257, 6), F(155, 4), F(75, 2), F(895, 24)]
    assert len(P.vertices(reg.Cset)) == 14 and len(reg.Cset) == 14
    assert reg.area(9) == reg.area(6) == F(895, 24)             # über N̄ hinaus gleich C


def test_c_corners_are_the_vertices_from_the_measurement():
    reg = R.regions(0.5, 5.0, 5.0)
    expected = {(F(-9, 2), F(-1, 2)), (F(-7, 2), F(-1)), (F(-2), F(-3, 2)), (F(0), F(-2)), (F(5, 2), F(-5, 2)), (F(5), F(-35, 12)), (F(5), F(0)),
                (F(9, 2), F(1, 2)), (F(7, 2), F(1)), (F(2), F(3, 2)), (F(0), F(2)), (F(-5, 2), F(5, 2)), (F(-5), F(35, 12)), (F(-5), F(0))}
    assert set(P.vertices(reg.Cset)) == expected


def test_c_is_control_invariant_exactly():
    """C ⊆ Vorbild von C: jeder Zustand aus C hat eine Stellgröße mit Folgezustand in C."""
    reg = R.regions(0.5, 5.0, 5.0)
    assert P.subset(reg.Cset, P.step_back(reg.Cset, reg.X, A_EX, B_EX, F(1, 2)))


def test_hand_example_states():
    """(1; 2): x₁(3) = 7 + 2u₀ + u₁ ≥ 11/2 > 5, nur in F_2. (−9/2; 3) in F_4, nicht in F_5 und nicht in C. (−9/2; 2) in C."""
    reg = R.regions(0.5, 5.0, 5.0)
    kleinste = min(7 + 2 * u0 + u1 for u0 in (F(-1, 2), F(1, 2)) for u1 in (F(-1, 2), F(1, 2)))
    assert kleinste == F(11, 2)
    assert P.contains(reg.F(2), (F(1), F(2))) and not P.contains(reg.F(3), (F(1), F(2)))
    s = (F(-9, 2), F(3))
    assert P.contains(reg.F(4), s) and not P.contains(reg.F(5), s) and not P.contains(reg.Cset, s)
    assert P.contains(reg.Cset, (F(-9, 2), F(2)))
    # stärkstes Bremsen u = −1/2: Positionen −3/2, 1, 3, 9/2, 11/2, 6 (Handrechnung)
    x = s
    pos = []
    for _ in range(6):
        x = (x[0] + x[1], max(x[1] - F(1, 2), F(0)))
        pos.append(x[0])
    assert pos == [F(-3, 2), F(1), F(3), F(9, 2), F(11, 2), F(6)]


def test_terminal_set_zero_gives_the_parallelogram_of_area_one():
    """Regressionstest: K_2({0}) = {|x₁ + 2x₂| ≤ 1/2, |x₁ + x₂| ≤ 1/2} hat die Fläche 1 (eine erste Rechnung warf beim Vereinfachen die Gleichungen weg, Fläche 0 bis N = 13)."""
    zero = P.box(0)
    k1 = P.step_back(zero, P.box(5), A_EX, B_EX, F(1, 2))
    assert P.area_of(k1) == 0                                   # K_1({0}) ist eine Strecke
    assert len(P.vertices(k1)) == 2
    k2 = P.step_back(k1, P.box(5), A_EX, B_EX, F(1, 2))
    assert P.area_of(k2) == 1
    # Ecken von Hand: (s; t) = (x₁ + 2x₂; x₁ + x₂) = (±1/2; ±1/2) gibt x₂ = s − t, x₁ = 2t − s
    assert set(P.vertices(k2)) == {(F(1, 2), F(0)), (F(-3, 2), F(1)), (F(-1, 2), F(0)), (F(3, 2), F(-1))}
    for x in ((F(0), F(0)), (F(1, 4), F(-1, 4)), (F(1, 2), F(-1, 2))):
        assert P.contains(k2, x) == (abs(x[0] + 2 * x[1]) <= F(1, 2) and abs(x[0] + x[1]) <= F(1, 2))


def test_terminal_zero_reaches_C_exactly_at_n13():
    T = R.terminal("null", 0.5, 5.0, 5.0)
    reg = R.regions(0.5, 5.0, 5.0)
    assert T.n_bar == 13 and P.same(T.K(13), reg.Cset) and not P.same(T.K(12), reg.Cset)
    assert [T.K(n) and P.area_of(T.K(n)) for n in range(4)] == [F(0), F(0), F(1), F(4)]


def test_lqr_terminal_sets_and_their_nesting():
    reg = R.regions(0.5, 5.0, 5.0)
    for key, n_full in (("lqr1", 9), ("lqr001", 10)):
        T = R.terminal(key, 0.5, 5.0, 5.0)
        assert P.subset(T.Xf, reg.Cset)
        for n in range(len(T.seq) - 1):
            assert P.subset(T.K(n), T.K(n + 1))                       # geschachtelt
            assert P.subset(T.K(n), reg.Cset)                         # in C (X_f steuerinvariant)
        assert P.same(T.K(n_full), reg.Cset) and not P.same(T.K(n_full - 1), reg.Cset)


# --- Randfälle -------------------------------------------------------------------------------------------------------------------------------

def test_n_zero_is_the_box():
    reg = R.regions(0.5, 5.0, 5.0)
    assert P.same(reg.F(0), P.box(5))


def test_zero_input_limit_is_the_free_motion():
    """u_max = 0: F_N = {x ∈ X : |x₁ + k·x₂| ≤ 5 für k = 1 … N} (Handrechnung: x₁(k) = x₁ + k·x₂ ohne Stellgröße). Es gibt kein endliches N̄: F_N schrumpft gegen die Strecke x₂ = 0,
    ohne sie zu erreichen."""
    from mzl_rng import SplitMix64

    X = P.box(5)
    seq, n_bar = P.reach_sequence(X, X, A_EX, B_EX, F(0), n_max=12)
    assert n_bar is None
    areas = [P.area_of(c) for c in seq]
    assert all(a > b > 0 for a, b in zip(areas, areas[1:]))
    rng = SplitMix64(5)
    for N in range(1, 7):
        for _ in range(60):
            x = (F(int(rng.below(2001)) - 1000, 200), F(int(rng.below(2001)) - 1000, 200))   # Vielfache von 1/200 in [−5; 5], exakt
            hand = abs(x[1]) <= 5 and all(abs(x[0] + k * x[1]) <= 5 for k in range(0, N + 1))
            assert P.contains(seq[N], x) == hand, (N, x)


def test_tiny_box_makes_C_the_origin_or_a_tiny_set():
    """Kasten |x₁| ≤ 0, |x₂| ≤ 0: nur der Ursprung, C = {0}, N̄ = 0."""
    reg = R.regions(0.5, 0.0, 0.0)
    assert reg.n_bar == 0
    assert P.area_of(reg.Cset) == 0 and P.vertices(reg.Cset) == [(F(0), F(0))]


def test_scalar_like_scaling_of_all_limits_is_exact():
    """F_N(10, 1) = 2·F_N(5, 1/2) für N = 0 bis 8 (Ecken), Fläche von C 149,17 gegen 37,29 (Faktor 4)."""
    small, big = R.regions(0.5, 5.0, 5.0), R.regions(1.0, 10.0, 10.0)
    for n in range(9):
        assert sorted(P.vertices(big.F(n))) == sorted((2 * v[0], 2 * v[1]) for v in P.vertices(small.F(n)))
    assert big.area(6) == 4 * small.area(6) == F(895, 6)


def test_no_finite_determinedness_index_for_example_12_2_sets():
    """BBM Beispiel 12.2 (nur die Mengen): A = [[2, 1], [0, 1/2]], B = (1; 0), |u| ≤ 1, |xᵢ| ≤ 10: bis N = 60 kein N̄; Flächen 400; 219,5; 130; 85; 62,5; 51,25; (−4; 17/2) nur in F_N für N ≤ 3."""
    A2 = [[F(2), F(1)], [F(0), F(1, 2)]]
    B2 = [F(1), F(0)]
    X = P.box(10)
    seq, n_bar = P.reach_sequence(X, X, A2, B2, F(1), n_max=12)
    assert n_bar is None and len(seq) == 13                           # Abbruch bei n_max, ohne Ende
    assert [P.area_of(c) for c in seq[:6]] == [F(400), F(439, 2), F(130), F(85), F(125, 2), F(205, 4)]
    assert float(P.area_of(seq[10])) == pytest.approx(40.3516, abs=1e-4) and P.area_of(seq[12]) < P.area_of(seq[10])         # fällt weiter
    p = (F(-4), F(17, 2))
    assert [P.contains(c, p) for c in seq[:6]] == [True, True, True, True, False, False]


def test_exit_area_is_positive_up_to_n_bar_and_zero_after():
    reg = R.regions(0.5, 5.0, 5.0)
    ea = R.exit_areas(reg, 8)
    assert ea[6] == F(1, 168)
    assert [ea[n] for n in range(1, 6)] == [F(2521, 168), F(1019, 84), F(1205, 168), F(31, 14), F(29, 70)]
    assert ea[7] == 0 and ea[8] == 0


def test_u_table_matches_the_measurement():
    table = {(r["umax"], r["s1"]): r for r in R.u_table()}
    n_bar = {(u, s): table[(u, s)]["n_bar"] for (u, s) in table}
    assert [n_bar[(u, 5.0)] for u in C.UMAX_OPTIONS] == [9, 6, 4, 3, 1]
    assert [n_bar[(u, 10.0)] for u in C.UMAX_OPTIONS] == [13, 9, 5, 3, 1]
    assert [float(table[(u, 5.0)]["area_C"]) for u in C.UMAX_OPTIONS] == pytest.approx([27.3611, 37.2917, 50.0, 65.0, 75.0], abs=1e-4)
    assert [float(table[(u, 10.0)]["area_C"]) for u in C.UMAX_OPTIONS] == pytest.approx([79.3942, 109.4444, 145.0, 165.0, 175.0], abs=1e-4)
