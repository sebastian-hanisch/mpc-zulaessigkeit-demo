"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): N = 0 und N = 1, jede Stellgrenzen- und Kastenstufe, entartete Terminalmenge ({0}: K_1 ist eine Strecke),
Start auf dem Rand, kleinste und größte Zahl der Starts; jede Kombination läuft durch Auswertung und Verdikt, ohne Fehler und ohne „nan“ im Text."""

import numpy as np
import pytest

import mzl_constants as C
import mzl_evaluation as ev
import mzl_loop as L
import mzl_poly as P
import mzl_regions as R


def _clean(text):
    assert "nan" not in text.lower() and "inf" not in text.lower() and "None" not in text, text


@pytest.mark.parametrize("umax", C.UMAX_OPTIONS)
@pytest.mark.parametrize("s", [(5.0, 5.0), (10.0, 5.0), (5.0, 10.0), (10.0, 10.0)])
@pytest.mark.parametrize("N", [0, 1, 5, 14])
@pytest.mark.parametrize("start", ["bbm_a", "bbm_d"])
def test_regions_analysis_and_verdict_for_all_limits(umax, s, N, start):
    ra = ev.analyse_regions(N, umax, s[0], s[1], start)
    kind, text = ev.verdict_regions(ra)
    assert kind in ("success", "info", "warning", "error")
    _clean(text)


@pytest.mark.parametrize("N", [1, 10])
@pytest.mark.parametrize("cost", C.COST_KEYS)
def test_falls_at_the_horizon_ends(N, cost):
    res = L.experiment_falls(N, cost, 100, 716, 0.5, 5.0, 5.0)
    c = res.counts()
    assert c["starts"] == 100 and c["faelle"] + c["konvergiert"] + c["randlauf"] + c["ende"] == 100
    kind, text = ev.verdict_falls(c, N, cost, 0.5, 5.0, 5.0)
    _clean(text)
    assert ev.falls_rows(res)[2]["Starts"] == 100


@pytest.mark.parametrize("umax", [0.25, 5.0])
@pytest.mark.parametrize("term", C.TERM_KEYS)
@pytest.mark.parametrize("N", [1, 10])
def test_terminal_at_the_ends_including_the_degenerate_zero_set(umax, term, N):
    ta = ev.analyse_terminal(term, N, umax, 5.0, 5.0, "bbm_a")
    tr = L.experiment_terminal(term, N, 50, 716, umax, 5.0, 5.0)
    kind, text = ev.verdict_terminal(ta, tr.counts())
    _clean(text)
    c = tr.counts()
    assert c["faelle"] == 0                                              # nie ein Fall mit steuerinvarianter Terminalmenge
    if term == "null" and N == 1:
        assert c["starts"] == 0 and ta.K_area == 0                      # K_1({0}) ist eine Strecke: keine Fläche, keine Startzustände
    else:
        assert c["starts"] == 50


def test_start_on_the_boundary_of_F_N_does_not_crash():
    reg = R.regions(0.5, 5.0, 5.0)
    prob = L.cost_problem(3, "q_i_r10", 0.5, 5.0, 5.0)
    for v in P.vertices(reg.F(3))[:4]:
        run = L.run_loop(prob, P.to_float(reg.F(3)), np.array([float(v[0]), float(v[1])]), 10)
        assert run.status in ("konvergiert", "verloren", "randlauf", "ende")


def test_horizon_one_is_the_hexagon_problem():
    """N = 1: nur x₁ ∈ X verlangt; F_1 ist das Sechseck, der Regler sieht nur einen Schritt (Handbeispiel: Fläche 75)."""
    ra = ev.analyse_regions(1, 0.5, 5.0, 5.0, "bbm_b")
    assert ra.areas[1] == 75 and ra.start_in_fN


def test_scalar_verdict_for_every_horizon_and_corner_start():
    for N in range(C.SK_N_MIN, C.SK_N_MAX + 1):
        for x0 in (C.SK_X0_MIN, -1.0, 0.0, 1.0, 1.125, C.SK_X0_MAX):
            for term in (False, True):
                for cost in C.SK_COST_KEYS:
                    kind, text = ev.verdict_scalar(ev.analyse_scalar(x0, N, term, cost))
                    _clean(text)


def test_few_and_many_starts_run():
    for n in (C.WF_STARTS_MIN, C.WF_STARTS_MAX):
        res = L.experiment_falls(3, "q_i_r10", n, 716, 0.5, 5.0, 5.0)
        assert res.counts()["starts"] == n
