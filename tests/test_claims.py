"""Claims-Tests: jede Zahl der Befunde B1 bis B11 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Voraussetzung überall: dieses Modell (Wagen und Skalar, Taktzeit 1), |u| ≤ 1/2, |xᵢ| ≤ 5 (wo nichts anderes steht), die genannten Kosten. Gebiete sind exakt (Brüche), Zählungen aus Regelkreisen
stehen in `data/mzl_frozen.json` (erzeugt von `tools/make_frozen.py`, Zufalls-Seed 716, 500 Starts); sie werden hier auf Konsistenz mit der README geprüft und einzelne Zellen neu gerechnet
(Band: Zählungen ± 2, plattformabhängig am Rand). Die vollständige Neu-Rechnung ist `python tools/make_frozen.py --check` (etwa eine halbe Minute)."""

import json
import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

import mzl_constants as C
import mzl_evaluation as ev
import mzl_loop as L
import mzl_poly as P
import mzl_regions as R

ROOT = Path(__file__).resolve().parent.parent
FROZEN = json.loads((ROOT / "data" / "mzl_frozen.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")
FLAT = README.replace("\\|", "|")


def _readme_has(*fragments):
    for frag in fragments:
        assert frag in FLAT, frag


def _col(cost, key):
    return [row[key] for row in FROZEN["costs"][cost]]


def _band(new, ref, tol=2):
    assert abs(new - ref) <= tol, (new, ref)


def _band_key(key, new, ref):
    """Zählungen ± 2; der Randlauf (das QP entscheidet bei 1e-9 nicht mehr) hängt an der letzten Binärstelle und ist plattformabhängig: dort ± 5 % der Starts (Linux-Lauf 2026-11-16:
    276 gegen 281 bei „nur Stellaufwand“, N = 3)."""
    _band(new, ref, 25 if key == "randlauf" else 2)


# --- B1 --------------------------------------------------------------------------------------------------------------------------------------

def test_b1_areas_shares_and_corners():
    reg = R.regions(0.5, 5.0, 5.0)
    assert [reg.area(n) for n in range(7)] == [F(100), F(75), F(219, 4), F(257, 6), F(155, 4), F(75, 2), F(895, 24)]
    shares = [ev.pct(reg.area(6) / reg.area(n)) for n in range(1, 6)]
    assert shares == ["49,7 %", "68,1 %", "87,1 %", "96,2 %", "99,4 %"]
    _readme_has("100; 75; 219/4 = 54,75; 257/6 = 42,83; 155/4 = 38,75; 75/2 = 37,5; 895/24 = 37,29", "**N̄ = 6**", "14 Ecken", "49,7 % der Fläche von F_1", "68,1 % von F_2", "87,1 % von F_3",
                "96,2 % von F_4", "99,4 % von F_5", "Fläche von C 149,17 gegen 37,29 (Faktor 4)")
    big = R.regions(1.0, 10.0, 10.0)
    assert float(big.area(6)) == pytest.approx(149.1667, abs=1e-4) and float(reg.area(6)) == pytest.approx(37.2917, abs=1e-4) and big.area(6) == 4 * reg.area(6)


# --- B2 --------------------------------------------------------------------------------------------------------------------------------------

def test_b2_bbm_states():
    reg = R.regions(0.5, 5.0, 5.0)
    s = (F(-9, 2), F(3))
    assert [n for n in range(8) if P.contains(reg.F(n), s)] == [0, 1, 2, 3, 4] and not P.contains(reg.Cset, s)
    assert [n for n in range(8) if P.contains(reg.F(n), (F(-3, 2), F(5, 2)))] == [0, 1, 2, 3]
    assert [n for n in range(8) if P.contains(reg.F(n), (F(1), F(2)))] == [0, 1, 2]
    assert P.contains(reg.Cset, (F(-9, 2), F(2)))
    _readme_has("(−4,5; 3) liegt in F_N für N = 0 bis 4, **nicht in C**", "−3/2, 1, 3, 9/2, 11/2, 6", "(−3/2; 5/2) liegt in F_3", "(1; 2) nur in F_2", "x₁(3) = 7 + 2u₀ + u₁ ≥ 11/2 > 5",
                "(−4,5; 2) liegt in C", "(−1,5; 2,5) nach (1; 2)", "(Periode 2)", "(−0,082; −0,040)")


# --- B3 --------------------------------------------------------------------------------------------------------------------------------------

def test_b3_scalar():
    import mzl_scalar as S

    assert S.f_seq(4) == [F(4), F(5, 2), F(7, 4), F(11, 8), F(19, 16)]
    assert S.next_interval(F(3, 2), 2) == (F(2), F(5, 2))
    _readme_has("f_N = 1 + 3/2^N (4; 5/2; 7/4; 11/8; 19/16 …)", "[2; 5/2], über f₂ = 7/4", "x₂ = 4", "gemessen x₁ = 2,5", "K_N = [−1; 1] = C für jedes N")


# --- B4 --------------------------------------------------------------------------------------------------------------------------------------

def test_b4_exit_areas():
    reg = R.regions(0.5, 5.0, 5.0)
    ea = R.exit_areas(reg, 8)
    assert [round(float(ea[n]), 3) for n in range(1, 6)] == [15.006, 12.131, 7.173, 2.214, 0.414]
    assert ea[6] == F(1, 168) and round(float(ea[6]), 4) == 0.0060 and ea[7] == ea[8] == 0
    _readme_has("15,006; 12,131; 7,173; 2,214; 0,414", "**1/168 = 0,0060**", "**für N ≥ 7 gleich 0**", "3 000 Zufallspaare")


# --- B5 und B6 (Messung, vorgerechnet) --------------------------------------------------------------------------------------------------------

def test_b5_frozen_counts_agree_with_the_readme():
    outside = _col("q_i_r10", "nicht_in_c")[:6]
    in_c = _col("q_i_r10", "in_c")
    assert outside == [235, 151, 60, 14, 3, 0] and in_c[:6] == [265, 349, 440, 486, 497, 500]
    for cost in ("q_i_r01", "q_i_r10", "pos"):                       # die Starts hängen nicht an den Kosten
        assert _col(cost, "nicht_in_c") == _col("q_i_r10", "nicht_in_c")
    for cost in ("q_i_r01", "q_i_r10"):
        assert _col(cost, "austritt")[1:] == [0] * 7                  # N = 2 bis 8: aus C fällt keiner
        assert _col(cost, "faelle_aussen")[1:] == _col(cost, "nicht_in_c")[1:]
    assert _col("q_i_r01", "austritt")[0] == 2 and _col("q_i_r10", "austritt")[0] == 201
    assert _col("pos", "austritt")[0] == 260 and _col("pos", "austritt")[1] == 0 and _col("pos", "austritt")[2:] == [0] * 6
    assert min(in_c[1:]) == 349 and max(in_c) == 500
    _readme_has("235; 151; 60; 14; 3; 0 von je 500", "(0 von 349 bis 500)", "2 von 265 Starts in C (R = 0,1) bzw. 201 von 265 (R = 10)", "N = 1 (260 von 265) und N = 2 (0 bei Zufalls-Seed 716;",
                "in sechs 1 bis 4 von je 326 bis 352 Starts")


def test_b5_ten_seed_check_for_pos_n2():
    rows = FROZEN["seeds"]["pos:2"]
    assert [r["seed"] for r in rows] == list(range(716, 726))
    nz = [r["austritt"] for r in rows if r["austritt"] > 0]
    assert len(nz) == 6 and min(nz) == 1 and max(nz) == 4
    assert (min(r["in_c"] for r in rows), max(r["in_c"] for r in rows)) == (326, 352)
    assert all(r["austritt"] == 0 for r in FROZEN["seeds"]["q_i_r10:2"]) and all(r["austritt"] == 0 for r in FROZEN["seeds"]["q_i_r01:2"])


def test_b6_frozen_counts_agree_with_the_readme():
    assert _col("stell", "austritt")[:5] == [260, 228, 102, 18, 0] and _col("stell", "austritt")[5:] == [0, 0, 0]
    assert _col("grenze", "austritt") == [260, 340, 323, 209, 48, 2, 0, 0]
    assert _col("grenze", "randlauf")[5:] == [498, 500, 500]
    assert _col("grenze", "faelle_innen") == _col("grenze", "austritt")
    seeds6 = [r["austritt"] for r in FROZEN["seeds"]["grenze:6"]]
    assert sum(1 for v in seeds6 if v > 0) == 9 and min(v for v in seeds6 if v > 0) == 1 and max(seeds6) == 3
    assert all(r["austritt"] == 0 and r["faelle_innen"] == 0 for r in FROZEN["seeds"]["grenze:7"])
    _readme_has("260; 228; 102; 18; 0 für N = 1 bis 5", "(Starts in C: 265; 349; 440; 486; 497)", "260; 340; 323; 209; 48; **2**; 0; 0 für N = 1 bis 8", "2 von 500 Starts (Zufalls-Seed 716)",
                "in neun von zehn Zufalls-Seeds (716 bis 725) 1 bis 3", "0,0009 und 0,0094", "Für N = 7 sind es in allen zehn Zufalls-Seeds 0", "498 von 500 bei N = 6, 500 bei N = 7 und 8")


@pytest.mark.parametrize("cost,N", [("q_i_r10", 3), ("q_i_r01", 1), ("q_i_r10", 1), ("pos", 1), ("stell", 3), ("grenze", 2), ("grenze", 5), ("grenze", 6), ("grenze", 7)])
def test_b5_b6_recomputed_cells_agree_with_the_frozen_file_within_the_band(cost, N):
    c = L.experiment_falls(N, cost, 500, 716, 0.5, 5.0, 5.0).counts()
    ref = FROZEN["costs"][cost][N - 1]
    for key in ("in_c", "nicht_in_c", "faelle_aussen", "faelle_innen", "austritt", "randlauf", "konvergiert"):
        _band_key(key, c[key], ref[key])


def test_b6_the_two_exits_at_n6_are_far_above_the_threshold():
    """Die Austritte bei N = 6 (Zufalls-Seed 716) sind kein Rundungsrand: Abstand 0,0009 und 0,0094 über der Kante von C (Schwelle der Prüfung 1e-6)."""
    import numpy as np

    reg = R.regions(0.5, 5.0, 5.0)
    res = L.experiment_falls(6, "grenze", 500, 716, 0.5, 5.0, 5.0)
    prob = L.cost_problem(6, "grenze", 0.5, 5.0, 5.0)
    ca, cb = P.to_float(reg.Cset)
    dist = []
    for i in np.where(res.exits_c & res.in_c)[0]:
        run = L.run_loop(prob, P.to_float(reg.F(6)), res.starts[i], 5)
        dist.append(float(np.max(run.traj[1] @ ca.T - cb)))
    assert len(dist) == 2 and sorted(round(d, 4) for d in dist) == [0.0009, 0.0094]


# --- B7 --------------------------------------------------------------------------------------------------------------------------------------

def test_b7_terminal_shares_and_areas():
    reg = R.regions(0.5, 5.0, 5.0)
    shares = {key: [R.share_of_c(R.terminal(key, 0.5, 5.0, 5.0).K(n), reg.Cset) for n in range(14)] for key in ("lqr1", "lqr001", "null")}
    pick = (0, 1, 2, 3, 6, 9)
    assert [round(float(shares["lqr1"][n]) * 100, 1) for n in pick] == [15.1, 34.9, 49.6, 63.1, 92.9, 100.0]
    assert [round(float(shares["lqr001"][n]) * 100, 1) for n in pick] == [7.1, 21.2, 39.3, 53.4, 85.6, 99.8]
    assert shares["lqr001"][10] == 1 and shares["lqr001"][9] < 1
    assert [round(float(shares["null"][n]) * 100, 1) for n in pick] == [0.0, 0.0, 2.7, 10.7, 56.2, 87.0]
    assert shares["null"][13] == 1 and shares["null"][12] < 1 and shares["lqr1"][9] == 1 and shares["lqr1"][8] < 1
    assert round(float(P.area_of(R.terminal("lqr1", 0.5, 5.0, 5.0).Xf)), 3) == 5.615 and round(float(P.area_of(R.terminal("lqr001", 0.5, 5.0, 5.0).Xf)), 3) == 2.653
    _readme_has("5,615 (15,1 % von C)", "15,1; 34,9; 49,6; 63,1; 92,9; **100 %**", "2,653 (7,1 %); 7,1; 21,2; 39,3; 53,4; 85,6; 99,8 % bei N = 9, **100 % bei N = 10**",
                "0; 0; 2,7; 10,7; 56,2; 87,0 %, **100 % erst bei N = 13**", "K_2({0}) ist ein Parallelogramm der Fläche 1", "schon bei N = 6 gleich C")


# --- B8 --------------------------------------------------------------------------------------------------------------------------------------

def test_b8_frozen_terminal_table_and_recomputed_cells():
    rows = FROZEN["terminal"]
    assert [r["N"] for r in rows] == list(range(1, 11))
    assert all(r["starts"] == 500 and r["faelle"] == 0 and r["konvergiert"] == 500 and r["wert_verletzt"] == 0 and r["randlauf"] == 0 for r in rows)
    for N in (1, 3, 9):
        c = L.experiment_terminal("lqr1", N, 500, 716, 0.5, 5.0, 5.0).counts()
        assert c["faelle"] == 0 and c["konvergiert"] == 500 and c["wert_verletzt"] == 0
    _readme_has("je N = 1 bis 10 in 500 Starts in K_N **0 Fälle, 500 von 500 konvergieren**", "0 Verstöße in 5 000 Läufen, Toleranz 10⁻⁸", "stimmen mit der DARE auf 5 Stellen überein")


# --- B9, B10 ---------------------------------------------------------------------------------------------------------------------------------

def test_b9_b10_stated_in_the_readme():
    _readme_has("N̄ = 9; 6; 4; 3; 1 für u_max = 1/4; 1/2; 1; 2; 5 bei s₁ = 5", "27,36; 37,29; 50; 65; 75 (27,4 bis 75 % des Kastens)", "N̄ = 13; 9; 5; 3; 1 und 79,39; 109,44; 145; 165; 175",
                "bis N = 60 keine endliche Determiniertheit", "(−4; 17/2) liegt nur in F_N für N ≤ 3", "bei N = 10 ist sie 40,35")
    verdict = ev.verdict_umax(R.u_table(s2=5.0), 0.25, 5.0)[1]
    assert "N̄ = 9" in verdict and "27,36" in verdict


# --- Allgemeines ------------------------------------------------------------------------------------------------------------------------------

def test_params_of_the_frozen_file_are_the_documented_ones():
    p = FROZEN["params"]
    assert (p["umax"], p["s1"], p["s2"], p["n_starts"], p["seed"]) == (0.5, 5.0, 5.0, 500, 716)
    assert ev.frozen_applies(0.5, 5.0, 5.0, 500, 716, FROZEN) and not ev.frozen_applies(0.5, 5.0, 5.0, 500, 717, FROZEN) and not ev.frozen_applies(1.0, 5.0, 5.0, 500, 716, FROZEN)


def test_make_frozen_check_matches_the_stored_file_for_one_cost():
    sys.path.insert(0, str(ROOT / "tools"))
    import make_frozen

    cell = L.experiment_falls(4, "stell", make_frozen.PARAMS["n_starts"], make_frozen.PARAMS["seed"], make_frozen.PARAMS["umax"], make_frozen.PARAMS["s1"], make_frozen.PARAMS["s2"]).counts()
    ref = FROZEN["costs"]["stell"][3]
    for key in ("in_c", "nicht_in_c", "austritt", "faelle", "randlauf"):
        _band_key(key, cell[key], ref[key])


def test_the_readme_names_no_unconfirmed_source_numbers_and_uses_the_house_terms():
    for bad in ("7 / 9 / 13", "7/9/13", "13 Regionen", "trägt", "tragen", "läuft lokal", "Saatwert", "geplant"):
        assert bad not in README, bad
    assert "Zufalls-Seed" in README and "Weitergabe nur mit Erlaubnis des Verlags" in README and "nichts davon ist wörtlich übernommen" in README
    assert "https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo" in README


def test_the_two_texts_of_the_app_and_the_readme_agree_on_the_hand_numbers():
    """Zwei Fassungen desselben Textes werden verglichen, nicht von Hand synchron gehalten: Die Handrechnungen der App stehen auch in der README."""
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for frag in ("1 + 3/2^N", "[2; 5/2]", "11/2 > 5", "1/168"):
        assert frag in app and frag in README, frag
