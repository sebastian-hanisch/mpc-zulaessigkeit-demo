"""Verdikte (DEMO-PLAYBOOK §12 und §11a Punkt 7): reine Funktionen (Art, Text). Je Preset die erwartete Zeile, je Zweig eine handgesetzte Analyse, jede Zahl im Text.

Ein Preset ohne Effekt meldet keinen Effekt: „Von Hand“ meldet keine Terminalmenge, „Sicher ab N = 7“ meldet keinen Austritt, „Grobe Stellgrenze“ meldet N̄ = 1."""

from fractions import Fraction as F

import pytest

import mzl_constants as C
import mzl_evaluation as ev
import mzl_loop as L
import mzl_regions as R

NB = " "          # geschütztes Leerzeichen


def _vals(preset):
    v = dict(umax=0.5, s1=5.0, s2=5.0, sk_N=2, sk_x0=1.5, sk_term=False, sk_cost="gut", wg_N=C.DEFAULT_WG_N, wg_start=C.DEFAULT_START, wf_N=C.DEFAULT_WF_N,
             wf_cost=C.DEFAULT_COST_FALL, wf_n=500, wf_seed=C.DEFAULT_SEED, tm_term=C.DEFAULT_TERM, tm_N=C.DEFAULT_TM_N)
    v.update(C.PRESETS[preset])
    return v


def _falls(v):
    res = L.experiment_falls(v["wf_N"], v["wf_cost"], v["wf_n"], v["wf_seed"], v["umax"], v["s1"], v["s2"])
    return res.counts()


def test_preset_von_hand_scalar_verdict():
    v = _vals("Von Hand: Skalar a = 2")
    kind, text = ev.verdict_scalar(ev.analyse_scalar(v["sk_x0"], v["sk_N"], v["sk_term"], v["sk_cost"]))
    assert kind == "warning"
    for frag in ("x₀ = 3/2 ist lösbar (in F_2 = [−7/4; 7/4])", "außerhalb von C = [−1; 1]", "fährt 1 Takt(e) und scheitert", "Jedes erlaubte x₁ liegt in [2; 5/2], über 7/4 (1,75)"):
        assert frag in text
    assert "Terminalmenge" not in text


def test_scalar_branches():
    a = ev.analyse_scalar(3.5, 2, False, "gut")                      # 7/4 < 3,5: außerhalb
    kind, text = ev.verdict_scalar(a)
    assert kind == "error" and "x₀ = 7/2 liegt außerhalb von F_2 = [−7/4; 7/4]" in text and "unlösbar" in text
    kind, text = ev.verdict_scalar(ev.analyse_scalar(1.5, 2, True, "gut"))
    assert kind == "error" and "liegt außerhalb von K_2 = [−1; 1]" in text and "weist den Start vor dem Losfahren ab" in text
    kind, text = ev.verdict_scalar(ev.analyse_scalar(1.0, 2, False, "grenze"))
    assert kind == "warning" and "liegt in C = [−1; 1] und scheitert trotzdem" in text and "x₁ = 2,500" in text and "über 7/4 (1,75)" in text
    kind, text = ev.verdict_scalar(ev.analyse_scalar(1.0, 2, False, "gut"))
    assert kind == "success" and "bleibt bei diesen Kosten in den 12 gerechneten Takten lösbar" in text and "keine Garantie" in text
    kind, text = ev.verdict_scalar(ev.analyse_scalar(1.0, 2, True, "grenze"))
    assert kind == "success" and "K_N = C für jedes N" in text and "nach 12 gerechneten Takten" in text
    kind, text = ev.verdict_scalar(ev.analyse_scalar(1.0, 0, False, "gut"))
    assert kind == "info" and text.startswith("N = 0: Es gibt kein Problem zu lösen")


def test_preset_falle_regions_verdict():
    v = _vals("Wagen: die Falle (BBM 12.1, N = 3)")
    kind, text = ev.verdict_regions(ev.analyse_regions(v["wg_N"], v["umax"], v["s1"], v["s2"], v["wg_start"]))
    assert kind == "warning"
    for frag in ("100,00; 75,00; 54,75; 42,83; 38,75; 37,50; 37,29", "N̄ = 6", "(14 Ecken, exakt steuerinvariant)", "Der Start (−4,5; 3) liegt in F_N für N = 0 bis 4, aber nicht in C",
                 "fährt der Regler 2 Takt(e) und scheitert bei (1; 2)", "die Position nach 3 Takten bei 11/2, über der Grenze 5"):
        assert frag in text, frag


def test_regions_branches():
    kind, text = ev.verdict_regions(ev.analyse_regions(4, 0.5, 5.0, 5.0, "bbm_a"))                   # in F_4 ohne C: Regler kann starten, scheitert später oder nicht
    assert "liegt in F_N für N = 0 bis 4, aber nicht in C" in text
    kind, text = ev.verdict_regions(ev.analyse_regions(5, 0.5, 5.0, 5.0, "bbm_a"))
    assert kind == "info" and "Mit N = 5 ist er nicht lösbar: Der Regler startet gar nicht." in text
    kind, text = ev.verdict_regions(ev.analyse_regions(0, 0.5, 5.0, 5.0, "bbm_a"))
    assert kind == "info" and "Bei N = 0 gibt es keinen Regler" in text
    kind, text = ev.verdict_regions(ev.analyse_regions(6, 0.5, 5.0, 5.0, "bbm_b"))
    assert kind == "success" and "liegt in C, also in F_N für jedes N" in text and "konvergiert der Regler in" in text
    kind, text = ev.verdict_regions(ev.analyse_regions(7, 0.5, 5.0, 5.0, "bbm_d"))
    assert "(1; 2) liegt in F_N für N = 0 bis 2, aber nicht in C" in text and "nicht lösbar" in text


def test_preset_falle_falls_verdict():
    v = _vals("Wagen: die Falle (BBM 12.1, N = 3)")
    c = _falls(v)
    kind, text = ev.verdict_falls(c, v["wf_N"], v["wf_cost"], v["umax"], v["s1"], v["s2"])
    assert kind == "info"
    assert f"{c['nicht_in_c']} von 500 Starts liegen außerhalb von C, {c['faelle_aussen']} davon fallen" in text
    assert f"Aus C heraus fallen 0 von {c['in_c']} Starts" in text and "Fläche des Teils außerhalb 7,173" in text and "gezählt wurden 0 Austritte aus C" in text
    assert c["faelle_aussen"] == c["nicht_in_c"] and c["faelle_innen"] == 0


def test_preset_sicher_ab_7_falls_verdict():
    v = _vals("Wagen: sicher ab N = 7")
    c = _falls(v)
    kind, text = ev.verdict_falls(c, v["wf_N"], v["wf_cost"], v["umax"], v["s1"], v["s2"])
    assert kind == "success" and "Alle 500 Starts liegen in C" in text and "Ab N = N̄ + 1 = 7 ist ein Austritt aus C exakt unmöglich" in text and "unabhängig von den Kosten" in text
    assert "an den Rand" in text and "Garantie gibt es erst" not in text


def test_falls_branches():
    adverse = {"starts": 500, "in_c": 440, "nicht_in_c": 60, "faelle": 383, "faelle_aussen": 60, "faelle_innen": 323, "austritt": 323, "konvergiert": 0, "randlauf": 114, "ende": 3}
    kind, text = ev.verdict_falls(adverse, 3, "grenze", 0.5, 5.0, 5.0)
    assert kind == "warning" and "323 von 440 Starts" in text and "114 Bahnen laufen an den Rand" in text and "gezählt wurden 323 Austritte aus C" in text
    kind, text = ev.verdict_falls(dict(adverse, faelle_innen=2, austritt=2, nicht_in_c=0, in_c=500, faelle_aussen=0, randlauf=0), 6, "grenze", 0.5, 5.0, 5.0)
    assert kind == "warning" and "Bei N = N̄ = 6 gibt es einen Rest" in text and "1/168 = 0,0060" in text and "gezählt wurden 2 Austritte in diesen 500 Starts" in text
    assert "Eine Garantie gibt es erst ab N = 7" in text
    kind, text = ev.verdict_falls(dict(adverse, faelle_innen=0, faelle_aussen=0, nicht_in_c=0, in_c=500, faelle=0, austritt=0, randlauf=0), 8, "q_i_r01", 0.5, 5.0, 5.0)
    assert kind == "success" and "Ab N = N̄ + 1 = 7" in text


def test_preset_terminalmenge_verdict():
    v = _vals("Wagen mit Terminalmenge")
    ta = ev.analyse_terminal(v["tm_term"], v["tm_N"], v["umax"], v["s1"], v["s2"], v["wg_start"])
    counts = L.experiment_terminal(v["tm_term"], v["tm_N"], 100, 716, v["umax"], v["s1"], v["s2"]).counts()
    kind, text = ev.verdict_terminal(ta, counts)
    assert kind == "success"
    for frag in ("O_LQR (R = 1), N = 3: K_N deckt 63,1 % von C (23,51 von 37,29)", "die Terminalmenge selbst (N = 0) 15,1 %", "K_N ist gleich C ab N = 9", "schon bei N̄ = 6",
                 "Der Start (−4,5; 3) liegt nicht in K_3: mit Terminalmenge wird er vor dem Losfahren abgewiesen", "100 Starts in K_N: 0 Fälle, 100 konvergieren"):
        assert frag in text, frag
    kind, text = ev.answer_terminal(ta)
    assert "K_N deckt C ab N = 9" in text and "bei N = 3 deckt K_N 63,1 % von C" in text and "weist ihn am Start ab" in text


def test_terminal_zero_set_reaches_C_only_at_n13():
    ta = ev.analyse_terminal("null", 3, 0.5, 5.0, 5.0, "bbm_a")
    assert ta.n_full == 13 and ta.share == F(4) / F(895, 24)          # K_3({0}) hat die Fläche 4, C die Fläche 895/24
    kind, text = ev.verdict_terminal(ta, None)
    assert "{0} (Ursprung), N = 3: K_N deckt 10,7 % von C (4,00 von 37,29)" in text and "K_N ist gleich C ab N = 13" in text


def test_preset_grobe_stellgrenze_verdict():
    v = _vals("Grobe Stellgrenze (|u| ≤ 5)")
    kind, text = ev.verdict_umax(R.u_table(s2=v["s2"]), v["umax"], v["s1"], v["s2"])
    assert kind == "info"
    for frag in ("|u| ≤ 5, |x₁| ≤ 5, |x₂| ≤ 5: N̄ = 1", "die Fläche von C ist 75,00, 75,0 % der Fläche des Kastens", "N̄ = 9 / 6 / 4 / 3 / 1", "Ab N = N̄ + 1 = 2 bleibt ein Zustand aus C auch ohne Terminalmenge lösbar"):
        assert frag in text, frag
    kind, text = ev.verdict_umax(R.u_table(s2=5.0), 0.25, 10.0, 5.0)
    assert "N̄ = 13" in text and "N̄ = 13 / 9 / 5 / 3 / 1" in text and "1 Schritt" not in text


def test_answers_from_default_settings():
    v = _vals("Von Hand: Skalar a = 2")
    ra = ev.analyse_regions(3, 0.5, 5.0, 5.0, "bbm_b")
    kind, text = ev.answer_solvable(ra)
    assert "Mit Horizont N = 3 ist das Problem für die Zustände in F_3 lösbar (Fläche 42,83)" in text and "(Fläche 37,29, 87,1 % von F_3)" in text and "Ab N̄ = 6" in text
    c = _falls(v)
    kind, text = ev.answer_blame(c, 3, "q_i_r10", 0.5, 5.0, 5.0)
    assert "Aus C heraus fällt keiner der 440 Starts" in text and "(60 von 60)" in text and "Eine Garantie gibt es erst ab N = 7" in text
    adverse = {"starts": 500, "in_c": 440, "nicht_in_c": 60, "faelle": 383, "faelle_aussen": 60, "faelle_innen": 323, "austritt": 323, "konvergiert": 0, "randlauf": 114, "ende": 3}
    kind, text = ev.answer_blame(adverse, 3, "grenze", 0.5, 5.0, 5.0)
    assert "Aus C heraus fallen 323 von 440 Starts: Bei diesen Kosten treiben sie den Zustand aus C" in text


def test_every_cost_is_a_distinct_verdict_line():
    """Zweig-Test: Die fünf Kostenwahlen liefern bei N = 2 nicht alle denselben Austrittszähler (die Wahl wirkt)."""
    seen = {L.experiment_falls(2, k, 200, 716, 0.5, 5.0, 5.0).counts()["austritt"] for k in C.COST_KEYS}
    assert len(seen) >= 3


def test_formatting_helpers():
    assert ev.de(1.5, 2) == "1,50" and ev.de(-3.0, 0) == "−3" and ev.fr(-1.5) == "−3/2" and ev.frd(0.25) == "1/4 (0,25)" and ev.frd(3) == "3"
    assert ev.pt((-4.5, 3.0)) == "(−4,5; 3)" and ev.pt((1.0, 2.0)) == "(1; 2)" and ev.pt((-0.082, -0.04)) == "(−0,082; −0,04)"
    assert ev.pct(0.6310) == "63,1 %" and ev.fmt_int(1215) == "1" + NB + "215"
