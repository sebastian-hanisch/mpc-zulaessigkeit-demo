# MPC-Zulässigkeit – wann bleibt das Problem lösbar – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-mpc-zulaessigkeit-demo.streamlit.app/)** -->

Diese Demo gehört zum Thema **Regelung (von LQR bis MPC)** auf der Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie setzt MPC als QP mit Stellgrenzen voraus
([`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo)) und hebt dort eine Annahme auf: Mit **Zustandsgrenzen** neben der Stellgrenze kann das Optimierungsproblem eines Takts lösbar und das des nächsten unlösbar sein.
Vehikel ist der **Wagen** (Doppelintegrator, Taktzeit 1) mit Stellgrenze und Zustandsgrenzen, dazu ein **Skalar** von Hand. Nachbarn zum Thema Regelung: [`horizont-falle-demo`](https://github.com/sebastian-hanisch/horizont-falle-demo) (dort die *Stabilität* bei endlichem Horizont; hier die *Zulässigkeit*),
[`windup-pi-demo`](https://github.com/sebastian-hanisch/windup-pi-demo) (Sättigung beim PI-Regler), [`offsetfrei-demo`](https://github.com/sebastian-hanisch/offsetfrei-demo) (Beschränkungen im Beharrungszustand) und
[`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo) (eine Periode Verzug ändert die Gebiete; hier ohne Verzug). Aus verwandten Themen: [`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo) und
[`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) (Rückkopplung am Lager, ohne Beschränkungsgebiete); [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo) klemmt die Bestellung wie hier die Stellgröße, misst aber Chaos statt Zulässigkeit.
Anwendungen mit Zustandsgleichung und Grenzen ohne rollierende Rückkopplung: [`battery-arbitrage-demo`](https://github.com/sebastian-hanisch/battery-arbitrage-demo) und [`energiefahrweise-demo`](https://github.com/sebastian-hanisch/energiefahrweise-demo).

## Warum dieses Problem

MPC wendet in jedem Takt nur die erste Stellgröße der optimalen Folge an und rechnet im nächsten Takt neu. Ob das Problem im nächsten Takt noch eine Lösung hat, hängt nicht nur am heutigen Zustand, sondern an der Zukunft:
**Welche Zustände sind lösbar, welche lassen sich für immer im Kasten halten, fällt eine Bahn heraus, und was verhindert eine Terminalmenge und was kostet sie?**
Am Wagen mit |u| ≤ 1/2 und |xᵢ| ≤ 5 ist der Zustand (−4,5; 3) mit Horizont 3 lösbar, liegt aber außerhalb der Menge C der für immer haltbaren Zustände; der Regler (Q = P = I, R = 10) fährt zwei Takte und scheitert bei (1; 2), weil dort schon die kleinste mögliche Position 11/2 über der Grenze 5 liegt (B2); ob er dorthin fährt, entscheiden die Kosten (B5, B6).
Das Gebiet der lösbaren Zustände schrumpft mit dem Horizont auf C (Fläche 100 bei N = 0, 37,29 bei N = 6; ab N = 6 gleich, B1). Ab N = 7 führt kein zulässiger erster Schritt mehr aus C hinaus, **unabhängig von den Kosten** (exakt, B4); darunter entscheiden die Kosten (B5, B6).
Eine steuerinvariante Terminalmenge macht die Lösbarkeit zur Garantie, kostet aber Horizont: Die LQR-Terminalmenge (R = 1) deckt C erst bei N = 9 ab (B7); gezählt fällt auf K_N für N = 1 bis 10 kein Start heraus (B8). Alle Zahlen sind **eigene Nachrechnung in diesem Modell, bei diesen Grenzen und Kosten**; die Begriffe stehen in der Lehrbuchliteratur, eine Neuheitsaussage gibt es nicht.

## Modell

- **Wagen:** x⁺ = A x + B u mit A = [[1, 1], [0, 1]], B = (0; 1); x₁ ist die Position, x₂ die Geschwindigkeit. Stellgrenze |u| ≤ u_max, Kasten |x₁| ≤ s₁, |x₂| ≤ s₂. Die Zustandsgrenzen gelten für die **vorhergesagten** Zustände x₁ bis x_N, der Anfangszustand zählt nicht.
  In dieser Lesart reproduziert die Demo die Aussage der Quelle zu Beispiel 12.1 (B2); mit freiem x_N wäre (1; 2) zulässig. Hauptmessung: u_max = 1/2, s₁ = s₂ = 5. <!-- lint-ok: dezimalpunkt -->
- **Kosten:** Σ (xᵀQx + R u²) + x_NᵀP_f x_N, dazu optional ein linearer Anteil Σ wᵀx_k. Die Kostenwahlen: Q = I mit R = 0,1 oder 10 (P_f = Q), „nur Position“ (Q = diag(1; 0), R = 1, P_f = Q), „nur Stellaufwand“ (Q = 0, R = 1, P_f = 0) und „zur Grenze treibend“ (Q = 0, R = 1, P_f = 0, dazu −x₁ linear: absichtlich ungünstig gewählt, um zu zeigen, dass die Kosten die Zulässigkeit mitbestimmen).
- **Begriffe:** F_N = Zustände, für die das Problem mit Horizont N lösbar ist (ohne Terminalmenge). C = maximale steuerinvariante Menge im Kasten (Zustände, die sich für immer im Kasten halten lassen). N̄ = kleinstes N mit F_N = F_(N+1); dann ist C = F_N̄.
  K_N(X_f) = lösbar mit der Terminalbedingung x(N) ∈ X_f. O_LQR = maximale positiv invariante zulässige Menge des Regelkreises mit dem auf drei Stellen gerundeten LQR-Gewinn (Q = I, R = 1 oder 0,01).
- **Skalar (Handbeispiel):** x⁺ = 2x + u, |u| ≤ 1, |x| ≤ 4; F_N = [−f_N; f_N] mit f_N = 1 + 3/2^N, C = [−1; 1]; Terminalmenge [−1; 1] (Rückführung u = −x). Kosten „gutartig“ (Q = 1, R = 1, P_f = 1) oder „zur Grenze treibend“ (Q = 0, R = 0,1, P_f = 0, dazu −x linear).
- **Schwellen und Gleichstände:** Ein Zustand gilt als konvergiert, wenn |x|∞ unter 10⁻³ liegt (ohne Terminalmenge) bzw. 10⁻⁷ (mit); eine Bahn verlässt C, wenn eine Kante um mehr als 10⁻⁶ verletzt ist; „Fall“ heißt: der Zustand liegt mehr als 10⁻⁷ außerhalb der exakten Menge F_N.
  Das QP bekommt 10⁻⁹ Lockerung (Zustände auf dem Rand). Meldet es unlösbar, obwohl der Zustand höchstens 10⁻⁷ außerhalb liegt, heißt das **Randlauf** (die Bahn kriecht an den Rand, die Zahl 10⁻⁹ entscheidet nicht mehr); Randläufe werden gezählt, nicht als Fall.
- Die Formeln stehen im Expander 📐 der App. Zufall (Startzustände): ganzzahliger SplitMix64 (`mzl_rng.py`), Zufalls-Seed 716 in den vorgerechneten Reihen.

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Gebiete** (`mzl_poly.py`, `mzl_regions.py`) | Exakte Polygone mit Brüchen: Vorbild unter x⁺ = Ax + Bu per Fourier-Motzkin-Elimination der einen Stellgröße, Ecken per Schnittpunkten je zweier Geraden, überflüssige Ungleichungen über die Ecken; entartete Mengen (Punkt, Strecke) bleiben erhalten | nur zwei Zustände und eine Stellgröße; Rechenzeit wächst mit der Kantenzahl |
| **QP-Löser** (`mzl_qp.py`) | Dualform (kleinster Abstand, Lawson und Hanson) mit `scipy.optimize.nnls`; je Takt ein QP mit N Variablen | eigener, kleiner Löser; Zustände nahe dem Rand siehe Randlauf |
| **Regelkreis** (`mzl_loop.py`) | je Takt QP lösen, erste Stellgröße anwenden; Startzustände gleichmäßig in F_N (Ablehnungsverfahren) | 150 Takte (80 mit Terminalmenge), danach „kein Ereignis“ |
| **Terminalmenge** (`mzl_regions.invariant_lqr`, `mzl_loop.lyapunov_pf`) | O_LQR exakt aus dem gerundeten Gewinn (Brüche); Endkosten P_f aus der Lyapunov-Gleichung A_Kᵀ P A_K − P = −(Q + R KᵀK) | nur LQR-invariant und {0}, keine ellipsoide Menge |
| **Skalar** (`mzl_scalar.py`) | Intervalle exakt (f_(N+1) = min(4, (f_N + 1)/2)), Regelkreis mit demselben QP-Löser | feste Grenzen |

Die Live-Rechnung der App liegt je Eingabe unter etwa 2 Sekunden (Messung auf dem Entwicklungsrechner, nicht festgeschrieben): die exakten Gebiete 0,02 bis 0,4 s (alle Stufen von Experiment 5 zusammen 0,4 s), ein Regelkreis-Experiment mit 500 Starts 0,1 bis 0,7 s, mit 1 000 Starts bis 1,7 s,
die Terminalmenge mit 500 Starts bis etwa 1,1 s (dort sind höchstens 500 Starts möglich). Die Ergebnisse stehen im `st.cache_data`. **Vorgerechnet** ist die Reihe über N in Experiment 3 (fünf Kostenwahlen mal acht Horizonte, 500 Starts, Zufalls-Seed 716) und die Tabelle der Terminalmenge in Experiment 4, dazu die Prüfung der Austritte über zehn Zufalls-Seeds: `data/mzl_frozen.json`,
erzeugt von `tools/make_frozen.py` (etwa eine halbe Minute; `--check` rechnet neu und vergleicht). Die App kennzeichnet sie als „vorgerechnet“ und zeigt sie nur bei den Grenzen, der Zahl der Starts und dem Zufalls-Seed, mit denen sie erzeugt wurden.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: in diesem Modell, bei |u| ≤ 1/2, |xᵢ| ≤ 5 (wo nichts anderes steht) und den genannten Kosten. Gebiete sind **exakt** (Brüche, Bruch gleich Bruch in den Tests); Zählungen aus Regelkreisen kommen aus 500 Starts mit Zufalls-Seed 716 (`data/mzl_frozen.json`), die Tests prüfen sie als Band
(Zählungen ± 2, der Randlauf ± 25, weil er an der letzten Binärstelle hängt: unter Linux fand der Prüflauf 276 statt 281). Die Spalte „Test“ nennt die Datei und die Funktion in `tests/`; „Messung“ heißt: Zahl dieser README aus `tools/make_frozen.py`, im Test auf Konsistenz mit der Neu-Rechnung als Band.

| Nr | Frage | Befund | Test |
|---|---|---|---|
| B1 | Gebiet ohne Terminalmenge | F_N hat die Fläche 100; 75; 219/4 = 54,75; 257/6 = 42,83; 155/4 = 38,75; 75/2 = 37,5; 895/24 = 37,29 für N = 0 bis 6, **N̄ = 6**; C = F_6 hat 14 Ecken und ist exakt steuerinvariant. F_1 ist das Sechseck \|x₁\|, \|x₂\|, \|x₁ + x₂\| ≤ 5 (Fläche 75). C hat 49,7 % der Fläche von F_1, 68,1 % von F_2, 87,1 % von F_3, 96,2 % von F_4 und 99,4 % von F_5: der Rest sind Zustände, die zulässig anfangen und nicht zu halten sind. Skalierung: F_N(10; 1) = 2·F_N(5; 1/2) für N = 0 bis 8, Fläche von C 149,17 gegen 37,29 (Faktor 4). | `test_poly::test_areas_of_F_N_main_measurement`, `test_c_corners_*`, `test_c_is_control_invariant_exactly`, `test_scalar_like_scaling_*`; `test_claims::test_b1_*` |
| B2 | BBM-Zustände | (−4,5; 3) liegt in F_N für N = 0 bis 4, **nicht in C**: Bremst der Wagen dauernd (u = −1/2), liegen die Positionen bei −3/2, 1, 3, 9/2, 11/2, 6, ab dem fünften Takt über 5. (−3/2; 5/2) liegt in F_3, nicht in C; (1; 2) nur in F_2: dort ist x₁(3) = 7 + 2u₀ + u₁ ≥ 11/2 > 5. (−4,5; 2) liegt in C (die Bremsbahn endet bei 1/2). Der Regelkreis mit Horizont 3, Q = P = I, R = 10 fährt aus (−4,5; 3) über (−1,5; 2,5) nach (1; 2) und ist dort unlösbar (Periode 2); aus (−4,5; 2) konvergiert er (nach 12 Takten (−0,082; −0,040), gegen eine SLSQP-Bahn mit Band 10⁻⁴). | `test_poly::test_hand_example_states`, `test_oracle_mpc_zulaessigkeit::test_bbm_example_12_1_trajectories_against_slsqp`, `test_loop::test_bbm_trajectories_status_and_period` |
| B3 | Skalar von Hand | F_N = [−f_N; f_N] mit f_N = 1 + 3/2^N (4; 5/2; 7/4; 11/8; 19/16 …), C = [−1; 1], f_N > 1 für jedes N (nie ein endliches N̄). x₀ = 3/2 liegt in F_2, nicht in F_3: Jedes erlaubte x₁ = 3 + u₀ liegt in [2; 5/2], über f₂ = 7/4. x₀ = 1 liegt in C: x₁ = 5/2 ist für N = 2 zulässig (x₂ = 4), liegt aber über 7/4; mit den „zur Grenze treibenden“ Kosten wählt der Regler es (gemessen x₁ = 2,5), mit den gutartigen Kosten nicht. Terminalmenge [−1; 1]: K_N = [−1; 1] = C für jedes N. | `test_scalar::*`, `test_verdicts::test_scalar_branches` |
| B4 | Hängt der Austritt am Horizont? | Bei den Standardgrenzen kann ein zulässiger erster Schritt aus C hinausführen, solange N ≤ 6 ist: Die Fläche des Teils außerhalb von C ist für N = 1 bis 6 15,006; 12,131; 7,173; 2,214; 0,414 und **1/168 = 0,0060**, **für N ≥ 7 gleich 0** (exakt). Ab N = N̄ + 1 = 7 ist der Austritt also unmöglich, unabhängig von den Kosten, weil F_(N−1) dann schon C ist. Bei anderen Grenzen ist die Fläche je N nachzurechnen: Bei 6 der 20 Grenzkombinationen der App (u_max = 1/4 mit s₁ = 5 oder 10, u_max = 1/2 mit s₁ = 10, je mit s₂ = 5 oder 10) ist sie schon bei N = N̄ gleich 0 (u_max = 1/4, s₁ = s₂ = 5: N̄ = 9, Fläche 1/32 bei N = 8, 0 bei N = 9). Gegenprobe: 3 000 Zufallspaare (x in C, u₀ gleichverteilt, nicht vom Optimierer gewählt), für N = 7 liegt der Folgezustand bei jedem LP-zulässigen u₀ in C. | `test_poly::test_exit_area_is_positive_up_to_n_bar_and_zero_after`, `test_eigenschaften_mpc_zulaessigkeit::test_from_N_bar_plus_one_*` |
| B5 | Wer fällt bei gutartigen Kosten (Q = I)? | Starts außerhalb von C: 235; 151; 60; 14; 3; 0 von je 500 in F_N für N = 1 bis 6. Bei Q = I mit R = 0,1 oder 10 (P = Q) fallen **genau diese Starts und sonst keiner** (N = 2 bis 8): Aus C heraus fällt kein Start (0 von 349 bis 500). Ausnahme N = 1: 2 von 265 Starts in C (R = 0,1) bzw. 201 von 265 (R = 10). Bei „nur Position“ ebenso, außer bei N = 1 (260 von 265) und N = 2 (0 bei Zufalls-Seed 716; bei zehn Zufalls-Seeds von 716 bis 725 verlässt in sechs 1 bis 4 von je 326 bis 352 Starts C). | `test_claims::test_b5_*` (Messung als Band), `test_eigenschaften::test_with_Q_equal_identity_*` |
| B6 | Kosten, die zur Grenze treiben | Austritte aus C bei „nur Stellaufwand“: 260; 228; 102; 18; 0 für N = 1 bis 5 (Starts in C: 265; 349; 440; 486; 497), bei „zur Grenze treibend“: 260; 340; 323; 209; 48; **2**; 0; 0 für N = 1 bis 8. Bei N = 6 ist der Austritt damit **beobachtet**: 2 von 500 Starts (Zufalls-Seed 716), in neun von zehn Zufalls-Seeds (716 bis 725) 1 bis 3; die beiden Austritte des Zufalls-Seeds 716 liegen 0,0009 und 0,0094 über der Kante von C. Für N = 7 sind es in allen zehn Zufalls-Seeds 0. Ab N = 6 laufen die Bahnen an den Rand von C (Randlauf: 498 von 500 bei N = 6, 500 bei N = 7 und 8). | `test_claims::test_b6_*`, `test_eigenschaften::test_at_N_6_an_exit_from_C_exists_*` |
| B7 | Terminalmenge: Preis und Gewinn | O_LQR (R = 1) hat die Fläche 5,615 (15,1 % von C); Anteil von K_N an C für N = 0, 1, 2, 3, 6, 9: 15,1; 34,9; 49,6; 63,1; 92,9; **100 %**. O_LQR (R = 0,01): 2,653 (7,1 %); 7,1; 21,2; 39,3; 53,4; 85,6; 99,8 % bei N = 9, **100 % bei N = 10**. Terminalmenge {0}: 0; 0; 2,7; 10,7; 56,2; 87,0 %, **100 % erst bei N = 13** (K_2({0}) ist ein Parallelogramm der Fläche 1). Das Gebiet ohne Terminalmenge ist schon bei N = 6 gleich C. Die K_N wachsen geschachtelt und liegen in C. | `test_poly::test_terminal_*`, `test_lqr_terminal_sets_*`, `test_eigenschaften::test_K_N_grows_*`, `test_claims::test_b7_*` |
| B8 | Hält die Terminalmenge ihr Versprechen? | Mit O_LQR (R = 1) und Endkosten aus der Lyapunov-Gleichung des gerundeten Gewinns (stimmen mit der DARE auf 5 Stellen überein): je N = 1 bis 10 in 500 Starts in K_N **0 Fälle, 500 von 500 konvergieren** (\|x\| < 10⁻⁷ in 80 Takten), die Wertfunktion nimmt in jedem Schritt um mindestens die Stufenkosten ab (0 Verstöße in 5 000 Läufen, Toleranz 10⁻⁸). | `test_claims::test_b8_*`, `test_loop::test_terminal_runs_*` |
| B9 | Stellgrenze und Kasten | N̄ = 9; 6; 4; 3; 1 für u_max = 1/4; 1/2; 1; 2; 5 bei s₁ = 5 (\|x₂\| ≤ 5) und die Fläche von C 27,36; 37,29; 50; 65; 75 (27,4 bis 75 % des Kastens); bei s₁ = 10: N̄ = 13; 9; 5; 3; 1 und 79,39; 109,44; 145; 165; 175. | `test_poly::test_u_table_matches_the_measurement` |
| B10 | Ein System ohne endliches N̄ (nur die Mengen) | A = [[2, 1], [0, 1/2]], B = (1; 0), \|u\| ≤ 1, \|xᵢ\| ≤ 10: F_N hat bis N = 60 keine endliche Determiniertheit (die Fläche fällt 400; 219,5; 130; 85; 62,5; 51,25 …, bei N = 10 ist sie 40,35, und fällt weiter); der Zustand (−4; 17/2) liegt nur in F_N für N ≤ 3, also nicht in C. Das ist ein Mengenbefund, kein Regelkreis. | `test_poly::test_no_finite_determinedness_index_for_example_12_2_sets` |
| B11 | Der QP-Löser | Gegen die KKT-Bedingungen (Aktivmenge, Multiplikatoren aus kleinsten Quadraten), gegen die Aufzählung aller 3^N Aktivmengen-Muster beim Kasten-QP (N = 1 bis 4), gegen die exakten Polygone und eine LP-Zulässigkeit und im Zielwert gegen SLSQP (die ungenauere Seite): in allen geprüften Zufallszuständen übereinstimmend. | `test_oracle_qp::*` |

### Zahl → Test (Band oder exakt)

| Zahl der README | Test | Art |
|---|---|---|
| Flächen von F_N, N̄ = 6, 14 Ecken, Ecken von C, Steuerinvarianz | `test_poly` | exakt (Brüche) |
| Anteile von C an F_N (49,7 bis 99,4 %), Anteile von K_N an C | `test_claims::test_b1_*`, `test_b7_*` | exakt (Brüche), auf eine Stelle gerundet |
| Austrittsflächen 15,006 … 1/168 | `test_poly::test_exit_area_*` | exakt |
| Starts außerhalb von C, Fälle, Austritte, Randläufe (B5, B6) | `test_claims::test_b5_*`, `test_b6_*` | Band ± 2 (Randlauf ± 25) gegen Neu-Rechnung einzelner Zellen; Konsistenz mit `data/mzl_frozen.json` und der README |
| 500 von 500 konvergieren, 0 Verstöße (B8) | `test_claims::test_b8_*` | Zählung, in der Neu-Rechnung für drei Horizonte |
| 2 von 500 bei N = 6, neun von zehn Zufalls-Seeds | `test_claims::test_b6_*`, `test_eigenschaften::test_at_N_6_*` | Messung (Datei), Zeuge im Test |
| Zeit je Eingabe | – | Messung dieser README, nicht im Test (die Tests prüfen nur, dass die Maximalstellungen durchlaufen) |

### Abweichungen gegenüber den ersten Überschlagsrechnungen

- **Andere Startzustände, gleiche Aussagen.** Eine erste Rechnung zog die Startzustände mit `numpy.random`, die Demo nutzt SplitMix64 (Zufalls-Seed 716); die Zählungen verschieben sich: außerhalb von C 235; 151; 60; 14; 3 (erste Rechnung 245; 158; 64; 19; 3), Starts in C 265; 349; 440; 486; 497; 500 (erste Rechnung 255; 342; 436; 481; 497; 500), „nur Stellaufwand“ 260; 228; 102; 18; 0 (erste Rechnung 252; 217; 107; 12; 0),
  „zur Grenze treibend“ 260; 340; 323; 209; 48; 2; 0 (erste Rechnung 252; 332; 314; 206; 53; 0). Zahlen aus einer ersten Rechnung sind nur Erwartungen; in der Tabelle stehen die Zahlen der Demo. Bei Q = I und N = 1 sind es 2 (R = 0,1) und 201 von 265 (R = 10) statt 3 und 187 von 255.
- **N = 6: Austritt beobachtet.** Eine erste Rechnung fand ihn „nicht beobachtet, nur exakt möglich“. Mit den Startzuständen der Demo tritt er bei „zur Grenze treibenden“ Kosten auf (B6); die exakte Aussage (Fläche 1/168) bleibt, die Beobachtung kommt hinzu. Bei den Standardgrenzen bleibt die Garantie „ab N = 7“ unverändert; für N = 6 sagt die Demo dort nicht „Garantie“. Bei Grenzen, bei denen die Austrittsfläche schon bei N = N̄ gleich 0 ist (B4), sagt sie das auch so.
- **„Nur Position“ bei N = 2.** Eine erste Rechnung fand 2 Austritte bei N = 2 als Ausnahme; mit Zufalls-Seed 716 sind es 0, bei zehn Zufalls-Seeds in sechs 1 bis 4 (B5). Die Ausnahme gilt also, aber nicht für jeden Zufalls-Seed.
- **Randläufe bei N ≥ 6.** Eine erste Rechnung hatte Randläufe für N ≥ 6 bei „zur Grenze treibend“ vermerkt; hier sind es bei N = 6 498, bei N = 7 und 8 alle 500 Starts. Die beiden Austritte bei N = 6 (B6) liegen mit 0,0009 und 0,0094 deutlich über der Schwelle 10⁻⁶ der Austrittsprüfung, sind also kein Rundungsrand.
- **Skalar, „Kosten, die x₁ = 5/2 bevorzugen“.** Eine erste Rechnung nannte keine Kosten. Hier: Q = 0, R = 0,1, P_f = 0 und ein linearer Anteil −x gibt x₁ = 5/2 (gemessen); hier gilt „wählt der Regler es“ nur für diese Kosten und R = 0,1 (gemessen: x₁ = 2,5 bei R = 0,1, aber nur 2,2 bei R = 0,5 und 2,07 bei R = 1,5; Rechnung von Hand zu x₀ = 1 in B3).
- **Layout.** Stellgrenze und Kasten stehen als Einstellungen in der Seitenleiste und gelten für die Experimente 2 bis 5; die Kernfragen stehen nach den fünf Experimenten, vor „Wo die Annahmen enden“.
- **Terminalexperiment: höchstens 500 Starts** (Rechenzeit); die vorgerechnete Tabelle (B8) hat 500.
- **Nicht genannt:** Die Zahl der Regionen des Regelgesetzes in Beispiel 12.3 der Quelle ließ sich bei der Nachrechnung nicht bestätigen (keine Zählweise ergab die Zahlen der Quelle); die Demo nennt sie nicht und sagt nicht, dass sie Beispiel 12.3 bestätigt. <!-- lint-ok: dezimalpunkt -->

## Quellen

Aus einem Quellenbericht übernommen, **hier nicht erneut im Wortlaut gelesen**; die Gleichungen und Befunde stehen in eigenen Worten.

- Borrelli, Bemporad, Morari, *Predictive Control for Linear and Hybrid Systems*, Fassung vom 26.10.2016: Kap. 12.3 (S. 251–253) zur dauerhaften Zulässigkeit (Lemma 12.1 und 12.2, Satz 12.1, Folgerungen 12.1 und 12.2); Beispiel 12.1 (S. 247–248): Doppelintegrator mit N = 3, P = Q = I, R = 10, |u| ≤ 0,5, |xᵢ| ≤ 5 ohne Terminalmenge, <!-- lint-ok: dezimalpunkt -->
  aus (−4,5; 2) konvergiert die Bahn, aus (−4,5; 3) bricht sie ab (die Demo rechnet es nach, B2); Beispiel 12.2 (S. 248–251): instabiles System, nur die Mengen sind nachgerechnet (B10), die Aussage zu „jedem Anfangszustand“ nicht. <!-- lint-ok: dezimalpunkt -->
- Rawlings, Mayne, Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2. Aufl.: Kap. 2.3 (S. 111–112) und Kap. 2.6 (S. 147) zur rekursiven Lösbarkeit mit steuerinvarianter Terminalmenge; S. 131–132 zu harten Zustandsgrenzen und weichen Grenzen (nicht gemessen). **Das PDF enthält den Vermerk „Weitergabe nur mit Erlaubnis des Verlags“; nichts davon ist wörtlich übernommen.**
- **Eigene Nachrechnung in diesem Modell:** alle Zahlen der Befunde; insbesondere „ab N̄ + 1 ist der Austritt aus C unmöglich“ (B4): Die Quelle formuliert dieselbe Idee über den Determiniertheitsindex; ob mit gleicher Zählung, ist offen.
  Die Demo behauptet keine Neuheit.

## Ehrliche Grenzen

- **Nur dieses Modell.** Ein Wagen (Doppelintegrator) und ein Skalar, Rechteckgrenzen, Taktzeit 1, kein Modellfehler und keine Störung: Eine Störung kann einen Zustand aus C in F_N schieben und die Lösbarkeit zusätzlich brechen; robuste Verfahren sind hier nicht gemessen.
- **Harte Zustandsgrenzen.** Das Verhalten bei weichen Grenzen mit Wiederherstellung nach einer Unlösbarkeit ist **nicht gemessen**. Von Hand rechenbar ist nur das Mindestmaß der Überschreitung: Der Bremsweg von (−4,5; 3) endet bei Position 6, 1 über der Grenze 5.
- **Die Kosten bestimmen die Bahn, nicht das Gebiet.** F_N und C hängen nur an den Grenzen und N; ob ein Start aus C herausfällt, hängt an den Kosten (B5 gegen B6). Andere Kostenformen (Mehrschritt-Gewichte, Endkosten ungleich P) sind nicht durchgespielt. „Zur Grenze treibend“ ist **absichtlich ungünstig**; kein realer Entwurf.
- **Stichproben, keine Garantien aus Zählungen.** Die Zählungen B5 und B6 sind 500 Starts je Zelle; „0 Austritte“ aus einer Zählung heißt nicht „unmöglich“ (B6 bei N = 6 zeigt es: exakt möglich, im Zufalls-Seed 716 selten, in einem von zehn Zufalls-Seeds nicht beobachtet). Die Garantie ab N̄ + 1 (bei den Standardgrenzen N = 7) stützt sich auf die exakte Rechnung (B4), nicht auf eine Zählung.
- **Randlauf.** Bei Kosten, die an den Rand treiben, entscheidet das QP dort bei 10⁻⁹ nicht mehr; diese Bahnen sind gezählt, nicht als Fall oder Nichtfall gewertet (bei „zur Grenze treibend“ ab N = 6 fast alle).
- **Lösbarkeit ist nicht Stabilität.** Ohne Terminalmenge ist die Konvergenz nur gezählt (bei Q = I und N ≥ 2 konvergieren alle Starts in C, 349 bis 500 je nach N), nicht begründet ([`horizont-falle-demo`](https://github.com/sebastian-hanisch/horizont-falle-demo) behandelt die Stabilität). Mit Terminalmenge ist die Lösbarkeit garantiert; die Konvergenz zählt die Demo (B8).
- **Terminalmengen** nur LQR-invariant (auf drei Stellen gerundeter Gewinn, damit exakt) und {0}; keine ellipsoide Menge.
- **Die Lesart der Zustandsgrenze bei x_N** (hier: x_N im Kasten) und die Zählweise des Determiniertheitsindex der Quelle sind für einen Vergleich mit anderen Beispielen der Quelle nicht geklärt.
- **Mit Stufen statt Reglern.** Stellgrenze und Kasten gibt es nur in den Stufen der App (u_max 1/4 bis 5, Kasten 5 oder 10); andere Werte sind im Code möglich, aber nicht gemessen.
- **Keine Rechenzeit festgeschrieben.** Die Zeiten der Methodik sind eine Messung auf dem Entwicklungsrechner.

## Verifikation

- **Kern:** Exakte Polygone gegen Handrechnung (Flächen, Ecken, entartete Terminalmenge {0} mit Regressionstest „K_2({0}) hat Fläche 1“), Skalar gegen LP an 5 × 161 Zuständen, QP-Löser gegen Handwerte (`tests/test_qp.py`, `test_scalar.py`, `test_poly.py`).
- **Orakel:** `tests/test_oracle_mpc_zulaessigkeit.py` (exakte Gebiete gegen LP-Zulässigkeit an Zufallspunkten, C steuerinvariant und maximal, BBM-Bahn gegen SLSQP, gerundeter Gewinn gegen DARE) und `tests/test_oracle_qp.py` (KKT, Aufzählung der Aktivmengen, Zulässigkeit, Zielwert gegen SLSQP).
- **Eigenschaften und Gegenprobe:** `tests/test_eigenschaften_mpc_zulaessigkeit.py`: F_(N+1) ⊆ F_N, C ⊆ F_N, K_N geschachtelt in C, Skalierung und Spiegelung, Fläche von C wächst mit den Grenzen, Kostenskalierung, die Gegenprobe „ab N = 7“ mit 3 000 Zufallspaaren und ein Zeuge für den Austritt bei N = 6.
- **Claims, Verdikte, Randfälle:** `tests/test_claims.py` (jede Zahl der Befunde), `test_verdicts.py` (je Preset die erwartete Zeile), `test_edge_cases.py` (N = 0, N = 1, jede Stufe, Terminalmenge {0}).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen, jede Option), Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jeder Regler hebt die Markierung auf), Footer, Laufzeit-Importe.
- **Tests:** 396 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Kernfragen, Annahmen, Formeln |
| `mzl_poly.py` | exakte Polytope im Zustandsraum (Brüche, Fourier-Motzkin) |
| `mzl_regions.py` | F_N, C, N̄, Terminalmengen, K_N, Austrittsflächen, Tabelle der Stellgrenzen |
| `mzl_qp.py` | QP-Löser (Dualform, NNLS) und das MPC-Problem in verdichteter Form |
| `mzl_loop.py` | Regelkreis, Startzustände, Experimente „Wer fällt“ und Terminalmenge |
| `mzl_scalar.py` | Handbeispiel Skalar |
| `mzl_rng.py` | SplitMix64 (Kopie der geprüften Vorlage des Website-Repos) |
| `mzl_constants.py` | Modell, Reglerstufen, Kosten, Presets, Farben |
| `mzl_evaluation.py` | Auswertung und Verdikte (reine Funktionen) |
| `mzl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `mzl_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `mzl_layout.py` | Fester Kopf (Stylesheet) |
| `tools/make_frozen.py` | erzeugt `data/mzl_frozen.json` (vorgerechnete Reihen) |
| `data/mzl_frozen.json` | vorgerechnete Zählungen (Zufalls-Seed 716) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit-Bibliotheken (ungepinnt), dazu pytest |
| `tests/` | Handwerte, Orakel, Eigenschaften, Claims, Verdikte, Randfälle, AppTest-Rauchtest, Layout, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Weiche Zustandsgrenzen mit Wiederherstellung** (die Quelle rät dazu): erst nach einer eigenen Messung, nicht gemessen.
- **Störung, robuste Verfahren, stochastisches Rauschen:** Die Demo hat kein Rauschen.
- **Andere Systeme als der Wagen** (außer den Mengen von B10) und **ellipsoide Terminalmengen.**
- **Ein eigenes Experiment zu BBM Beispiel 12.2** und zur Regionenzahl des Regelgesetzes: die Zählweise der Quelle ist ungeklärt. <!-- lint-ok: dezimalpunkt -->
- **ADMM, Warmstart, schnelle Löser:** der QP-Löser ist klein und exakt genug; Rechenzeit ist hier nicht das Thema.
- **PDF-Export:** nicht gebaut.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Regelung von LQR bis MPC: vom Plan zur Regel](https://sebastianhanisch.net/konzepte-regelung.html).
