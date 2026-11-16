"""Konstanten: Modell, Reglergrenzen, Kosten, Presets, Farben (Thema Regelung: Zulässigkeit).

Modell des Wagens: x⁺ = A x + B u mit A = [[1, 1], [0, 1]], B = (0; 1) (Doppelintegrator, Ts = 1). Handbeispiel: der Skalar x⁺ = 2 x + u.
Zufall nur über `mzl_rng.SplitMix64` (ganzzahlig, plattformfest)."""

from fractions import Fraction as F

# --- Modell --------------------------------------------------------------------------------------------------------------------------------
A_CAR = ((F(1), F(1)), (F(0), F(1)))
B_CAR = (F(0), F(1))
SK_A, SK_UMAX, SK_S = 2, 1, 4                     # Skalar: x⁺ = 2x + u, |u| ≤ 1, |x| ≤ 4
SK_TERM = F(1)                                     # steuerinvariante Terminalmenge des Skalars: [−1, 1] (Rückführung u = −x, x⁺ = x)

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
UMAX_OPTIONS = (0.25, 0.5, 1.0, 2.0, 5.0)         # Stellgrenze |u| ≤ u_max des Wagens
S_OPTIONS = (5.0, 10.0)                            # Grenze für |x₁| bzw. |x₂|
DEFAULT_UMAX, DEFAULT_S1, DEFAULT_S2 = 0.5, 5.0, 5.0
SK_N_MIN, SK_N_MAX, SK_DEFAULT_N = 0, 8, 2
SK_X0_MIN, SK_X0_MAX, SK_X0_STEP, SK_DEFAULT_X0 = -4.0, 4.0, 0.125, 1.5
WG_N_MIN, WG_N_MAX, DEFAULT_WG_N = 0, 14, 6
WF_N_MIN, WF_N_MAX, DEFAULT_WF_N = 1, 10, 3
WF_STARTS_MIN, WF_STARTS_MAX, WF_STARTS_STEP, DEFAULT_WF_STARTS = 100, 1000, 100, 500
SEED_MAX, DEFAULT_SEED = 999999, 716
TM_N_MIN, TM_N_MAX, DEFAULT_TM_N = 1, 10, 6
TM_STARTS_MAX = 500                                # Starts im Terminalmengen-Experiment höchstens (Rechenzeit unter etwa 2 s)
N_BAR_MAX = 60                                     # Abbruch der Gebietsfolge (BBM-Beispiel 12.2 hat keine endliche Determiniertheit)
T_LOOP = 150                                       # Perioden je Regelkreis ohne Terminalmenge
T_LOOP_TERM = 80                                   # mit Terminalmenge
CONV_TOL = 1e-3                                    # |x|∞ unter dieser Schwelle gilt als konvergiert (ohne Terminalmenge)
CONV_TOL_TERM = 1e-7                               # mit Terminalmenge
EXIT_TOL = 1e-6                                    # eine Bahn verlässt C, wenn eine Kante um mehr als 1e-6 verletzt ist
REGION_TOL = 1e-7                                  # Zustand höchstens so weit außerhalb der exakten Menge: Randlauf statt Fall

# --- Kosten (Q, R, P_f, linearer Anteil) -----------------------------------------------------------------------------------------------------
COSTS = {
    "q_i_r01": {"label": "Q = I, R = 0,1", "q": ((1.0, 0.0), (0.0, 1.0)), "r": 0.1, "lin": None},
    "q_i_r10": {"label": "Q = I, R = 10 (wie Beispiel 12.1)", "q": ((1.0, 0.0), (0.0, 1.0)), "r": 10.0, "lin": None},
    "pos": {"label": "nur Position (Q = diag(1; 0), R = 1)", "q": ((1.0, 0.0), (0.0, 0.0)), "r": 1.0, "lin": None},
    "stell": {"label": "nur Stellaufwand (Q = 0, R = 1)", "q": ((0.0, 0.0), (0.0, 0.0)), "r": 1.0, "lin": None},
    "grenze": {"label": "zur Grenze treibend (Q = 0, R = 1, dazu −x₁ linear)", "q": ((0.0, 0.0), (0.0, 0.0)), "r": 1.0, "lin": (-1.0, 0.0)},
}
COST_KEYS = tuple(COSTS)
DEFAULT_COST_FALL = "q_i_r10"
SK_COSTS = {
    "gut": {"label": "gutartig (Q = 1, R = 1)", "q": 1.0, "r": 1.0, "p": 1.0, "lin": None},
    "grenze": {"label": "zur Grenze treibend (Q = 0, R = 0,1, dazu −x linear)", "q": 0.0, "r": 0.1, "p": 0.0, "lin": (-1.0,)},
}
SK_COST_KEYS = tuple(SK_COSTS)

# --- Terminalmengen des Wagens ----------------------------------------------------------------------------------------------------------------
# Gerundete LQR-Gewinne (u = −K x, Q = I) als Brüche (drei Nachkommastellen; damit sind die invarianten Mengen exakt). Gegenprobe: tests/test_oracle_lqr.py.
LQR_GAINS = {
    "1": (F(211, 500), F(311, 250)),               # R = 1:    K = (0,422; 1,244)
    "0.1": (F(579, 1000), F(773, 500)),            # R = 0,1:  K = (0,579; 1,546)
    "0.01": (F(307, 500), F(161, 100)),            # R = 0,01: K = (0,614; 1,610)
}
LQR_R = {"1": 1.0, "0.1": 0.1, "0.01": 0.01}
TERM_KEYS = ("lqr1", "lqr001", "null")
TERM_LABELS = {"lqr1": "O_LQR (R = 1)", "lqr001": "O_LQR (R = 0,01)", "null": "{0} (Ursprung)"}
TERM_R = {"lqr1": "1", "lqr001": "0.01"}
DEFAULT_TERM = "lqr1"

# --- BBM-Zustände (Beispiel 12.1 und 12.2 der Quelle) ----------------------------------------------------------------------------------------
START_POINTS = {
    "bbm_a": {"label": "(−4,5; 3): Beispiel 12.1, bricht ab", "x": (F(-9, 2), F(3))},
    "bbm_b": {"label": "(−4,5; 2): Beispiel 12.1, konvergiert", "x": (F(-9, 2), F(2))},
    "bbm_c": {"label": "(−1,5; 2,5): erster Schritt der Falle", "x": (F(-3, 2), F(5, 2))},
    "bbm_d": {"label": "(1; 2): dritter Zustand der Falle", "x": (F(1), F(2))},
}
START_KEYS = tuple(START_POINTS)
DEFAULT_START = "bbm_b"

# --- Farben ------------------------------------------------------------------------------------------------------------------------------------
COLOR_C = "#2F6B65"
COLOR_F = "rgba(76,120,168,0.35)"
COLOR_FLINE = "#4c78a8"
COLOR_X = "rgba(150,150,150,0.8)"
COLOR_OK = "#2F6B65"
COLOR_FALL = "#c0392b"
COLOR_EXIT = "#e08a1e"
COLOR_EDGE = "#7f7f7f"
COLOR_K = "rgba(224,138,30,0.35)"
COLOR_KLINE = "#e08a1e"

# --- Presets -------------------------------------------------------------------------------------------------------------------------------------
# Jeder Preset legt nur fest, was er braucht; alles Übrige gilt als Standardwert (mzl_presets.aktives_preset).
PRESETS = {
    "Von Hand: Skalar a = 2": {},
    "Wagen: die Falle (BBM 12.1, N = 3)": {"wg_N": 3, "wg_start": "bbm_a", "wf_N": 3, "wf_cost": "q_i_r10"},  # lint-ok: dezimalpunkt
    "Wagen: sicher ab N = 7": {"wg_N": 7, "wg_start": "bbm_b", "wf_N": 7, "wf_cost": "grenze"},
    "Wagen mit Terminalmenge": {"wg_N": 3, "wg_start": "bbm_a", "wf_N": 3, "tm_N": 3, "tm_term": "lqr1"},
    "Grobe Stellgrenze (|u| ≤ 5)": {"umax": 5.0},
}
PRESET_HELP = {
    "Von Hand: Skalar a = 2": "Der Skalar x⁺ = 2x + u mit |u| ≤ 1, |x| ≤ 4, Horizont N = 2 und Start x₀ = 3/2 (Handbeispiel); darunter die Wagen-Experimente mit ihren Standardwerten.",
    "Wagen: die Falle (BBM 12.1, N = 3)": "Der Wagen mit |u| ≤ 1/2, |xᵢ| ≤ 5, Horizont 3, Kosten Q = I, R = 10 und Start (−4,5; 3): lösbar, aber nicht in C; der Regler fährt zwei Takte und scheitert.",  # lint-ok: dezimalpunkt
    "Wagen: sicher ab N = 7": "Horizont 7 und Kosten, die absichtlich zur Positionsgrenze treiben: Das Gebiet F₇ ist schon C, kein zulässiger erster Schritt führt hinaus.",
    "Wagen mit Terminalmenge": "Horizont 3 mit der LQR-Terminalmenge (R = 1): K₃ deckt nur einen Teil von C ab, der Start (−4,5; 3) wird schon vor dem Losfahren abgewiesen.",
    "Grobe Stellgrenze (|u| ≤ 5)": "Mit |u| ≤ 5 reicht ein Schritt (N̄ = 1): C ist das Sechseck F₁, ab N = 2 bleibt ein Zustand aus C auch ohne Terminalmenge lösbar.",
}
