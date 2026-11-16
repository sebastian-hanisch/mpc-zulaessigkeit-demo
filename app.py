"""MPC-Zulässigkeit – wann bleibt das Problem lösbar – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zum Thema „Regelung (von LQR bis MPC)“: der Wagen (Doppelintegrator) mit Stellgrenze und Zustandsgrenzen, dazu ein Skalar von Hand.
Gebiete exakt mit Brüchen (mzl_poly.py, mzl_regions.py), Regelkreis mit eigenem QP-Löser (mzl_qp.py, mzl_loop.py); alles eigene Nachrechnung in diesem Modell.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import mzl_constants as C
import mzl_evaluation as ev
import mzl_loop as L
import mzl_poly as P
import mzl_regions as R
from mzl_layout import dunkles_design, kopf_css
from mzl_presets import (
    SETTING_SPECS,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mzl_visualization import (
    build_area_chart,
    build_fall_profile,
    build_falls,
    build_regions,
    build_scalar_line,
    build_scalar_traj,
    build_share_chart,
    build_terminal,
    build_umax,
)

st.set_page_config(page_title="MPC-Zulässigkeit – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _scalar(x0, N, term, cost):
    return ev.analyse_scalar(x0, N, term, cost)


@st.cache_data(show_spinner=False)
def _regions(N, umax, s1, s2, start):
    return ev.analyse_regions(N, umax, s1, s2, start)


@st.cache_data(show_spinner="Rechne die Regelkreise (QP je Takt) …")
def _falls(N, cost, n, seed, umax, s1, s2):
    return L.experiment_falls(N, cost, n, seed, umax, s1, s2)


@st.cache_data(show_spinner=False)
def _terminal(term, N, umax, s1, s2, start):
    return ev.analyse_terminal(term, N, umax, s1, s2, start)


@st.cache_data(show_spinner="Rechne die Regelkreise mit Terminalmenge …")
def _term_run(term, N, n, seed, umax, s1, s2):
    return L.experiment_terminal(term, N, n, seed, umax, s1, s2)


@st.cache_data(show_spinner=False)
def _utable(s2):
    return R.u_table(s2=s2)


@st.cache_data(show_spinner=False)
def _frozen():
    return ev.load_frozen()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mzl_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🚧 MPC-Zulässigkeit – wann bleibt das Problem lösbar")
st.markdown(
    """
**MPC** löst in jedem Takt ein Optimierungsproblem und wendet nur die erste Stellgröße an. Gibt es neben der Stellgrenze auch **Zustandsgrenzen**, kann das Problem in einem Takt **lösbar** und im
nächsten **unlösbar** sein. Die Demo zeigt am Wagen (Doppelintegrator) und an einem Skalar von Hand, **welche Zustände lösbar sind und welche sich für immer im Kasten halten lassen (die Menge C)**,
ob die Regelung einen Verlust hätte verhindern können und **was eine Terminalmenge** davon verhindert und kostet. Die Demo setzt MPC als QP mit Stellgrenzen voraus
([mpc-stellgrenzen-demo](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo)); alle Zahlen sind **eigene Nachrechnung in diesem Modell, bei diesen Grenzen und Kosten**.

Weiter unten: fünf Experimente (🔬), die **Kernfragen** mit Antworten, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Zulässigkeit bei MPC", expanded=True):
    st.markdown(
        """
1. **Das Problem eines Takts:** Von x aus wird eine Stellfolge u₀ … u_(N−1) gesucht, deren vorhergesagte Zustände x₁ … x_N im Kasten bleiben (|xᵢ| ≤ sᵢ) und die |u| ≤ u_max einhält.
   Gibt es keine, ist das Problem **unlösbar**.
2. **Lösbar heute:** F_N ist die Menge der Zustände, für die das gilt. F_N wird mit wachsendem N kleiner, weil jeder Takt eine Bedingung hinzufügt.
3. **Haltbar für immer:** C ist die Menge der Zustände, die sich unter irgendeiner Regelung für immer im Kasten halten lassen (maximale steuerinvariante Menge). Es gilt C ⊆ F_N für jedes N.
4. **Der Haken:** Der Regler wendet nur u₀ an. Der Folgezustand liegt in F_(N−1), muss aber für den nächsten Takt in F_N liegen. Ist F_(N−1) größer als C, kann ein zulässiger erster Schritt aus C hinausführen.
5. **Terminalmenge:** Verlangt man zusätzlich x_N ∈ X_f mit einer steuerinvarianten Menge X_f, bleibt die Lösbarkeit erhalten; das Gebiet K_N(X_f) liegt in C und ist für kleines N kleiner.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Wagen: Grenzen** (gelten für die Experimente 2 bis 5)")
    umax = st.select_slider("Stellgrenze |u| ≤", options=list(C.UMAX_OPTIONS), format_func=lambda v: ev.fr(P.frac(v)), key="umax",
                            help="Die Beschleunigung des Wagens ist auf ±u_max begrenzt. Die Messungen der Demo gelten für 1/2; die anderen Stufen sind Experiment 5.")
    s1 = st.select_slider("Positionsgrenze |x₁| ≤", options=list(C.S_OPTIONS), format_func=lambda v: ev.fr(P.frac(v)), key="s1",
                          help="Der Wagen darf den Kasten |x₁| ≤ s₁ nicht verlassen (die vorhergesagten Positionen x₁ bis x_N).")
    s2 = st.select_slider("Geschwindigkeitsgrenze |x₂| ≤", options=list(C.S_OPTIONS), format_func=lambda v: ev.fr(P.frac(v)), key="s2",
                          help="Der Wagen darf den Kasten |x₂| ≤ s₂ nicht verlassen. Die Hauptmessung hat 5.")
    st.caption("Der Skalar des ersten Experiments hat feste Grenzen (|u| ≤ 1, |x| ≤ 4).")

umax, s1, s2 = float(umax), float(s1), float(s2)
reg = R.regions(umax, s1, s2)

# --- Experiment 1: der Skalar ----------------------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Experiment 1: Von Hand, der Skalar x⁺ = 2x + u")
st.caption("Eine instabile Strecke mit |u| ≤ 1 und |x| ≤ 4: Die Gebiete sind Intervalle und von Hand zu rechnen, F_N = [−f_N; f_N] mit f_N = 1 + 3/2^N, C = [−1; 1]. Terminalmenge: [−1; 1] (Rückführung u = −x).")
c1, c2, c3, c4 = st.columns([1, 1.4, 1.6, 1.2])
sk_N = c1.slider("Horizont N", C.SK_N_MIN, C.SK_N_MAX, key="sk_N", help="N = 0 hat keinen Regler: F₀ ist nur der Kasten.")
sk_x0 = c2.slider("Start x₀", C.SK_X0_MIN, C.SK_X0_MAX, step=C.SK_X0_STEP, format="%.3f", key="sk_x0", help="Vielfache von 1/8; als Bruch exakt gerechnet.")
sk_cost = c3.selectbox("Kosten", C.SK_COST_KEYS, format_func=lambda k: C.SK_COSTS[k]["label"], key="sk_cost",
                       help="Gutartig: der Zustand wird zum Ursprung gezogen. Zur Grenze treibend: ein linearer Anreiz −x belohnt große Zustände (absichtlich ungünstig).")
sk_term = c4.checkbox("Terminalmenge [−1; 1]", key="sk_term", help="Verlangt x_N ∈ [−1; 1]. Diese Menge ist steuerinvariant: aus x gilt mit u = −x wieder x⁺ = x.")
sa = _scalar(float(sk_x0), int(sk_N), bool(sk_term), sk_cost)
k1, k2 = st.columns(2)
k1.plotly_chart(build_scalar_line(sa), width="stretch", key="sk_line")
k2.plotly_chart(build_scalar_traj(sa), width="stretch", key="sk_traj")
kind, text = ev.verdict_scalar(sa)
getattr(st, kind)(text)
st.caption("Links: F_N (bzw. K_N) für N = 0 bis 8, grün das Band C; rechts: die Bahn des Reglers, gestrichelt grün ±C, strichpunktiert blau ±f_N.")

st.markdown("---")

# --- Experiment 2: Wagen, Gebiet gegen Horizont ---------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Experiment 2: Der Wagen, Gebiet gegen Horizont")
st.caption(f"Grenzen: {ev.limits_text(umax, s1, s2)} (Seitenleiste). Die Gebiete sind exakt mit Brüchen gerechnet; die Bahn zeigt den Regler mit den Kosten von Beispiel 12.1 (Q = I, R = 10).")
d1, d2 = st.columns([1.2, 2])
wg_N = d1.slider("Horizont N", C.WG_N_MIN, C.WG_N_MAX, key="wg_N", help="Das Problem sieht N Takte voraus. F_N schrumpft mit N und bleibt ab N̄ gleich C.")
wg_start = d2.selectbox("Startzustand", C.START_KEYS, format_func=lambda k: C.START_POINTS[k]["label"], key="wg_start",
                        help="Die vier Zustände aus Beispiel 12.1 der Quelle. Der Regler fährt vom gewählten Start; fehlt er in F_N, startet er gar nicht.")
ra = _regions(int(wg_N), umax, s1, s2, wg_start)
e1, e2 = st.columns([1.5, 1])
e1.plotly_chart(build_regions(ra), width="stretch", key="reg_plane")
e2.plotly_chart(build_area_chart(ra), width="stretch", key="reg_area")
kind, text = ev.verdict_regions(ra)
getattr(st, kind)(text)
st.caption("Blau: F_N, grün: C, gestrichelt: der Kasten. Zustände in F_N und nicht in C sind lösbar und trotzdem verloren; die Fläche rechts ist exakt (Brüche).")

st.markdown("---")

# --- Experiment 3: wer fällt? -------------------------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Experiment 3: Wer fällt, und wer ist schuld?")
st.caption(f"Starts gleichmäßig in F_N (Ablehnungsverfahren, SplitMix64), Regelkreis ohne Terminalmenge über höchstens {C.T_LOOP} Takte; Grenzen {ev.limits_text(umax, s1, s2)}. "
           "„Fall“: Der Zustand verlässt F_N, dann ist das nächste Problem unlösbar.")
f1, f2, f3, f4 = st.columns([1, 2, 1, 1])
wf_N = f1.slider("Horizont N", C.WF_N_MIN, C.WF_N_MAX, key="wf_N", help="Ab N = N̄ + 1 führt aus C kein zulässiger erster Schritt mehr hinaus (exakt).")
wf_cost = f2.selectbox("Kosten", C.COST_KEYS, format_func=lambda k: C.COSTS[k]["label"], key="wf_cost",
                       help="„Zur Grenze treibend“ ist absichtlich ungünstig gewählt: Ein linearer Anreiz −x₁ zieht die Bahn zur Positionsgrenze. Q = I mit R = 0,1 oder 10 ist ab N = 2 gutartig (in den gezählten Läufen), bei N = 1 nicht.")
wf_n = f3.slider("Zahl der Starts", C.WF_STARTS_MIN, C.WF_STARTS_MAX, step=C.WF_STARTS_STEP, key="wf_n", help="Gleichmäßig in F_N gezogen; je mehr Starts, desto länger die Rechnung.")
wf_seed = f4.number_input("Zufalls-Seed", 0, C.SEED_MAX, key="wf_seed", step=1, help="Legt die Startzustände fest (SplitMix64). Die vorgerechnete Reihe unten gilt für den Standard-Zufalls-Seed.")
f4.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)
fres = _falls(int(wf_N), wf_cost, int(wf_n), int(wf_seed), umax, s1, s2)
counts = fres.counts()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Starts außerhalb von C", f"{counts['nicht_in_c']} von {counts['starts']}", help="Diese Zustände lassen sich unter keinem Gesetz für immer im Kasten halten.")
m2.metric("Fälle, Start außerhalb C", f"{counts['faelle_aussen']}", help="Davon fallen alle, die in den gerechneten Takten ihre Lösbarkeit verlieren.")
m3.metric("Fälle, Start in C", f"{counts['faelle_innen']} von {counts['in_c']}", help="Ein Austritt aus C: Hier haben der Horizont und die Kosten zusammen den Zustand verloren.")
m4.metric("Randläufe", f"{counts['randlauf']}", help="Das QP meldet unlösbar, obwohl der Zustand höchstens 1e-7 außerhalb der exakten Menge liegt; nicht als Fall gezählt.")
g1, g2 = st.columns([1.5, 1])
g1.plotly_chart(build_falls(fres, reg, s1, s2), width="stretch", key="falls_plane")
with g2:
    st.dataframe(pd.DataFrame(ev.falls_rows(fres)), hide_index=True, width="stretch")
    fz = _frozen()
    if ev.frozen_applies(umax, s1, s2, int(wf_n), int(wf_seed), fz):
        st.plotly_chart(build_fall_profile(fz, wf_cost, int(wf_N)), width="stretch", key="falls_profile")
        st.caption("Vorgerechnet (tools/make_frozen.py, Zufalls-Seed und Starts wie eingestellt, Grenzen 1/2 und 5): Fälle gegen N für die gewählten Kosten.")
    else:
        st.caption("Die vorgerechnete Reihe über N gilt nur für |u| ≤ 1/2, |xᵢ| ≤ 5, 500 Starts und den Standard-Zufalls-Seed.")
kind, text = ev.verdict_falls(counts, int(wf_N), wf_cost, umax, s1, s2)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4: Terminalmenge ----------------------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Experiment 4: Terminalmenge, Preis und Gewinn")
st.caption("Die Terminalmenge X_f ist steuerinvariant: Sie bleibt unter der Rückführung des gewählten Gewinns in sich und im Kasten. K_N(X_f) sind die Zustände, die nach N Takten in X_f ankommen können. "
           f"Stufenkosten Q = I, R wie der Gewinn (bei {{0}}: R = 1); Zufalls-Seed wie im Experiment 3 ({int(wf_seed)}), Zahl der Starts wie dort, höchstens {C.TM_STARTS_MAX} "
           "(die Rechnung mit Terminalbedingung ist länger).")
t1, t2 = st.columns([2, 1])
tm_term = t1.selectbox("Terminalmenge", C.TERM_KEYS, format_func=lambda k: C.TERM_LABELS[k], key="tm_term",
                       help="O_LQR: die größte Menge, die unter dem (auf drei Stellen gerundeten) LQR-Gewinn invariant bleibt und beide Grenzen erfüllt. {0}: nur der Ursprung.")
tm_N = t2.slider("Horizont N", C.TM_N_MIN, C.TM_N_MAX, key="tm_N", help="Mit Terminalmenge muss das Problem nach N Takten in X_f enden.")
ta = _terminal(tm_term, int(tm_N), umax, s1, s2, wg_start)
tres = _term_run(tm_term, int(tm_N), min(int(wf_n), C.TM_STARTS_MAX), int(wf_seed), umax, s1, s2)
tcounts = tres.counts()
all_terms = {k: _terminal(k, int(tm_N), umax, s1, s2, wg_start) for k in C.TERM_KEYS}
h1, h2 = st.columns([1.5, 1])
h1.plotly_chart(build_terminal(ta), width="stretch", key="term_plane")
h2.plotly_chart(build_share_chart(all_terms, int(tm_N), reg), width="stretch", key="term_share")
kind, text = ev.verdict_terminal(ta, tcounts)
getattr(st, kind)(text)
fz = _frozen()
if fz.get("terminal"):
    with st.expander("Vorgerechnet: O_LQR (R = 1), je N = 1 bis 10"):
        rows = [{"N": r["N"], "Starts": r["starts"], "Fälle": r["faelle"], "konvergieren": r["konvergiert"], "Wertfunktion nicht um die Stufenkosten gefallen": r["wert_verletzt"]}
                for r in fz["terminal"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("tools/make_frozen.py, Grenzen 1/2 und 5, Zufalls-Seed und Starts wie in `data/mzl_frozen.json` (siehe README). Konvergenz: |x| < 10⁻⁷ in 80 Takten.")

st.markdown("---")

# --- Experiment 5: Stellgrenze und Kasten --------------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Experiment 5: Stellgrenze und Kasten")
st.caption("Wie groß ist C, und wie viele Takte braucht F_N, bis es gleich C ist (N̄)? Die Stellgrenze und die Positionsgrenze stehen in der Seitenleiste; die Tabelle zeigt alle Stufen für |x₂| ≤ "
           f"{ev.fr(P.frac(s2))}.")
table = _utable(s2)
i1, i2 = st.columns([1, 1.2])
with i1:
    st.dataframe(pd.DataFrame([{"|u| ≤": ev.fr(P.frac(r["umax"])), "|x₁| ≤": ev.fr(P.frac(r["s1"])), "N̄": r["n_bar"] if r["n_bar"] is not None else ">60",
                                "Fläche von C": round(float(r["area_C"]), 2), "Anteil am Kasten": float(r["share_X"]) * 100} for r in table]),
                 hide_index=True, width="stretch", column_config={"Anteil am Kasten": st.column_config.NumberColumn(format="%.1f %%")})
i2.plotly_chart(build_umax(table, s1, s2, umax), width="stretch", key="umax_plane")
kind, text = ev.verdict_umax(table, umax, s1, s2)
getattr(st, kind)(text)

sync_query_params({k: st.session_state[k] for k in SETTING_SPECS})

st.markdown("---")

# --- Kernfragen ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Welche Zustände sind lösbar, welche bleiben es für immer?**")
kind, text = ev.answer_solvable(ra)
getattr(st, kind)(text)
st.markdown("**2. Fällt eine Bahn aus C heraus, und wer ist schuld, Horizont oder Kosten?**")
kind, text = ev.answer_blame(counts, int(wf_N), wf_cost, umax, s1, s2)
getattr(st, kind)(text)
st.markdown("**3. Was kostet die Terminalmenge?**")
kind, text = ev.answer_terminal(ta)
getattr(st, kind)(text)
st.caption("Die Antworten folgen den Einstellungen der Experimente 2 bis 4; die Zahlen der README gelten für |u| ≤ 1/2, |xᵢ| ≤ 5 und die dort genannten Kosten.")

st.markdown("---")

# --- Grenzen ------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme der Demo | Was dahinter offen bleibt | Wer setzt an |
|---|---|---|
| **Das Modell ist exakt, keine Störung.** | Eine Störung kann den Zustand aus F_N schieben, auch wenn x in C liegt; robuste Verfahren behandelt die Literatur gesondert (nicht gemessen). | – |
| **Harte Zustandsgrenzen.** | Bei weichen Grenzen mit Wiederherstellung nach Unlösbarkeit ist das Verhalten nicht gemessen; von Hand rechenbar ist nur das Mindestmaß der Überschreitung: Der Bremsweg von (−4,5; 3) endet bei Position 6, 1 über der Grenze 5. | – |
| **Ein System, ein Kasten.** | Der Wagen (und ein Skalar) mit Rechteckgrenzen; andere Systeme (BBM Beispiel 12.2: keine endliche Determiniertheit bis N = 60) nur als Mengen gerechnet, nicht als Regelkreis. | – |
| **Terminalmengen nur LQR-invariant (gerundeter Gewinn) und {0}.** | Keine ellipsoide Menge; andere Kostenformen (Mehrschritt-Gewichte, Endkosten ungleich P) nicht durchgespielt. | – |
| **Lösbarkeit ist nicht Stabilität.** | Mit Terminalmenge ist die Lösbarkeit garantiert; dass die Bahn zum Ursprung läuft, zählt die Demo (500 von 500 Starts in K_N), begründet es hier nicht. | [horizont-falle-demo](https://github.com/sebastian-hanisch/horizont-falle-demo) |
| **Keine Verzögerung.** | Eine Periode Rechenzeit macht den Zustand veraltet und ändert die Gebiete. | [regelung-verzug-demo](https://github.com/sebastian-hanisch/regelung-verzug-demo) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* das Modell des Wagens, die Grenzen $u_{\max}, s_1, s_2$, ein Horizont $N$ und Kosten. *Gesucht:* die Menge $F_N$ der Zustände, für die das Optimierungsproblem lösbar ist,
die Menge $C$, die sich für immer halten lässt, und die Wirkung einer Terminalmenge $X_f$.

| Symbol | Bedeutung |
|---|---|
| $x_k = (x_{k,1}; x_{k,2})$ | Zustand des Wagens in Takt $k$: Position und Geschwindigkeit |
| $u_k$ | Stellgröße (Beschleunigung) in Takt $k$, $\lvert u_k\rvert \le u_{\max}$ |
| $X$ | Kasten $\lvert x_1\rvert \le s_1,\ \lvert x_2\rvert \le s_2$ |
| $N$ | Horizont: Zahl der vorhergesagten Takte |
| $F_N$ | Zustände, für die das Problem mit Horizont $N$ lösbar ist (ohne Terminalmenge) |
| $C$ | maximale steuerinvariante Menge in $X$: $F_N$ für große $N$ |
| $\bar N$ | kleinstes $N$ mit $F_N = F_{N+1}$ |
| $X_f$, $K_N(X_f)$ | Terminalmenge; Zustände, die nach $N$ Takten in $X_f$ ankommen können |
| $Q, R, P_f$ | Gewichte der Stufenkosten und der Endkosten |

**Modell.** Der Wagen ist ein Doppelintegrator mit der Taktzeit 1:

$$x^{+} = A x + B u, \qquad A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \quad B = \begin{pmatrix} 0 \\ 1 \end{pmatrix}.$$

**Das Problem eines Takts** (ohne Terminalmenge: $X_f = \mathbb R^2$) ist für den Zustand $x$ das quadratische Programm

$$\min_{u_0, \dots, u_{N-1}} \ \sum_{k=0}^{N-1} \bigl( x_k^{\top} Q x_k + R u_k^2 \bigr) + x_N^{\top} P_f\, x_N \quad \text{mit } x_0 = x,\ x_{k+1} = A x_k + B u_k,\ \lvert u_k\rvert \le u_{\max},\ x_k \in X \ (k = 1, \dots, N),\ x_N \in X_f .$$

Die Zustandsgrenze gilt für die vorhergesagten Zustände $x_1$ bis $x_N$, der Anfangszustand zählt nicht. (Mit freiem $x_N$ wäre $(1; 2)$ im Beispiel 12.1 der Quelle zulässig; in dieser Lesart reproduziert die Demo die dortige Aussage.)

**Gebiete.** Die Mengen der lösbaren Zustände entstehen rückwärts:

$$F_0 = X, \qquad F_{N+1} = \{\, x \in X : \exists\, u,\ \lvert u\rvert \le u_{\max},\ A x + B u \in F_N \,\}.$$

Mit Terminalmenge beginnt dieselbe Folge bei $K_0 = X_f$. Es gilt $F_{N+1} \subseteq F_N$; $C = F_{\bar N}$ ist steuerinvariant, jeder Zustand aus $C$ hat eine Stellgröße mit $A x + B u \in C$. Die Demo rechnet
die Folge exakt: Die Stellgröße $u$ wird aus den Ungleichungen eliminiert (Fourier-Motzkin), die Eckpunkte stehen als Brüche.

**Austritt aus C.** Das Problem mit Horizont $N$ verlangt nur $x_1 \in F_{N-1}$. Ein zulässiger erster Schritt führt also genau dann aus $C$ hinaus, wenn

$$\bigl( A\,C + B\,[-u_{\max}, u_{\max}] \bigr) \cap F_{N-1} \not\subseteq C .$$

Für $N \ge \bar N + 1$ ist $F_{N-1} = C$, die Menge links liegt dann in $C$: kein zulässiger erster Schritt führt hinaus, unabhängig von den Kosten. Für $N < \bar N$ bleibt bei den Grenzen der App ein Rest; bei $N = \bar N$ nur bei einem Teil der Grenzen (Standardgrenzen: Fläche $1/168$), bei den übrigen ist die Fläche des Teils außerhalb von $C$ schon dort $0$.

**Handbeispiel, Skalar.** Für $x^{+} = 2x + u$, $\lvert u\rvert \le 1$, $\lvert x\rvert \le 4$ ist $F_N = [-f_N, f_N]$ mit

$$f_{N+1} = \min\!\Bigl(4,\ \frac{f_N + 1}{2}\Bigr), \qquad f_N = 1 + \frac{3}{2^N}, \qquad C = [-1, 1].$$

$f_N > 1$ für jedes $N$: eine endliche Determiniertheit gibt es nicht. Für $x_0 = 3/2$ und $N = 2$ liegt jedes erlaubte $x_1 = 3 + u_0$ in $[2; 5/2]$, über $f_2 = 7/4$.

**Handbeispiel, Wagen.** Von $x(0) = (1; 2)$ aus ist $x_1(3) = 7 + 2u_0 + u_1 \ge 11/2 > 5$: unzulässig. Dieselbe Gleichung zeigt, dass $(1; 2)$ nur in $F_2$ liegt.

**QP-Löser.** Die Demo löst das Problem über die Dualform (kleinster Abstand, Lawson und Hanson) mit `scipy.optimize.nnls`; er ist gegen die KKT-Bedingungen und die Aufzählung aller Aktivmengen geprüft.

**Quellen** (aus einem Quellenbericht übernommen, hier nicht erneut im Wortlaut gelesen): Borrelli, Bemporad, Morari, *Predictive Control for Linear and Hybrid Systems*, Fassung vom 26.10.2016,
Kap. 12.3 (S. 251–253; Lemma 12.1 und 12.2, Satz 12.1, Folgerung 12.1 und 12.2) und Beispiele 12.1 und 12.2 (S. 247–251). Rawlings, Mayne, Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2. Aufl., <!-- lint-ok: dezimalpunkt -->
Kap. 2.3 (S. 111–112) und Kap. 2.6 (S. 147) zur rekursiven Lösbarkeit und zur Terminalmenge. Die Messungen dieser Demo sind eigene Rechnung; der Befund „Austritt aus C ab $\bar N + 1$ unmöglich“ ist eigene Nachrechnung
in diesem Modell, die Quelle formuliert ihn über den Determiniertheitsindex, ob mit gleicher Zählung, ist offen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Regelung von LQR bis MPC: vom Plan zur Regel](https://sebastianhanisch.net/konzepte-regelung.html)."
)
