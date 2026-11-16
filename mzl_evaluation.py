"""Auswertung und Verdikte: aus den Gebieten (mzl_regions), den Regelkreisen (mzl_loop) und dem Skalar (mzl_scalar) die Zahlen und Sätze der App.

Jedes Verdikt ist eine reine Funktion `verdict_*(…) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`. Alle Zahlen im Text kommen aus den übergebenen Analysen,
keine ist fest eingetragen. Gültigkeit überall: dieses Modell (Wagen mit Stellgrenze und Zustandsgrenzen, Skalar-Handbeispiel), die gewählten Grenzen und Kosten.
Zufall nur über `mzl_rng.SplitMix64` (Zufalls-Seed der Startzustände)."""

import json
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path

import numpy as np

import mzl_constants as C
import mzl_loop as L
import mzl_poly as P
import mzl_regions as R
import mzl_scalar as S

ROOT = Path(__file__).resolve().parent
FROZEN_PATH = ROOT / "data" / "mzl_frozen.json"


# --- Formatierung ------------------------------------------------------------------------------------------------------------------------------

def de(x, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma; ganze Zahlen ohne Nachkommastellen, wenn nd = 0."""
    return f"{float(x):.{nd}f}".replace(".", ",").replace("-", "−")


def fr(x) -> str:
    """Bruch als Text mit Minuszeichen: 7/4, −3/2, 5."""
    return P.fmt_frac(x).replace("-", "−")


def frd(x, nd: int = 2) -> str:
    """Bruch mit Dezimalwert: ‚7/4 (1,75)‘; ganze Zahlen nur als Zahl."""
    x = P.frac(x)
    return fr(x) if x.denominator == 1 else f"{fr(x)} ({de(x, nd)})"


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", " ")        # geschütztes Leerzeichen als Tausendertrenner


def pt(p) -> str:
    """Zustand als ‚(−4,5; 3)‘ (höchstens drei Nachkommastellen, ohne überflüssige Nullen)."""
    def one(v):
        t = f"{float(v):.3f}".rstrip("0").rstrip(".")
        return t.replace(".", ",").replace("-", "−")

    return "(" + "; ".join(one(v) for v in p) + ")"


def pct(x, nd: int = 1) -> str:
    return de(float(x) * 100, nd) + " %"


def umax_text(u) -> str:
    u = P.frac(u)
    return fr(u)


def limits_text(umax, s1, s2) -> str:
    s1, s2 = P.frac(s1), P.frac(s2)
    if s1 == s2:
        return f"|u| ≤ {umax_text(umax)}, |xᵢ| ≤ {fr(s1)}"
    return f"|u| ≤ {umax_text(umax)}, |x₁| ≤ {fr(s1)}, |x₂| ≤ {fr(s2)}"


# --- Gefrorene Messreihen (data/mzl_frozen.json, erzeugt von tools/make_frozen.py) -----------------------------------------------------------------

def load_frozen() -> dict:
    try:
        return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def frozen_applies(umax, s1, s2, n_starts, seed, frozen=None) -> bool:
    """Die vorgerechneten Reihen gelten nur für die Grenzen, die Zahl der Starts und den Zufalls-Seed, mit denen sie erzeugt wurden."""
    fz = load_frozen() if frozen is None else frozen
    p = fz.get("params")
    return bool(p) and (float(umax), float(s1), float(s2), int(n_starts), int(seed)) == (p["umax"], p["s1"], p["s2"], p["n_starts"], p["seed"])


# --- Experiment 1: der Skalar ------------------------------------------------------------------------------------------------------------------

@dataclass
class ScalarAnalysis:
    x0: F
    N: int
    terminal: bool
    cost_key: str
    klass: str                # aussen | in_FN_nicht_C | in_C
    f_list: list              # Halbbreiten f_0 … f_max(N, 8)
    f_N: F
    c: F
    nxt: tuple | None         # erlaubtes Intervall für x₁
    run: L.Run | None
    k_fail: int | None


def analyse_scalar(x0: float, N: int, terminal: bool, cost_key: str, T: int = 12) -> ScalarAnalysis:
    x0f = P.frac(x0)
    start = C.SK_TERM if terminal else None
    f_list = S.f_seq(max(N, C.SK_N_MAX), start=start)
    fN = f_list[N]
    klass = S.classify(x0f, N, terminal) if N >= 0 else "aussen"
    run = S.run(float(x0f), N, cost_key, terminal, T) if (N >= 1 and klass != "aussen") else None
    return ScalarAnalysis(x0f, N, terminal, cost_key, klass, f_list, fN, S.c_halb(), S.next_interval(x0f, N, terminal) if N >= 1 else None, run,
                          run.k_stop if run is not None and run.status == "verloren" else None)


def _set_name(a: ScalarAnalysis) -> str:
    return "K" if a.terminal else "F"


def verdict_scalar(a: ScalarAnalysis) -> tuple:
    """Verdikt zum Skalar x⁺ = 2x + u (|u| ≤ 1, |x| ≤ 4): lösbar? in C? Bahn des Reglers."""
    nm = _set_name(a)
    cl = f"C = [−{fr(a.c)}; {fr(a.c)}]"
    base = f"[−{fr(a.f_N)}; {fr(a.f_N)}]"
    if a.N == 0:
        return "info", f"N = 0: Es gibt kein Problem zu lösen. {nm}₀ = {base} ist nur der Kasten; ein Regler braucht N ≥ 1."
    if a.klass == "aussen":
        return "error", (f"x₀ = {fr(a.x0)} liegt außerhalb von {nm}_{a.N} = {base} (Halbbreite {frd(a.f_N)}): schon das erste Problem ist unlösbar"
                         + (", die Terminalmenge weist den Start vor dem Losfahren ab." if a.terminal else "."))
    run = a.run
    if a.klass == "in_FN_nicht_C":
        if run.status == "verloren":
            extra = ""
            if a.k_fail == 1 and a.nxt is not None and a.nxt[0] > a.f_N:
                extra = (f" Jedes erlaubte x₁ liegt in [{fr(a.nxt[0])}; {fr(a.nxt[1])}], über {frd(a.f_N)}: der nächste Takt ist für jede Wahl unzulässig.")
            return "warning", (f"x₀ = {fr(a.x0)} ist lösbar (in {nm}_{a.N} = {base}), liegt aber außerhalb von {cl}: Kein Regelgesetz hält ihn für immer im Kasten. "
                               f"Der Regler fährt {a.k_fail} Takt(e) und scheitert.{extra}")
        return "info", (f"x₀ = {fr(a.x0)} liegt außerhalb von {cl}, der Regler ist aber noch nach {len(run.traj) - 1} gerechneten Takten lösbar: "
                        f"nicht für immer, nur bis die Bahn den Kasten verlässt.")
    # in C
    if run.status == "verloren":
        x1 = float(run.traj[1][0]) if len(run.traj) > 1 else float("nan")
        return "warning", (f"x₀ = {fr(a.x0)} liegt in {cl} und scheitert trotzdem: Der Regler wählt bei diesen Kosten x₁ = {de(x1, 3)}, das über {frd(a.f_N)} "
                           f"(Halbbreite von {nm}_{a.N}) liegt. Aus C führt ein zulässiger erster Schritt hinaus, solange das Ziel {nm}_{a.N - 1} größer als C ist; "
                           f"hier entscheiden die Kosten. Im Skalar ist f_N > 1 für jedes N, ein N ohne diese Möglichkeit gibt es nicht.")
    if a.terminal:
        return "success", (f"x₀ = {fr(a.x0)} liegt in C, mit Terminalmenge X_f = {cl} ist K_N = C für jedes N: Der Start ist lösbar und bleibt es, "
                           f"nach {len(run.traj) - 1} gerechneten Takten.")
    return "success", (f"x₀ = {fr(a.x0)} liegt in {cl} und bleibt bei diesen Kosten in den {len(run.traj) - 1} gerechneten Takten lösbar. "
                       f"Das ist keine Garantie: Im Skalar ist f_N > 1 für jedes N, andere Kosten führen aus C hinaus (Preset: zur Grenze treibend).")


# --- Experiment 2: der Wagen, Gebiet gegen Horizont ---------------------------------------------------------------------------------------------

BBM_COST = "q_i_r10"      # Q = I, R = 10, P = Q: die Kosten von Beispiel 12.1 der Quelle


@dataclass
class RegionAnalysis:
    N: int
    umax: float
    s1: float
    s2: float
    reg: R.Regions
    areas: list               # exakte Flächen F_0 … F_N̄ (oder bis zum Ende der Folge)
    n_vertices_c: int
    start_key: str
    start: tuple
    n_max_in: int | None      # größtes N mit Start ∈ F_N (None: nirgends; bei Start in C: n_bar)
    start_in_c: bool
    start_in_fN: bool
    run: L.Run | None
    blame: str | None


def min_position_after(x, N: int, umax) -> F:
    """Kleinste Position nach N Takten bei dauerndem Bremsen (u = −u_max) ohne Geschwindigkeitsgrenze: x₁ + N·x₂ − u_max·N(N−1)/2 (Handrechnung)."""
    um = P.frac(umax)
    return P.frac(x[0]) + N * P.frac(x[1]) - um * F(N * (N - 1), 2)


def analyse_regions(N: int, umax: float, s1: float, s2: float, start_key: str) -> RegionAnalysis:
    reg = R.regions(umax, s1, s2)
    start = C.START_POINTS[start_key]["x"]
    areas = [reg.area(n) for n in range(len(reg.seq))]
    in_seq = [P.contains(c, start) for c in reg.seq]
    in_c = in_seq[-1] if reg.n_bar is not None else False
    n_max_in = None
    if any(in_seq):
        n_max_in = (reg.n_bar if reg.n_bar is not None else len(reg.seq) - 1) if in_seq[-1] else max(i for i, v in enumerate(in_seq) if v)
    start_in_fN = P.contains(reg.F(N), start)
    run, blame = None, None
    if N >= 1 and start_in_fN:
        prob = L.cost_problem(N, BBM_COST, umax, s1, s2)
        run = L.run_loop(prob, P.to_float(reg.F(N)), [float(start[0]), float(start[1])], 60)
        if run.status == "verloren":
            xe = run.traj[-1]
            xf = (F(float(xe[0])).limit_denominator(10000), F(float(xe[1])).limit_denominator(10000))     # Gleitkomma auf den nächsten einfachen Bruch (1/10 000)
            mp = min_position_after(xf, N, umax)
            if mp > P.frac(s1):
                blame = f"Auch bei dauerndem Bremsen (u = −{fr(umax)}) liegt die Position nach {N} Takten bei {fr(mp)}, über der Grenze {fr(s1)}."
    return RegionAnalysis(N, umax, s1, s2, reg, areas, len(P.vertices(reg.Cset)), start_key, start, n_max_in, in_c, start_in_fN, run, blame)


def verdict_regions(a: RegionAnalysis) -> tuple:
    reg = a.reg
    lim = limits_text(a.umax, a.s1, a.s2)
    if reg.n_bar is None:
        first = (f"Bis N = {len(reg.seq) - 1} ist keine endliche Determiniertheit erreicht ({lim}): F_N schrumpft weiter, die Fläche fällt von {de(a.areas[0])} auf {de(a.areas[-1])}.")
    else:
        flaechen = "; ".join(de(x) for x in a.areas)
        first = (f"{lim}: F_N schrumpft mit N, die Fläche von N = 0 bis N = N̄ = {reg.n_bar} ist {flaechen}; bei N̄ ist F_N gleich C "
                 f"({a.n_vertices_c} Ecken, exakt steuerinvariant).")
    nm = C.START_POINTS[a.start_key]["label"].split(":")[0]
    if a.n_max_in is None:
        second = f" Der Start {nm} liegt in keinem F_N, auch nicht in F_0 (außerhalb des Kastens)."
    elif a.start_in_c:
        second = f" Der Start {nm} liegt in C, also in F_N für jedes N."
    else:
        second = f" Der Start {nm} liegt in F_N für N = 0 bis {a.n_max_in}, aber nicht in C."
    if a.N == 0:
        third = " Bei N = 0 gibt es keinen Regler (F₀ ist der Kasten)."
    elif not a.start_in_fN:
        third = f" Mit N = {a.N} ist er nicht lösbar: Der Regler startet gar nicht."
    elif a.run is None:
        third = ""
    elif a.run.status == "verloren":
        xe = a.run.traj[-1]
        third = (f" Mit N = {a.N} und den Kosten von Beispiel 12.1 (Q = I, R = 10) fährt der Regler {a.run.k_stop} Takt(e) und scheitert bei "
                 f"{pt(xe)}, dort ist F_{a.N} verlassen." + (" " + a.blame if a.blame else ""))
    elif a.run.status == "konvergiert":
        third = f" Mit N = {a.N} und diesen Kosten konvergiert der Regler in {a.run.k_stop} Takten."
    else:
        third = f" Mit N = {a.N} bleibt der Regler in den {len(a.run.traj) - 1} gerechneten Takten lösbar."
    kind = "success" if (a.run is not None and a.run.status != "verloren") else ("warning" if (a.run is not None) else "info")
    return kind, first + second + third


# --- Experiment 3: wer fällt? -----------------------------------------------------------------------------------------------------------------

def exit_area_for(umax, s1, s2, N: int) -> F:
    reg = R.regions(umax, s1, s2)
    return P.exit_area(reg.Cset, reg.F(N - 1), R.A_EX, R.B_EX, reg.umax)


def verdict_falls(counts: dict, N: int, cost_key: str, umax, s1, s2) -> tuple:
    reg = R.regions(umax, s1, s2)
    c = counts
    label = C.COSTS[cost_key]["label"]
    if c["nicht_in_c"] == 0:
        head = (f"{label}, N = {N}: Alle {c['starts']} Starts liegen in C (F_N ist hier gleich C). "
                f"Aus C heraus fallen {c['faelle_innen']} von {c['in_c']} Starts (Verlust der Lösbarkeit).")
    else:
        head = (f"{label}, N = {N}: {c['nicht_in_c']} von {c['starts']} Starts liegen außerhalb von C, {c['faelle_aussen']} davon fallen (Verlust der Lösbarkeit). "
                f"Aus C heraus fallen {c['faelle_innen']} von {c['in_c']} Starts.")
    if reg.n_bar is None:
        tail = " Die Gebietsfolge hat bis N = 60 kein Ende; eine Garantie gibt es nicht."
    elif N >= reg.n_bar + 1:
        tail = (f" Ab N = N̄ + 1 = {reg.n_bar + 1} ist ein Austritt aus C exakt unmöglich: Das Ziel des ersten Schritts, F_(N−1), ist schon C. Das gilt unabhängig von den Kosten.")
    elif exit_area_for(umax, s1, s2, N) == 0:
        # Die Fläche des Austrittsteils kann schon bei N = N̄ gleich 0 sein (bei einigen Grenzen); dann gibt es dort keinen Rest.
        tail = (f" Bei N = {N} ist ein Austritt aus C schon exakt unmöglich (Fläche des Teils außerhalb 0), obwohl F_(N−1) noch größer als C ist; das gilt unabhängig von den Kosten. "
                f"Gezählt wurden {c['austritt']} Austritte aus C.")
    elif N == reg.n_bar:
        ea = exit_area_for(umax, s1, s2, N)
        tail = (f" Bei N = N̄ = {reg.n_bar} gibt es einen Rest: Ein zulässiger erster Schritt kann C noch verlassen (Fläche des Teils außerhalb {fr(ea)} = {de(ea, 4)}); "
                f"gezählt wurden {c['austritt']} Austritte in diesen {c['in_c']} Starts. Eine Garantie gibt es erst ab N = {reg.n_bar + 1}.")
    else:
        ea = exit_area_for(umax, s1, s2, N)
        tail = (f" Bei N = {N} kann ein zulässiger erster Schritt aus C hinausführen (Fläche des Teils außerhalb {de(ea, 3)}); ob ihn der Regler wählt, entscheiden die Kosten: "
                f"gezählt wurden {c['austritt']} Austritte aus C.")
    rand = f" {c['randlauf']} Bahnen laufen an den Rand und werden nicht als Fall gezählt (das QP entscheidet dort bei 10⁻⁹ nicht mehr)." if c["randlauf"] else ""
    if c["faelle_innen"] > 0:
        kind = "warning"
    elif c["faelle_aussen"] > 0:
        kind = "info"
    else:
        kind = "success"
    return kind, head + tail + rand


def falls_rows(res: "L.FallResult") -> list:
    """Tabelle zu Experiment 3: je Gruppe (Start außerhalb C, in C, alle) Starts, Fälle, Austritte aus C, Konvergenz, Randlauf, kein Ereignis."""
    st = list(res.status)
    rows = []
    for name, mask in (("Start außerhalb von C", ~res.in_c), ("Start in C", res.in_c), ("alle Starts", np.ones(len(st), bool))):
        idx = [i for i in range(len(st)) if mask[i]]
        sub = [st[i] for i in idx]
        rows.append({"Gruppe": name, "Starts": len(idx), "fallen": sub.count("verloren"), "Austritt aus C": int(sum(bool(res.exits_c[i]) for i in idx)),
                     "konvergieren": sub.count("konvergiert"), "Randlauf": sub.count("randlauf"), "kein Ereignis": sub.count("ende")})
    return rows


# --- Experiment 4: Terminalmenge -------------------------------------------------------------------------------------------------------------

@dataclass
class TerminalAnalysis:
    term_key: str
    N: int
    umax: float
    s1: float
    s2: float
    reg: R.Regions
    T: R.Terminal
    K_area: F
    share: F
    shares: list              # Anteil von K_N an C für N = 0 … N_max
    n_full: int | None        # kleinstes N mit K_N = C
    start_key: str
    start_in_K: bool
    xf_share: F


def analyse_terminal(term_key: str, N: int, umax: float, s1: float, s2: float, start_key: str, N_max: int = 14) -> TerminalAnalysis:
    reg = R.regions(umax, s1, s2)
    T = R.terminal(term_key, umax, s1, s2)
    shares = [R.share_of_c(T.K(n), reg.Cset) for n in range(N_max + 1)]
    n_full = next((n for n, v in enumerate(shares) if v == 1), None)
    start = C.START_POINTS[start_key]["x"]
    return TerminalAnalysis(term_key, N, umax, s1, s2, reg, T, P.area_of(T.K(N)), shares[N] if N <= N_max else R.share_of_c(T.K(N), reg.Cset), shares, n_full, start_key,
                            P.contains(T.K(N), start), shares[0])


def verdict_terminal(a: TerminalAnalysis, counts: dict | None) -> tuple:
    name = C.TERM_LABELS[a.term_key]
    aC = P.area_of(a.reg.Cset)
    text = (f"{name}, N = {a.N}: K_N deckt {pct(a.share)} von C ({de(a.K_area)} von {de(aC)}); die Terminalmenge selbst (N = 0) {pct(a.xf_share)}. ")
    if a.n_full is not None:
        text += f"K_N ist gleich C ab N = {a.n_full}"
    else:
        text += "K_N erreicht C bis N = 14 nicht"
    if a.reg.n_bar is not None:
        text += f", das Gebiet ohne Terminalmenge schon bei N̄ = {a.reg.n_bar}: Die Terminalmenge macht die Lösbarkeit zur Garantie und kostet Horizont."
    else:
        text += "."
    nm = C.START_POINTS[a.start_key]["label"].split(":")[0]
    text += (f" Der Start {nm} liegt {'in' if a.start_in_K else 'nicht in'} K_{a.N}"
             + (": mit Terminalmenge wird er vor dem Losfahren abgewiesen." if not a.start_in_K else "."))
    if counts is not None and counts["starts"] > 0:
        text += (f" {counts['starts']} Starts in K_N: {counts['faelle']} Fälle, {counts['konvergiert']} konvergieren (|x| < 10⁻⁷ in {C.T_LOOP_TERM} Takten), "
                 f"die Wertfunktion nimmt in {counts['wert_verletzt']} Schritten nicht um die Stufenkosten ab.")
    ok = counts is not None and counts["faelle"] == 0
    return ("success" if ok else "info"), text


# --- Experiment 5: Stellgrenze und Kasten ------------------------------------------------------------------------------------------------------

def verdict_umax(table: list, umax: float, s1: float, s2: float = 5.0) -> tuple:
    row = next(r for r in table if r["umax"] == umax and r["s1"] == s1)
    same_s = [r for r in table if r["s1"] == s1]
    nbars = " / ".join(str(r["n_bar"]) if r["n_bar"] is not None else ">60" for r in same_s)
    umaxes = " / ".join(fr(P.frac(r["umax"])) for r in same_s)
    n = row["n_bar"]
    text = (f"|u| ≤ {fr(P.frac(umax))}, |x₁| ≤ {fr(P.frac(s1))}, |x₂| ≤ {fr(P.frac(s2))}: "
            + (f"N̄ = {n}, " if n is not None else "N̄ nicht erreicht, ")
            + f"die Fläche von C ist {de(row['area_C'])}, {pct(row['share_X'])} der Fläche des Kastens. Über u_max = {umaxes}: N̄ = {nbars} (bei |x₁| ≤ {fr(P.frac(s1))}).")
    if n is not None:
        text += f" Ab N = N̄ + 1 = {n + 1} bleibt ein Zustand aus C auch ohne Terminalmenge lösbar."
    if n == 1:
        text += " Bei N̄ = 1 ist C schon F₁: Ein Schritt Vorausschau reicht."
    return "info", text


# --- Kernfragen (kurze Antworten, live aus den Experimenten) --------------------------------------------------------------------------------------

def answer_solvable(a: RegionAnalysis) -> tuple:
    """Kernfrage 1: Welche Zustände sind lösbar, welche für immer haltbar?"""
    reg = a.reg
    aN, aC = reg.area(a.N), P.area_of(reg.Cset)
    text = (f"Mit Horizont N = {a.N} ist das Problem für die Zustände in F_{a.N} lösbar (Fläche {de(aN)}), für immer haltbar sind nur die in C (Fläche {de(aC)}, "
            f"{pct(aC / aN) if aN else '–'} von F_{a.N}). "
            + (f"Ab N̄ = {reg.n_bar} sind beide Mengen gleich." if reg.n_bar is not None else "Eine endliche Determiniertheit gibt es bis N = 60 nicht.")
            + " Ein Zustand in F_N und nicht in C ist heute lösbar und trotzdem verloren: Die Zukunft entscheidet.")
    return "info", text


def answer_blame(counts: dict, N: int, cost_key: str, umax, s1, s2) -> tuple:
    """Kernfrage 2: Fällt eine Bahn aus C heraus, und wer ist schuld, Horizont oder Kosten?"""
    reg = R.regions(umax, s1, s2)
    c = counts
    label = C.COSTS[cost_key]["label"]
    if c["faelle_innen"] > 0:
        schuld = f"Aus C heraus fallen {c['faelle_innen']} von {c['in_c']} Starts: Bei diesen Kosten treiben sie den Zustand aus C, der Horizont allein reicht nicht."
    elif c["nicht_in_c"] == 0:
        schuld = f"Alle {c['starts']} Starts liegen in C (F_N ist hier gleich C), und keiner fällt."
    else:
        schuld = (f"Aus C heraus fällt keiner der {c['in_c']} Starts; es fallen nur Starts, die schon außerhalb von C lagen ({c['faelle_aussen']} von {c['nicht_in_c']}): "
                  f"Dort ist der Zustand verloren, bevor der Regler losfährt.")
    if reg.n_bar is not None and N >= reg.n_bar + 1:
        garantie = f" Ab N = N̄ + 1 = {reg.n_bar + 1} ist ein Austritt aus C exakt unmöglich, unabhängig von den Kosten."
    elif reg.n_bar is not None and exit_area_for(umax, s1, s2, N) == 0:
        garantie = f" Bei diesen Grenzen ist ein Austritt aus C schon bei N = {N} exakt unmöglich (Fläche des Teils außerhalb 0), unabhängig von den Kosten."
    elif reg.n_bar is not None:
        garantie = f" Eine Garantie gibt es erst ab N = {reg.n_bar + 1}; darunter hängt es an den Kosten."
    else:
        garantie = ""
    return "info", f"{label}, N = {N}: {schuld}{garantie}"


def answer_terminal(a: TerminalAnalysis) -> tuple:
    """Kernfrage 3: Was kostet die Terminalmenge?"""
    name = C.TERM_LABELS[a.term_key]
    nf = f"ab N = {a.n_full}" if a.n_full is not None else "bis N = 14 nicht"
    nb = f"schon bei N̄ = {a.reg.n_bar}" if a.reg.n_bar is not None else "nie in endlich vielen Schritten"
    return "info", (f"Mit {name} ist K_N in C enthalten und invariant: Die Lösbarkeit wird zur Garantie. Der Preis ist Horizont: K_N deckt C {nf}, das Gebiet ohne Terminalmenge {nb}; "
                    f"bei N = {a.N} deckt K_N {pct(a.share)} von C. Die Terminalmenge verhindert nicht, dass ein Zustand außerhalb von K_N verloren ist, sie weist ihn am Start ab.")
