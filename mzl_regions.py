"""Gebiete des Wagens: F_N (lösbar ohne Terminalmenge), C (maximale steuerinvariante Menge), Terminalmengen und K_N(X_f), alles exakt mit Brüchen.

Begriffe (eigene Rechnung in diesem Modell, bei diesen Grenzen):
  F_N        Zustände, für die das Problem mit Horizont N lösbar ist (ohne Terminalmenge): F_0 = X, F_(N+1) = {x ∈ X : es gibt u mit A x + B u ∈ F_N}.
  C          maximale steuerinvariante Menge im Kasten X: F_N für große N. N̄ ist das kleinste N mit F_N = F_(N+1); dann ist C = F_N̄.
  K_N(X_f)   lösbar mit der Terminalbedingung x(N) ∈ X_f: K_0 = X_f, K_(N+1) = {x ∈ X : es gibt u mit A x + B u ∈ K_N}.
  O_LQR      maximale positiv invariante zulässige Menge des Regelkreises mit dem (auf drei Stellen gerundeten) LQR-Gewinn: Zustandsgrenzen und |K x| ≤ u_max.
Alle Mengen gehören zu den Grenzen (u_max, s₁, s₂): |u| ≤ u_max, |x₁| ≤ s₁, |x₂| ≤ s₂."""

from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache

import mzl_constants as C
import mzl_poly as P

A_EX = [list(r) for r in C.A_CAR]
B_EX = list(C.B_CAR)


@dataclass(frozen=True)
class Regions:
    """Die Mengenfolge F_N ohne Terminalmenge für feste Grenzen."""

    umax: F
    s1: F
    s2: F
    seq: tuple          # F_0, F_1, … bis zum Abbruch (letztes Element: F_N̄, falls n_bar nicht None)
    n_bar: int | None   # kleinstes N mit F_N = F_(N+1); None: bis N_BAR_MAX nicht erreicht

    def F(self, N: int):
        """F_N (für N über das Ende der Folge hinaus: das letzte Glied, also C, wenn n_bar bekannt ist)."""
        return self.seq[min(N, len(self.seq) - 1)]

    @property
    def Cset(self):
        return self.seq[-1]

    @property
    def X(self):
        return self.seq[0]

    def area(self, N: int) -> F:
        return P.area_of(self.F(N))


def _fr(x) -> F:
    return P.frac(x)


@lru_cache(maxsize=64)
def regions(umax: float, s1: float, s2: float) -> Regions:
    um, a, b = _fr(umax), _fr(s1), _fr(s2)
    X = P.box(a, b)
    seq, n_bar = P.reach_sequence(X, X, A_EX, B_EX, um, n_max=C.N_BAR_MAX)
    return Regions(um, a, b, tuple(tuple(c) for c in seq), n_bar)


def invariant_lqr(K, umax, X, n_max=200):
    """Maximale positiv invariante zulässige Menge von x⁺ = (A − B K) x unter X und |K x| ≤ u_max (exakt). Rückgabe (Menge, Schritte) oder (None, n_max)."""
    um = _fr(umax)
    AK = [[A_EX[0][0] - B_EX[0] * K[0], A_EX[0][1] - B_EX[0] * K[1]], [A_EX[1][0] - B_EX[1] * K[0], A_EX[1][1] - B_EX[1] * K[1]]]
    O = P.prune(list(X) + [(K[0], K[1], um), (-K[0], -K[1], um)])
    for it in range(1, n_max + 1):
        pre = [(a1 * AK[0][0] + a2 * AK[1][0], a1 * AK[0][1] + a2 * AK[1][1], b) for a1, a2, b in O]
        nxt = P.prune(O + pre)
        if P.same(nxt, O):
            return O, it
        O = nxt
    return None, n_max


@dataclass(frozen=True)
class Terminal:
    """Terminalmenge X_f und die Folge K_N(X_f) für feste Grenzen."""

    key: str
    Xf: tuple
    seq: tuple
    n_bar: int | None   # kleinstes N, ab dem K_N = K_(N+1) (dann ist K_N = C, falls X_f ⊆ C steuerinvariant ist und die Folge C erreicht)

    def K(self, N: int):
        return self.seq[min(N, len(self.seq) - 1)]


@lru_cache(maxsize=64)
def terminal(key: str, umax: float, s1: float, s2: float) -> Terminal:
    um, a, b = _fr(umax), _fr(s1), _fr(s2)
    X = P.box(a, b)
    if key == "null":
        Xf = P.box(0)
    else:
        Xf, _ = invariant_lqr(C.LQR_GAINS[C.TERM_R[key]], um, X)
    seq, n_bar = P.reach_sequence(X, Xf, A_EX, B_EX, um, n_max=C.N_BAR_MAX)
    return Terminal(key, tuple(Xf), tuple(tuple(c) for c in seq), n_bar)


def share_of_c(cons, Cset) -> F:
    """Anteil der Fläche von `cons` an der von C (exakt)."""
    aC = P.area_of(Cset)
    return P.area_of(cons) / aC if aC else F(0)


def exit_areas(reg: Regions, N_max: int | None = None) -> dict:
    """Fläche des Teils der zulässigen Folgezustände aus C (erste Stellgröße mit Folgezustand in F_(N−1)), der außerhalb von C liegt, je Horizont N = 1 … N_max.
    Null heißt: Aus C führt kein zulässiger erster Schritt hinaus (das gilt, sobald F_(N−1) = C, also ab N = N̄ + 1)."""
    N_max = (reg.n_bar or 0) + 2 if N_max is None else N_max
    return {N: P.exit_area(reg.Cset, reg.F(N - 1), A_EX, B_EX, reg.umax) for N in range(1, N_max + 1)}


def u_table(options=C.UMAX_OPTIONS, positions=C.S_OPTIONS, s2=5.0) -> list:
    """Tabelle zu Experiment 5: je (u_max, s₁) N̄, Fläche von C, Anteil an X (|x₂| ≤ s₂)."""
    rows = []
    for um in options:
        for s in positions:
            reg = regions(um, s, s2)
            aX = P.area_of(reg.X)
            aC = P.area_of(reg.Cset)
            rows.append({"umax": um, "s1": s, "n_bar": reg.n_bar, "area_C": aC, "share_X": aC / aX, "areas": [P.area_of(reg.F(N)) for N in (1, 2, 3)]})
    return rows
