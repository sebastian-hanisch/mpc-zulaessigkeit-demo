"""Handbeispiel: der Skalar x⁺ = a·x + u mit |u| ≤ u_max und |x| ≤ s (Standard a = 2, u_max = 1, s = 4).

Die Gebiete sind Intervalle [−f, f] und exakt mit Brüchen zu rechnen: F_(N+1) = {|x| ≤ s : es gibt u, |u| ≤ u_max, |a x + u| ≤ f_N}, also
f_(N+1) = min(s, (f_N + u_max)/a). Für a = 2, u_max = 1, s = 4 ist f_N = 1 + 3/2^N; C = [−1, 1], und f_N > 1 für jedes N (nie ein endliches N̄).
Die Regelung im Kreis löst das Problem mit dem QP-Löser aus `mzl_qp` (Zustandsdimension 1)."""

from fractions import Fraction as F

import numpy as np

import mzl_constants as C
import mzl_loop as L
import mzl_poly as P
import mzl_qp as Q


def f_seq(N_max: int, a=C.SK_A, umax=C.SK_UMAX, s=C.SK_S, start=None) -> list:
    """Halbbreiten f_0 … f_N_max der Gebiete F_N (start: Halbbreite des Ziels, Standard s = F_0); mit start = t die Gebiete K_N([−t, t])."""
    a, umax, s = P.frac(a), P.frac(umax), P.frac(s)
    f = [s if start is None else P.frac(start)]
    for _ in range(N_max):
        f.append(min(s, (f[-1] + umax) / a))
    return f


def c_halb(a=C.SK_A, umax=C.SK_UMAX, s=C.SK_S) -> F:
    """Halbbreite von C: der Fixpunkt min(s, u_max/(a − 1)) für a > 1."""
    a, umax, s = P.frac(a), P.frac(umax), P.frac(s)
    return min(s, umax / (a - 1)) if a > 1 else s


def next_interval(x0, N: int, terminal: bool = False, a=C.SK_A, umax=C.SK_UMAX, s=C.SK_S):
    """Die erlaubten nächsten Zustände x₁ = a·x₀ + u₀ bei Horizont N: Intervall (lo, hi) oder None (leer; Ziel F_(N−1) bzw. K_(N−1))."""
    a, umax = P.frac(a), P.frac(umax)
    x0 = P.frac(x0)
    f_prev = f_seq(N - 1, start=C.SK_TERM if terminal else None)[N - 1] if N >= 1 else None
    if f_prev is None:
        return None
    lo, hi = max(a * x0 - umax, -f_prev), min(a * x0 + umax, f_prev)
    return (lo, hi) if lo <= hi else None


def problem(N: int, cost_key: str, terminal: bool, a=C.SK_A, umax=C.SK_UMAX, s=C.SK_S) -> Q.Problem:
    c = C.SK_COSTS[cost_key]
    term = [(1.0, float(C.SK_TERM)), (-1.0, float(C.SK_TERM))] if terminal else None
    return Q.build_problem(N, [[c["q"]]], c["r"], [[c["p"]]], [float(s)], float(umax), term=term, lin=c["lin"], A=[[float(a)]], B=[[1.0]])


def region_float(N: int, terminal: bool, a=C.SK_A, umax=C.SK_UMAX, s=C.SK_S):
    """Das Gebiet des Problems als (a, b): |x| ≤ f_N (f_N aus F_N bzw. K_N)."""
    fN = float(f_seq(N, a, umax, s, start=C.SK_TERM if terminal else None)[N])
    return np.array([[1.0], [-1.0]]), np.array([fN, fN])


def run(x0: float, N: int, cost_key: str, terminal: bool, T: int = 12):
    """Regelkreis des Skalars ab x₀ (Zustand als Zahlenfeld mit einer Spalte); N = 0 gibt keinen Regler: F_0 = [−s, s] ist nur der Kasten."""
    if N < 1:
        return L.Run(np.array([[float(x0)]]), "kein_regler", None)
    return L.run_loop(problem(N, cost_key, terminal), region_float(N, terminal), np.array([float(x0)]), T, conv_tol=1e-6)


def classify(x0, N: int, terminal: bool = False) -> str:
    """Wo liegt x₀? 'aussen' (nicht in F_N bzw. K_N), 'in_FN_nicht_C' (lösbar, aber nicht für immer), 'in_C'. Exakt."""
    x0 = P.frac(x0)
    fN = f_seq(N, start=C.SK_TERM if terminal else None)[N]
    if abs(x0) > fN:
        return "aussen"
    return "in_C" if abs(x0) <= c_halb() else "in_FN_nicht_C"
