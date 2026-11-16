"""Exakte Polytope im Zustandsraum R² mit Brüchen (fractions): Gebiete der Zulässigkeit ohne Rundung.

Ein Polytop ist eine Liste von Ungleichungen `a1·x1 + a2·x2 ≤ b` (Tupel aus drei Brüchen). Alle Operationen sind exakt:
Vorbild (Urbild unter x⁺ = A x + B u, |u| ≤ u_max) per Fourier-Motzkin-Elimination der einen Stellgröße, Ecken per Aufzählung
der Schnittpunkte je zweier Geraden, überflüssige Ungleichungen über die Ecken. Das Modul ist eine Neuschreibung nach dem Muster
der Vormessung (`rg_v16_poly.py` im Planungsordner der Linie, nur gelesen, nicht importiert); nur Standardbibliothek.

Entartete Mengen (Punkt, Strecke) bleiben erhalten: `prune` entfernt dort nichts, sonst gingen die Gleichungen verloren. Die Vormessung fand
genau diesen Fehler (das Vereinfachen warf die Gleichungen weg, K_N({0}) hatte Fläche 0 bis N = 13); `tests/test_poly.py` hält den
Regressionstest „K_2({0}) hat Fläche 1“.
"""

import math
from fractions import Fraction as F
from itertools import combinations

Cons = list  # Liste von (a1, a2, b)


def frac(x) -> F:
    """Bruch aus Zahl oder Text; Gleitkommazahlen werden über ihre Dezimaldarstellung gelesen (0,1 wird 1/10, nicht ein Binärrest)."""
    if isinstance(x, F):
        return x
    if isinstance(x, float):
        return F(repr(x))
    return F(x)


def box(l1, l2=None) -> Cons:
    """Rechteck |x1| ≤ l1, |x2| ≤ l2 (l2 gleich l1, wenn nicht angegeben)."""
    l1 = frac(l1)
    l2 = l1 if l2 is None else frac(l2)
    return [(F(1), F(0), l1), (F(-1), F(0), l1), (F(0), F(1), l2), (F(0), F(-1), l2)]


def _norm(c):
    a1, a2, b = c
    s = abs(a1) + abs(a2)
    return (a1 / s, a2 / s, b / s)


def tidy(cons) -> Cons:
    """Normieren (|a1| + |a2| = 1) und je Richtung nur die engste Ungleichung behalten."""
    best: dict = {}
    for c in cons:
        if c[0] == 0 and c[1] == 0:             # 0 ≤ b: leere Bedingung (b ≥ 0) fällt weg, b < 0 macht die Menge leer und bleibt
            if c[2] < 0:
                best[(F(0), F(0))] = F(-1)
            continue
        a1, a2, b = _norm(c)
        key = (a1, a2)
        if key not in best or b < best[key]:
            best[key] = b
    return [(a1, a2, b) for (a1, a2), b in best.items()]


def vertices(cons) -> list:
    """Alle Ecken (exakt): Schnittpunkte je zweier Geraden, die alle Ungleichungen erfüllen; gegen den Uhrzeigersinn sortiert."""
    cons = tidy(cons)
    pts = set()
    for (a1, a2, b), (c1, c2, d) in combinations(cons, 2):
        det = a1 * c2 - a2 * c1
        if det == 0:
            continue
        x1 = (b * c2 - a2 * d) / det
        x2 = (a1 * d - b * c1) / det
        if all(p * x1 + q * x2 <= r for p, q, r in cons):
            pts.add((x1, x2))
    return ordered(pts)


def ordered(pts) -> list:
    """Punkte gegen den Uhrzeigersinn um den Schwerpunkt sortiert (die Sortierung nutzt Gleitkomma nur für den Winkel)."""
    pts = list(pts)
    if len(pts) < 3:
        return sorted(pts)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: (math.atan2(float(p[1] - cy), float(p[0] - cx)), p))


def area(pts) -> F:
    """Fläche des Vielecks aus den Ecken (Gaußsche Trapezformel, exakt); weniger als drei Ecken geben 0."""
    pts = ordered(pts)
    if len(pts) < 3:
        return F(0)
    s = F(0)
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def area_of(cons) -> F:
    return area(vertices(cons))


def prune(cons) -> Cons:
    """Nur die Ungleichungen behalten, die eine Kante mit mindestens zwei verschiedenen Ecken haben. Entartete Mengen bleiben unverändert."""
    cons = tidy(cons)
    vs = vertices(cons)
    if len(vs) < 3 or area(vs) == 0:
        return cons
    return [(a1, a2, b) for a1, a2, b in cons if sum(1 for v in vs if a1 * v[0] + a2 * v[1] == b) >= 2]


def contains(cons, pt) -> bool:
    return all(a1 * pt[0] + a2 * pt[1] <= b for a1, a2, b in cons)


def subset(a, b) -> bool:
    """A ⊆ B: alle Ecken von A erfüllen B (exakt)."""
    return all(contains(b, v) for v in vertices(a))


def same(a, b) -> bool:
    return subset(a, b) and subset(b, a)


def intersect(a, b) -> Cons:
    return prune(list(a) + list(b))


def scaled(cons, c) -> Cons:
    """c·P mit c > 0: jede Ungleichung a·x ≤ b wird a·x ≤ c·b."""
    c = frac(c)
    return tidy([(a1, a2, c * b) for a1, a2, b in cons])


def preimage(cons, A, B, umax) -> Cons:
    """{x : es gibt u, |u| ≤ umax, A x + B u ∈ P}; A ist 2×2, B ein Spaltenvektor (2), alles Brüche. Fourier-Motzkin-Elimination von u."""
    umax = frac(umax)
    rows = []                                   # c1 x1 + c2 x2 + d u ≤ e
    for a1, a2, b in cons:
        rows.append((a1 * A[0][0] + a2 * A[1][0], a1 * A[0][1] + a2 * A[1][1], a1 * B[0] + a2 * B[1], b))
    rows.append((F(0), F(0), F(1), umax))
    rows.append((F(0), F(0), F(-1), umax))
    up, lo, zero = [], [], []
    for c1, c2, d, e in rows:
        if d > 0:
            up.append((c1 / d, c2 / d, e / d))   # u ≤ e/d − (c1/d) x1 − (c2/d) x2
        elif d < 0:
            lo.append((c1 / d, c2 / d, e / d))   # u ≥ e/d − (c1/d) x1 − (c2/d) x2 (d < 0 dreht das Ungleichheitszeichen)
        else:
            zero.append((c1, c2, e))
    out = list(zero)
    for p1, p2, p3 in up:
        for q1, q2, q3 in lo:
            out.append((p1 - q1, p2 - q2, p3 - q3))   # q3 − q1 x1 − q2 x2 ≤ p3 − p1 x1 − p2 x2
    res = []
    for a1, a2, b in out:
        if a1 == 0 and a2 == 0:
            if b < 0:
                return [(F(0), F(0), F(-1))]      # leere Menge: 0 ≤ −1
            continue
        res.append((a1, a2, b))
    return res


def is_empty(cons) -> bool:
    return any(a1 == 0 and a2 == 0 and b < 0 for a1, a2, b in cons) or not vertices(cons)


def step_back(target, X, A, B, umax) -> Cons:
    """Ein Schritt rückwärts: {x ∈ X : es gibt u mit A x + B u ∈ Ziel}."""
    return prune(preimage(target, A, B, umax) + list(X))


def reach_sequence(X, target, A, B, umax, n_max=60) -> tuple:
    """Folge K_0 = Ziel, K_(N+1) = step_back(K_N). Abbruch, sobald K_(N+1) = K_N (dann ist n_bar = N) oder bei n_max.
    Rückgabe (Liste der Mengen K_0 … K_last, n_bar oder None). Mit Ziel = X ist die Folge F_N ohne Terminalmenge."""
    seq = [prune(list(target))]
    for _ in range(n_max):
        nxt = step_back(seq[-1], X, A, B, umax)
        if same(nxt, seq[-1]):
            return seq, len(seq) - 1
        seq.append(nxt)
    return seq, None


def hull(points) -> list:
    """Konvexe Hülle (Monotone Chain, exakt) als Ecken gegen den Uhrzeigersinn."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def hull_constraints(points) -> Cons:
    h = hull(points)
    cons = []
    for i in range(len(h)):
        p, q = h[i], h[(i + 1) % len(h)]
        a1, a2 = q[1] - p[1], -(q[0] - p[0])
        cons.append((a1, a2, a1 * p[0] + a2 * p[1]))
    return tidy(cons)


def image_with_input(cons, A, B, umax) -> Cons:
    """{A x + B u : x ∈ P, |u| ≤ umax} (konvexe Hülle der Bilder aller Ecken und der beiden Stellgrößen-Enden, exakt)."""
    umax = frac(umax)
    pts = []
    for v in vertices(cons):
        for u in (-umax, umax):
            pts.append((A[0][0] * v[0] + A[0][1] * v[1] + B[0] * u, A[1][0] * v[0] + A[1][1] * v[1] + B[1] * u))
    return hull_constraints(pts)


def exit_area(C, target, A, B, umax) -> F:
    """Fläche des Teils von {A x + B u : x ∈ C, |u| ≤ umax} ∩ Ziel, der außerhalb von C liegt (0: ein zulässiger Schritt aus C führt nie hinaus).
    Das ist die Frage „Kann die erste Stellgröße einen Zustand aus C verlassen lassen, ohne das Problem sofort zu verletzen?“ mit Ziel = F_(N−1)."""
    s = intersect(image_with_input(C, A, B, umax), target)
    return area_of(s) - area_of(intersect(s, C))


def to_float(cons) -> tuple:
    """(a, b) als Gleitkomma-Felder für schnelle Mitgliedschaftsprüfungen (numpy)."""
    import numpy as np

    return np.array([[float(c[0]), float(c[1])] for c in cons]), np.array([float(c[2]) for c in cons])


def float_vertices(cons) -> list:
    return [(float(x), float(y)) for x, y in vertices(cons)]


def fmt_frac(x) -> str:
    """Bruch als Text: 'a/b' oder ganze Zahl (für Anzeige und Tests)."""
    x = frac(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
