"""Der MPC-Regelkreis: je Periode das QP lösen, die erste Stellgröße anwenden, wieder von vorn (Receding Horizon).

Eine Bahn endet auf eine von vier Arten:
  "konvergiert"  der Betrag des Zustands fällt unter die Schwelle;
  "verloren"     der neue Zustand liegt außerhalb des Gebiets, in dem das Problem lösbar ist (exakte Menge, Toleranz 1e-7): ab hier gibt es keine Stellfolge,
                 die die Nebenbedingungen des nächsten Problems erfüllt. Das ist der Fall, den die Demo zählt;
  "randlauf"     das QP meldet „unlösbar“, obwohl der Zustand höchstens 1e-7 außerhalb der exakten Menge liegt (die Bahn kriecht an den Rand, die Zahl 1e-9
                 entscheidet nicht mehr): wird gezählt, aber nicht als Fall;
  "ende"         die Perioden sind aufgebraucht, ohne dass eines davon eintrat.
Zufall nur aus `mzl_rng.SplitMix64` (Startzustände gleichmäßig im Gebiet, Ablehnungsverfahren)."""

from dataclasses import dataclass

import numpy as np

import mzl_constants as C
import mzl_qp as Q
import mzl_regions as R
import mzl_poly as P
from mzl_rng import SplitMix64


@dataclass
class Run:
    traj: np.ndarray          # (Perioden + 1, Zustandsdimension)
    status: str               # konvergiert | verloren | randlauf | ende | kein_regler
    k_stop: int | None        # Periode, in der die Bahn endete (Verlust, Randlauf, Konvergenz) oder None
    value_violations: int = 0  # nur mit track_value: Schritte ohne Abnahme der Wertfunktion um die Stufenkosten


def inside(region, x, tol=0.0) -> bool:
    a, b = region
    return bool(np.all(a @ x <= b + tol))


def inside_many(region, X, tol=0.0):
    a, b = region
    return np.all(X @ a.T <= b[None, :] + tol, axis=1)


def run_loop(problem: Q.Problem, region, x0, T: int, conv_tol: float = C.CONV_TOL, track_value: bool = False) -> Run:
    """Regelkreis ab x0 über höchstens T Perioden. `region`: das Gebiet des Problems als (a, b) mit a x ≤ b (F_N oder K_N). Mit `track_value` zählt
    `Run.value_violations` (Attribut) die Schritte, in denen die Wertfunktion nicht um mindestens die Stufenkosten fällt (Toleranz 1e-8)."""
    A, B = problem.Phi[1], problem.Gam[1][:, 0]
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    vprev = None
    violations = 0
    status, k_stop = "ende", None
    for k in range(T):
        if not inside(region, x, C.REGION_TOL):
            status, k_stop = "verloren", k
            break
        U, st = problem.solve(x)
        if st != "ok":
            status, k_stop = "randlauf", k
            break
        if track_value:
            v = problem.cost(x, U)
            if vprev is not None and v - (vprev[0] - vprev[1]) > 1e-8:
                violations += 1
            u0 = float(U[0])
            vprev = (v, float(x @ problem.Qm @ x) + problem.R * u0 * u0)
        x = A @ x + B * U[0]
        traj.append(x.copy())
        if np.abs(x).max() < conv_tol:
            status, k_stop = "konvergiert", k + 1
            break
    return Run(np.array(traj), status, k_stop, violations)


def sample_starts(region, bounds, n: int, seed: int, max_draws: int = 400000) -> np.ndarray:
    """Bis zu n Startzustände gleichmäßig im Gebiet `region` (a x ≤ b) innerhalb des Kastens |x_i| ≤ bounds[i] (Ablehnungsverfahren mit SplitMix64).
    Kleiner als n nur, wenn das Gebiet so klein ist, dass max_draws Kandidaten nicht reichen (entartete Menge)."""
    rng = SplitMix64(int(seed))
    nx = len(bounds)
    a, b = region
    out = []
    draws = 0
    while len(out) < n and draws < max_draws:
        x = np.array([(2.0 * rng.random() - 1.0) * bounds[i] for i in range(nx)])
        draws += 1
        if np.all(a @ x <= b):
            out.append(x)
    return np.array(out).reshape(-1, nx)


def cost_problem(N: int, cost_key: str, umax: float, s1: float, s2: float, term=None, pf=None, r=None) -> Q.Problem:
    """Problem des Wagens mit einer Kostenwahl aus `C.COSTS`; mit Terminalmenge (`term`) und Endkosten `pf` und Stellgewicht `r` aus dem Aufrufer."""
    c = C.COSTS[cost_key]
    Qm = np.array(c["q"], dtype=float)
    return Q.build_problem(N, Qm, c["r"] if r is None else r, Qm if pf is None else pf, [s1, s2], umax, term=term, lin=c["lin"])


# --- Experiment „Wer fällt?“ (ohne Terminalmenge) -----------------------------------------------------------------------------------------------

@dataclass
class FallResult:
    N: int
    cost_key: str
    starts: np.ndarray        # (n, 2)
    in_c: np.ndarray          # bool je Start: liegt in C
    status: list              # Status je Start
    exits_c: np.ndarray       # bool je Start: Bahn verlässt C (nur für Starts in C sinnvoll)
    traj_len: np.ndarray

    def counts(self) -> dict:
        st = np.array(self.status)
        fall = st == "verloren"
        n = len(st)
        return {
            "starts": n,
            "in_c": int(self.in_c.sum()),
            "nicht_in_c": int((~self.in_c).sum()),
            "faelle": int(fall.sum()),
            "faelle_aussen": int((fall & ~self.in_c).sum()),
            "faelle_innen": int((fall & self.in_c).sum()),
            "austritt": int((self.exits_c & self.in_c).sum()),
            "konvergiert": int((st == "konvergiert").sum()),
            "randlauf": int((st == "randlauf").sum()),
            "ende": int((st == "ende").sum()),
        }


def experiment_falls(N: int, cost_key: str, n_starts: int, seed: int, umax: float, s1: float, s2: float, T: int = C.T_LOOP) -> FallResult:
    """Starts gleichmäßig in F_N, Regelkreis ohne Terminalmenge. Zählt Fälle (Verlust der Lösbarkeit), davon außerhalb und innerhalb von C, und die Austritte aus C."""
    reg = R.regions(umax, s1, s2)
    zul = P.to_float(reg.F(N))
    cset = P.to_float(reg.Cset)
    starts = sample_starts(zul, (s1, s2), n_starts, seed)
    prob = cost_problem(N, cost_key, umax, s1, s2)
    in_c = inside_many(cset, starts)
    ca, cb = cset
    status, exits, lens = [], np.zeros(len(starts), bool), np.zeros(len(starts), int)
    for i, x0 in enumerate(starts):
        run = run_loop(prob, zul, x0, T)
        status.append(run.status)
        lens[i] = len(run.traj) - 1
        exits[i] = bool(np.max(run.traj @ ca.T - cb[None, :]) > C.EXIT_TOL)
    return FallResult(N, cost_key, starts, in_c, status, exits, lens)


# --- Experiment Terminalmenge ----------------------------------------------------------------------------------------------------------------

def lyapunov_pf(K, r: float) -> np.ndarray:
    """Endkosten P_f: Lösung von A_Kᵀ P A_K − P = −(I + r KᵀK) für den (gerundeten) LQR-Gewinn K (u = −K x)."""
    import scipy.linalg as la

    Kn = np.array([[float(K[0]), float(K[1])]])
    AK = Q.A_CAR - Q.B_CAR @ Kn
    return la.solve_discrete_lyapunov(AK.T, np.eye(2) + r * (Kn.T @ Kn))


@dataclass
class TermResult:
    N: int
    term_key: str
    starts: np.ndarray
    status: list
    value_violations: int

    def counts(self) -> dict:
        st = np.array(self.status)
        return {
            "starts": len(st),
            "faelle": int((st == "verloren").sum()),
            "konvergiert": int((st == "konvergiert").sum()),
            "randlauf": int((st == "randlauf").sum()),
            "ende": int((st == "ende").sum()),
            "wert_verletzt": self.value_violations,
        }


def term_setup(term_key: str, N: int, umax: float, s1: float, s2: float):
    """(Problem, Gebiet K_N als Gleitkomma, K_N exakt) für die gewählte Terminalmenge; Stufenkosten Q = I, R aus dem Gewinn (bei {0}: R = 1)."""
    T = R.terminal(term_key, umax, s1, s2)
    Kset = T.K(N)
    Xf_cons = [(float(a1), float(a2), float(b)) for a1, a2, b in T.Xf]
    if term_key == "null":
        r, pf = 1.0, np.eye(2)
    else:
        r = C.LQR_R[C.TERM_R[term_key]]
        pf = lyapunov_pf(C.LQR_GAINS[C.TERM_R[term_key]], r)
    prob = Q.build_problem(N, np.eye(2), r, pf, [s1, s2], umax, term=Xf_cons)
    return prob, P.to_float(Kset), Kset


def experiment_terminal(term_key: str, N: int, n_starts: int, seed: int, umax: float, s1: float, s2: float, T: int = C.T_LOOP_TERM) -> TermResult:
    """Starts gleichmäßig in K_N(X_f), Regelkreis mit Terminalmenge: Fälle, Konvergenz, Abnahme der Wertfunktion um mindestens die Stufenkosten."""
    prob, kfloat, _ = term_setup(term_key, N, umax, s1, s2)
    starts = sample_starts(kfloat, (s1, s2), n_starts, seed)
    status, viol = [], 0
    for x0 in starts:
        run = run_loop(prob, kfloat, x0, T, conv_tol=C.CONV_TOL_TERM, track_value=True)
        status.append(run.status)
        viol += run.value_violations
    return TermResult(N, term_key, starts, status, viol)
