"""Erzeugt `data/mzl_frozen.json`: die vorgerechneten Messreihen der App und der README (Regelkreise mit eigenem QP-Löser, Startzustände aus SplitMix64).

Aufruf (Wurzelordner des Repos): `python tools/make_frozen.py` schreibt die Datei, `python tools/make_frozen.py --check` rechnet neu und vergleicht (Zählungen ± 2, Randlauf ± 25).
Rechenzeit: etwa eine halbe Minute. Die Reihen gelten für |u| ≤ 1/2, |xᵢ| ≤ 5, 500 Starts und den Zufalls-Seed 716; die Zählungen hängen an den Startzuständen
(gezogen mit `mzl_rng.SplitMix64`, auf jeder Plattform gleich) und an der QP-Lösung (Zählungen an Rändern können sich plattformabhängig um 1 bis 2 verschieben)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mzl_constants as C  # noqa: E402
import mzl_loop as L  # noqa: E402

OUT = ROOT / "data" / "mzl_frozen.json"
PARAMS = {"umax": 0.5, "s1": 5.0, "s2": 5.0, "n_starts": 500, "seed": 716, "terminal": "lqr1", "T_loop": C.T_LOOP, "T_term": C.T_LOOP_TERM}
N_RANGE = tuple(range(1, 9))
N_TERM = tuple(range(1, 11))
SEEDS = tuple(range(716, 726))             # zehn Zufalls-Seeds für die Stabilitätsprüfung der Austritte
SEED_CELLS = (("grenze", 5), ("grenze", 6), ("grenze", 7), ("q_i_r10", 2), ("q_i_r10", 3), ("pos", 2), ("q_i_r01", 2))


def falls_cells() -> dict:
    out = {}
    for cost in C.COST_KEYS:
        rows = []
        for N in N_RANGE:
            c = L.experiment_falls(N, cost, PARAMS["n_starts"], PARAMS["seed"], PARAMS["umax"], PARAMS["s1"], PARAMS["s2"]).counts()
            rows.append({"N": N, **c})
        out[cost] = rows
    return out


def terminal_cells() -> list:
    rows = []
    for N in N_TERM:
        c = L.experiment_terminal(PARAMS["terminal"], N, PARAMS["n_starts"], PARAMS["seed"], PARAMS["umax"], PARAMS["s1"], PARAMS["s2"]).counts()
        rows.append({"N": N, **c})
    return rows


def seed_cells() -> dict:
    out = {}
    for cost, N in SEED_CELLS:
        per_seed = []
        for seed in SEEDS:
            c = L.experiment_falls(N, cost, PARAMS["n_starts"], seed, PARAMS["umax"], PARAMS["s1"], PARAMS["s2"]).counts()
            per_seed.append({"seed": seed, "in_c": c["in_c"], "nicht_in_c": c["nicht_in_c"], "faelle_aussen": c["faelle_aussen"], "faelle_innen": c["faelle_innen"],
                             "austritt": c["austritt"], "randlauf": c["randlauf"]})
        out[f"{cost}:{N}"] = per_seed
    return out


def build() -> dict:
    return {"params": PARAMS, "costs": falls_cells(), "terminal": terminal_cells(), "seeds": seed_cells()}


def main(argv) -> int:
    data = build()
    if "--check" in argv:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        bad = []
        for cost, rows in data["costs"].items():
            for new, ref in zip(rows, old["costs"][cost]):
                for k, v in new.items():
                    if isinstance(v, int) and abs(v - ref[k]) > (25 if k == "randlauf" else 2):
                        bad.append((cost, new["N"], k, v, ref[k]))
        print("Abweichungen über 2:", bad if bad else "keine")
        return 1 if bad else 0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("geschrieben:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
