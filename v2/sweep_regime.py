"""Barrido de PRIOR_NEW_REGIME y CAR_WINDOW sobre las primeras 6 carreras de 2022 y 2026."""
import itertools
import sys
import pandas as pd

import config as C
from evaluate import _race_metrics
from features import prepare_results, build_training_table
from model import fit_bundle

SHOW = ["spearman", "spearman_grid", "mae", "mae_grid", "top3_overlap", "winner_logloss", "podium_brier"]


def run(raw):
    table = build_training_table(prepare_results(raw))
    out = {}
    for season in (2022, 2026):
        ts = sorted(table.loc[(table["season"] == season) & (table["round"] <= 6), "t"].unique())
        bundle = fit_bundle(table[table["t"] < ts[0]])
        out[season] = pd.DataFrame([_race_metrics(bundle, table[table["t"] == t]) for t in ts])[SHOW].mean()
    return out


def main():
    raw = pd.read_csv(C.RESULTS_CSV)
    rows = []
    for shrink, pr, win in itertools.product([0.5, 0.75, 0.9], [0.5, 1.5, 3.0], [3, 5, 8]):
        C.PRIOR_NEW_REGIME = {"shrink": shrink, "prior_races": pr}
        C.CAR_WINDOW = win
        r = run(raw)
        row = {"shrink": shrink, "prior_races": pr, "window": win}
        for s, m in r.items():
            row.update({f"{k}_{s}": round(v, 3) for k, v in m.items() if k in ("spearman", "mae", "winner_logloss", "podium_brier")})
            row[f"spearman_grid_{s}"] = round(m["spearman_grid"], 3)
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df["spearman_avg"] = (df["spearman_2022"] + df["spearman_2026"]) / 2
    df.sort_values("spearman_avg", ascending=False).to_csv(C.DATA_DIR / "sweep_regime.csv", index=False)
    print("\n", df.sort_values("spearman_avg", ascending=False).to_string())


if __name__ == "__main__":
    sys.exit(main())
