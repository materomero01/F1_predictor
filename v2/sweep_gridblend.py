"""Hipotesis: el ranker subpondera la grilla. Suma -beta*log(grid_adj) al puntaje (tau se refitea por fold)."""
import numpy as np
import pandas as pd
import config as C
import evaluate as E
import model as M
from features import build_training_table, prepare_results

T = build_training_table(prepare_results(pd.read_csv(C.RESULTS_CSV)))
ts = sorted(T.loc[T.season >= 2024, "t"].unique())
SHOW = ["spearman", "winner_hit_prob", "winner_hit_grid", "winner_logloss", "podium_brier", "top3_overlap"]
orig = M._fit_ranker


class Blend:
    def __init__(self, r, beta):
        self.r, self.beta = r, beta

    def predict(self, X):
        return self.r.predict(X) - self.beta * np.log(X["grid_adj"].to_numpy())


for beta in (0.8, 1.2, 1.6, 2.4, 4.0):
    M._fit_ranker = lambda tr, beta=beta: Blend(orig(tr), beta)
    rows = []
    for i in range(0, len(ts), 6):
        ch = ts[i:i + 6]
        b = M.fit_bundle(T[T.t < ch[0]])
        rows += [E._race_metrics(b, T[T.t == t]) for t in ch]
    print(f"beta={beta}", pd.DataFrame(rows)[SHOW].mean().round(3).to_dict(), flush=True)
