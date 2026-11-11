"""Barrido de TAU_TOP_K: calibracion de ganador/podio en el backtest walk-forward."""
import pandas as pd
import config as C
from evaluate import _race_metrics
from features import build_training_table, prepare_results
from model import fit_bundle

T = build_training_table(prepare_results(pd.read_csv(C.RESULTS_CSV)))
ts = sorted(T.loc[T.season >= 2024, "t"].unique())
SHOW = ["spearman", "winner_hit_prob", "winner_hit_grid", "winner_logloss", "podium_brier", "podium_brier_base", "top3_overlap"]
for k in (None, 5, 3, 1):
    C.TAU_TOP_K = k
    rows, taus = [], []
    for i in range(0, len(ts), 6):
        ch = ts[i:i + 6]
        b = fit_bundle(T[T.t < ch[0]])
        taus.append(b["tau"])
        rows += [_race_metrics(b, T[T.t == t]) for t in ch]
    m = pd.DataFrame(rows)[SHOW].mean().round(3).to_dict()
    print(f"K={k} tau_medio={sum(taus)/len(taus):.2f}", m, flush=True)
