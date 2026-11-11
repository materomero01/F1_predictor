"""Diagnóstico: ¿por qué el modelo acierta menos ganadores que la pole? (backtest walk-forward desde 2024)"""
import pandas as pd

import config as C
from features import build_training_table, prepare_results
from model import fit_bundle, predict_race

raw = pd.read_csv(C.RESULTS_CSV)
T = build_training_table(prepare_results(raw))
ts = sorted(T.loc[T.season >= 2024, "t"].unique())
rows = []
for i in range(0, len(ts), 6):
    ch = ts[i:i + 6]
    b = fit_bundle(T[T.t < ch[0]])
    for t in ch:
        g = T[T.t == t]
        p = predict_race(b, g, n_sims=4000, seed=1).merge(g[["driver_id", "pos", "grid_adj", "dnf"]], on="driver_id")
        m = p.sort_values("prob_ganar", ascending=False).iloc[0]
        pole = p.sort_values("grid_adj").iloc[0]
        w = p[p.pos == 1].iloc[0]
        rows.append(dict(season=g.season.iloc[0], rnd=g["round"].iloc[0], model=m.driver_id, p_model=m.prob_ganar,
                         pole=pole.driver_id, p_pole=pole.prob_ganar, pole_won=pole.pos == 1, model_won=m.pos == 1,
                         same=m.driver_id == pole.driver_id, winner_grid=w.grid_adj, p_winner=w.prob_ganar))
    print("fold", i, flush=True)
d = pd.DataFrame(rows)
d.to_csv(C.DATA_DIR / "diag_win.csv", index=False)
print("modelo==pole:", d.same.mean().round(3))
print(d.groupby("same")[["pole_won", "model_won"]].mean().round(3), d.groupby("same").size())
print("prob media a la pole:", d.p_pole.mean().round(3), "| pole gana real:", d.pole_won.mean().round(3))
print("prob media top modelo:", d.p_model.mean().round(3), "| acierta:", d.model_won.mean().round(3))
print("ganador salio de grilla<=2:", (d.winner_grid <= 2).mean().round(3))
print(d[~d.same][["season", "rnd", "model", "pole", "p_model", "p_pole", "pole_won", "model_won"]].to_string())
