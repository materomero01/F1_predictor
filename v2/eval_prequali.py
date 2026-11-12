"""Compara 3 formas de predecir ANTES de clasificar (walk-forward, mismas carreras):
  heuristica  : grilla estimada con estimate_grid (car_q + delta de clasificacion), tratada como cierta
  quali_modelo: grilla = argmax de un ranker de clasificacion entrenado, tratada como cierta
  muestreo    : grillas muestreadas del ranker, carrera promediada sobre ellas (incertidumbre propagada)
    python eval_prequali.py [--last 30]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
from features import build_training_table, prepare_results
from model import fit_bundle, predict_race
from prequali import QUALI_COLS, fit_quali, predict_prequali, regrid

ap = argparse.ArgumentParser()
ap.add_argument("--last", type=int, default=30)
a = ap.parse_args()

T = build_training_table(prepare_results(pd.read_csv(C.RESULTS_CSV)))
ts = sorted(T["t"].unique())[-a.last:]


def metrics(pred, g):
    p = pred.merge(g[["driver_id", "pos"]], on="driver_id")
    top = p.sort_values("prob_ganar", ascending=False).iloc[0]
    p_win = float(np.clip(p.loc[p["pos"] == 1, "prob_ganar"].sum(), 1e-3, 1))
    podium = (p["pos"] <= 3).astype(float)
    return {"spearman": spearmanr(p["pos_esperada"], p["pos"])[0], "winner_hit": float(top["pos"] == 1),
            "winner_logloss": -np.log(p_win), "podium_brier": float(((p["prob_podio"] - podium) ** 2).mean()),
            "top3_overlap": len(set(p.nsmallest(3, "pos_esperada")["driver_id"]) & set(p.nsmallest(3, "pos")["driver_id"]))}


rows, qrows = [], []
for i in range(0, len(ts), 6):
    chunk = ts[i:i + 6]
    train = T[T["t"] < chunk[0]]
    bundle, qb = fit_bundle(train), fit_quali(train)
    for t in chunk:
        g = T[T["t"] == t].reset_index(drop=True)
        n = len(g)
        h_rank = (g["car_q"] * (n / 2) + g["drv_quali_delta"]).rank(method="first").to_numpy()
        q_score = qb["ranker"].predict(g[QUALI_COLS])
        q_rank = pd.Series(-q_score).rank(method="first").to_numpy()
        qrows.append({"heuristica": spearmanr(h_rank, g["quali_adj"])[0], "ranker": spearmanr(q_rank, g["quali_adj"])[0]})
        rows.append({"modo": "heuristica", **metrics(predict_race(bundle, regrid(g, h_rank), seed=1), g)})
        rows.append({"modo": "quali_modelo", **metrics(predict_race(bundle, regrid(g, q_rank), seed=1), g)})
        rows.append({"modo": "muestreo", **metrics(predict_prequali(bundle, qb, g, seed=1), g)})
    print(f"  {len(qrows)}/{len(ts)}", flush=True)

print(f"\n=== Orden de clasificacion (Spearman vs. real, {len(qrows)} carreras) ===")
print(pd.DataFrame(qrows).mean().round(3).to_string())
print("\n=== Carrera, pre-clasificacion ===")
print(pd.DataFrame(rows).groupby("modo").mean().round(3).T.to_string())
