"""Backtest PRE-clasificacion: la grilla se estima con estimate_grid (como en predict.py sin --grid) en vez de usar la real.

Cada carrera usa un modelo entrenado sin ella y un historial recortado a las carreras anteriores.
    python backtest_pre_quali.py [--last 30]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
from features import build_training_table, prepare_results
from model import fit_bundle
from predict import estimate_grid, run

ap = argparse.ArgumentParser()
ap.add_argument("--last", type=int, default=30)
a = ap.parse_args()

raw = pd.read_csv(C.RESULTS_CSV)
T = build_training_table(prepare_results(raw))
ts = sorted(T["t"].unique())[-a.last:]
rows = []
for i in range(0, len(ts), 6):
    chunk = ts[i:i + 6]
    bundle = fit_bundle(T[T["t"] < chunk[0]])
    for t in chunk:
        g = T[T["t"] == t]
        s, r, circ = int(g["season"].iloc[0]), int(g["round"].iloc[0]), None
        before = raw[(raw["season"] < s) | ((raw["season"] == s) & (raw["round"] < r))]
        race = raw[(raw["season"] == s) & (raw["round"] == r)]
        circ = race["circuit_id"].iloc[0]
        lineup = race[["driver_id", "code", "driver_name", "constructor_id", "constructor_name"]].reset_index(drop=True)
        est = estimate_grid(before, s, r, circ, lineup)
        est["is_sprint"] = int(race["is_sprint_weekend"].iloc[0])
        pred = run(before, bundle, s, r, circ, est).merge(g[["driver_id", "pos", "grid_adj"]], on="driver_id")
        pred = pred.merge(est[["driver_id", "grid"]].rename(columns={"grid": "grid_est"}), on="driver_id")
        top = pred.sort_values("prob_ganar", ascending=False).iloc[0]
        p_win = float(np.clip(pred.loc[pred["pos"] == 1, "prob_ganar"].sum(), 1e-3, 1))
        podium = (pred["pos"] <= 3).astype(float)
        rows.append({
            "season": s, "round": r,
            "spearman": spearmanr(pred["pos_esperada"], pred["pos"])[0],
            "spearman_gridest": spearmanr(pred["grid_est"], pred["pos"])[0],
            "spearman_gridreal": spearmanr(pred["grid_adj"], pred["pos"])[0],
            "winner_hit": float(top["pos"] == 1),
            "winner_hit_gridest": float(pred.sort_values("grid_est").iloc[0]["pos"] == 1),
            "winner_logloss": -np.log(p_win),
            "podium_brier": float(((pred["prob_podio"] - podium) ** 2).mean()),
            "podium_brier_base": float(((podium.mean() - podium) ** 2).mean()),
            "top3_overlap": len(set(pred.nsmallest(3, "pos_esperada")["driver_id"]) & set(pred.nsmallest(3, "pos")["driver_id"])),
            "top3_overlap_gridest": len(set(pred.nsmallest(3, "grid_est")["driver_id"]) & set(pred.nsmallest(3, "pos")["driver_id"])),
        })
    print(f"  {len(rows)}/{len(ts)}", flush=True)
res = pd.DataFrame(rows)
res.to_csv(C.DATA_DIR / "backtest_pre_quali.csv", index=False)
print(f"\n=== PRE-CLASIFICACION ({len(res)} carreras, grilla estimada) ===")
print(res.drop(columns=["season", "round"]).mean().round(3).to_string())
