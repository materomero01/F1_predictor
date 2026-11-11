"""Backtest walk-forward: para cada carrera evaluada, el modelo solo vio carreras anteriores.

Se reentrena cada `--retrain-every` carreras. Se compara contra el baseline
"terminar donde largás" y se separa el rendimiento en:
  - regime_early: primeras 6 carreras de una temporada con cambio de reglamento (2022, 2026)
  - regular: el resto

Uso:
    python evaluate.py                       # evalúa desde la temporada 2024
    python evaluate.py --from-season 2022    # incluye el arranque del reglamento 2022
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
from features import build_training_table, prepare_results
from model import fit_bundle, predict_race


def _race_metrics(bundle, g: pd.DataFrame) -> dict:
    pred = predict_race(bundle, g, n_sims=4000, seed=1).merge(
        g[["driver_id", "pos", "dnf", "grid_adj"]], on="driver_id")
    pred["rank_pred"] = pred["pos_esperada"].rank(method="first")   # orden entero, comparable con la grilla
    true_win = pred["pos"] == 1
    podium = (pred["pos"] <= 3).astype(float)
    top3_pred = set(pred.nsmallest(3, "pos_esperada")["driver_id"])
    top3_true = set(pred.nsmallest(3, "pos")["driver_id"])
    top3_grid = set(pred.nsmallest(3, "grid_adj")["driver_id"])
    p_win = float(np.clip(pred.loc[true_win, "prob_ganar"].sum(), 1e-3, 1))
    return {
        "spearman": spearmanr(pred["pos_esperada"], pred["pos"])[0],
        "spearman_grid": spearmanr(pred["grid_adj"], pred["pos"])[0],
        "mae": (pred["rank_pred"] - pred["pos"]).abs().mean(),
        "mae_grid": (pred["grid_adj"] - pred["pos"]).abs().mean(),
        "winner_hit": float(pred.sort_values("pos_esperada").iloc[0]["pos"] == 1),
        "winner_hit_prob": float(pred.sort_values("prob_ganar", ascending=False).iloc[0]["pos"] == 1),
        "winner_hit_grid": float(pred.sort_values("grid_adj").iloc[0]["pos"] == 1),
        "top3_overlap": len(top3_pred & top3_true),
        "top3_overlap_grid": len(top3_grid & top3_true),
        "winner_logloss": -np.log(p_win),
        "podium_brier": float(((pred["prob_podio"] - podium) ** 2).mean()),
        "podium_brier_base": float(((podium.mean() - podium) ** 2).mean()),
        "dnf_brier": float(((pred["prob_dnf"] - pred["dnf"]) ** 2).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=2024)
    ap.add_argument("--retrain-every", type=int, default=6)
    a = ap.parse_args()

    raw = pd.read_csv(C.RESULTS_CSV)
    table = build_training_table(prepare_results(raw))
    eval_ts = sorted(table.loc[table["season"] >= a.from_season, "t"].unique())
    print(f"Evaluando {len(eval_ts)} carreras (desde {a.from_season}), reentrenando cada {a.retrain_every}\n")

    rows = []
    for i in range(0, len(eval_ts), a.retrain_every):
        chunk = eval_ts[i:i + a.retrain_every]
        train = table[table["t"] < chunk[0]]
        bundle = fit_bundle(train)
        for t in chunk:
            g = table[table["t"] == t]
            m = _race_metrics(bundle, g)
            first = g.iloc[0]
            m.update({"season": int(first["season"]), "round": int(first["round"]), "race": first["race_name"],
                      "segment": "regime_early" if (first["season"] in C.REGIME_CHANGE_SEASONS
                                                    and first["round"] <= 6) else "regular"})
            rows.append(m)
        print(f"  fold hasta {train['season'].iloc[-1] if len(train) else '-'} -> evaluadas {len(chunk)} carreras")

    res = pd.DataFrame(rows)
    res.to_csv(C.DATA_DIR / "backtest_races.csv", index=False)
    num = [c for c in res.columns if c not in ("season", "round", "race", "segment")]
    summary = {"overall": res[num].mean().round(4).to_dict(),
               "by_segment": {k: v[num].mean().round(4).to_dict() for k, v in res.groupby("segment")},
               "by_season": {int(k): v[num].mean().round(4).to_dict() for k, v in res.groupby("season")},
               "n_races": int(len(res))}
    C.METRICS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    show = ["spearman", "spearman_grid", "mae", "mae_grid", "winner_hit", "winner_hit_prob", "winner_hit_grid",
            "top3_overlap", "top3_overlap_grid", "winner_logloss", "podium_brier", "podium_brier_base"]
    print("\n=== GLOBAL ===")
    print(res[show].mean().round(3).to_string())
    print("\n=== POR SEGMENTO ===")
    print(res.groupby("segment")[show].mean().round(3).T.to_string())
    print("\n=== POR TEMPORADA ===")
    print(res.groupby("season")[show].mean().round(3).T.to_string())
    print(f"\n✓ Detalle: {C.DATA_DIR / 'backtest_races.csv'} | Resumen: {C.METRICS_JSON}")


if __name__ == "__main__":
    main()
