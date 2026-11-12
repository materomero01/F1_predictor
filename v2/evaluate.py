"""Backtest walk-forward: para cada carrera evaluada, el modelo solo vio carreras anteriores.

Se reentrena cada `--retrain-every` carreras. Baselines:
  * "grilla": terminar donde largás (solo para Spearman / MAE / top-3 / acierto del ganador)
  * "grid_pl": Plackett-Luce que usa SOLO la grilla (con su propio tau y el mismo modelo de abandono).
    Es el rival honesto para las métricas probabilísticas: ln(20) o la tasa base son baselines demasiado débiles.
Métricas probabilísticas principales (usan toda la distribución de posiciones, menos ruidosas que "el ganador"):
  * rps: Ranked Probability Score de la posición final (menor es mejor)
  * pos_logloss: -log P(posición exacta), suavizado
Además: calibración (ECE) de ganar / podio / puntos / abandono, y tests pareados modelo vs grid_pl
(IC95 bootstrap por carrera y Wilcoxon). Segmentos:
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
from scipy.stats import spearmanr, wilcoxon

import config as C
from features import build_training_table, prepare_results
from model import fit_bundle, predict_race, simulate

N_SIMS = 4000


def _dist_metrics(P: np.ndarray, a: np.ndarray, prefix: str = "") -> dict:
    """P: (sims, n) posiciones simuladas por piloto; a: posición real por piloto."""
    n = P.shape[1]
    ks = np.arange(1, n)
    cdf = (P[:, :, None] <= ks[None, None, :]).mean(axis=0)                  # (n, n-1)
    rps = float(((cdf - (a[:, None] <= ks[None, :])) ** 2).mean())
    exact = (P == a[None, :]).sum(axis=0)
    pos_ll = float(-np.log((exact + 0.5) / (P.shape[0] + 1)).mean())
    p_win = float(np.clip((P[:, a == 1] == 1).mean(), 1e-3, 1)) if (a == 1).any() else np.nan
    return {prefix + "rps": rps, prefix + "pos_logloss": pos_ll, prefix + "winner_logloss": -np.log(p_win)}


def _race_metrics(bundle, g: pd.DataFrame, return_rows: bool = False):
    g = g.reset_index(drop=True)
    pred, P = predict_race(bundle, g, n_sims=N_SIMS, seed=1, return_pos=True)
    pr = pred.set_index("driver_id").loc[g["driver_id"]].reset_index()      # mismo orden que g / columnas de P
    a = g["pos"].to_numpy(float)
    pred = pr.merge(g[["driver_id", "pos", "dnf", "grid_adj"]], on="driver_id")
    pred["rank_pred"] = pred["pos_esperada"].rank(method="first")
    top3_pred = set(pred.nsmallest(3, "pos_esperada")["driver_id"])
    top3_true = set(pred.nsmallest(3, "pos")["driver_id"])
    top3_grid = set(pred.nsmallest(3, "grid_adj")["driver_id"])
    podium = (pred["pos"] <= 3).astype(float)

    # baseline grid_pl: mismo modelo de abandono, puntaje = -log(grilla)
    X = g[bundle["feature_cols"]]
    Pg = simulate(-np.log(g["grid_adj"].to_numpy(float)), bundle["tau_grid"],
                  bundle["dnf"].predict_proba(X)[:, 1], N_SIMS, np.random.default_rng(1))

    out = {
        "spearman": spearmanr(pred["pos_esperada"], pred["pos"])[0],
        "spearman_grid": spearmanr(pred["grid_adj"], pred["pos"])[0],
        "spearman_gridpl": spearmanr(Pg.mean(axis=0), a)[0],
        "mae": (pred["rank_pred"] - pred["pos"]).abs().mean(),
        "mae_grid": (pred["grid_adj"] - pred["pos"]).abs().mean(),
        "winner_hit": float(pred.sort_values("pos_esperada").iloc[0]["pos"] == 1),
        "winner_hit_prob": float(pred.sort_values("prob_ganar", ascending=False).iloc[0]["pos"] == 1),
        "winner_hit_grid": float(pred.sort_values("grid_adj").iloc[0]["pos"] == 1),
        "top3_overlap": len(top3_pred & top3_true),
        "top3_overlap_grid": len(top3_grid & top3_true),
        "podium_brier": float(((pred["prob_podio"] - podium) ** 2).mean()),
        "podium_brier_base": float(((podium.mean() - podium) ** 2).mean()),
        "dnf_brier": float(((pred["prob_dnf"] - pred["dnf"]) ** 2).mean()),
        **_dist_metrics(P, a), **_dist_metrics(Pg, a, "gridpl_"),
    }
    out["winner_logloss"] = out.pop("winner_logloss")                         # (orden estable de columnas)
    if not return_rows:
        return out
    rows = pd.DataFrame({"p_win": pred["prob_ganar"], "y_win": (pred["pos"] == 1).astype(float),
                         "p_podium": pred["prob_podio"], "y_podium": podium,
                         "p_top10": pred["prob_top10"], "y_top10": (pred["pos"] <= 10).astype(float),
                         "p_dnf": pred["prob_dnf"], "y_dnf": pred["dnf"].astype(float)})
    return out, rows


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def paired(res: pd.DataFrame, col: str, base: str, better_low: bool, n_boot: int = 2000, seed: int = 0) -> dict:
    d = (res[col] - res[base]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return {"diff": round(float(d.mean()), 4), "ic95": [round(float(np.percentile(boots, 2.5)), 4),
                                                       round(float(np.percentile(boots, 97.5)), 4)],
            "p_wilcoxon": round(p, 4), "mejor": bool((d.mean() < 0) == better_low)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=2024)
    ap.add_argument("--retrain-every", type=int, default=6)
    a = ap.parse_args()

    raw = pd.read_csv(C.RESULTS_CSV)
    table = build_training_table(prepare_results(raw))
    eval_ts = sorted(table.loc[table["season"] >= a.from_season, "t"].unique())
    print(f"Evaluando {len(eval_ts)} carreras (desde {a.from_season}), reentrenando cada {a.retrain_every}\n")

    rows, drv = [], []
    for i in range(0, len(eval_ts), a.retrain_every):
        chunk = eval_ts[i:i + a.retrain_every]
        train = table[table["t"] < chunk[0]]
        bundle = fit_bundle(train)
        for t in chunk:
            g = table[table["t"] == t]
            m, d = _race_metrics(bundle, g, return_rows=True)
            first = g.iloc[0]
            m.update({"season": int(first["season"]), "round": int(first["round"]), "race": first["race_name"],
                      "segment": "regime_early" if (first["season"] in C.REGIME_CHANGE_SEASONS
                                                    and first["round"] <= 6) else "regular"})
            rows.append(m)
            drv.append(d.assign(season=m["season"]))
        print(f"  fold hasta {train['season'].iloc[-1] if len(train) else '-'} -> evaluadas {len(chunk)} carreras")

    res = pd.DataFrame(rows)
    res.to_csv(C.DATA_DIR / "backtest_races.csv", index=False)
    drv = pd.concat(drv, ignore_index=True)
    num = [c for c in res.columns if c not in ("season", "round", "race", "segment")]
    calib = {k: round(ece(drv[f"p_{k}"].to_numpy(), drv[f"y_{k}"].to_numpy()), 4) for k in ("win", "podium", "top10", "dnf")}
    tests = {
        "rps": paired(res, "rps", "gridpl_rps", True), "pos_logloss": paired(res, "pos_logloss", "gridpl_pos_logloss", True),
        "winner_logloss": paired(res, "winner_logloss", "gridpl_winner_logloss", True),
        "spearman": paired(res, "spearman", "spearman_gridpl", False),
    }
    summary = {"overall": res[num].mean().round(4).to_dict(),
               "by_segment": {k: v[num].mean().round(4).to_dict() for k, v in res.groupby("segment")},
               "by_season": {int(k): v[num].mean().round(4).to_dict() for k, v in res.groupby("season")},
               "ece": calib, "paired_vs_gridpl": tests, "n_races": int(len(res))}
    C.METRICS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    show = ["spearman", "spearman_grid", "spearman_gridpl", "mae", "mae_grid", "winner_hit_prob", "winner_hit_grid",
            "top3_overlap", "top3_overlap_grid", "rps", "gridpl_rps", "pos_logloss", "gridpl_pos_logloss",
            "winner_logloss", "gridpl_winner_logloss", "podium_brier", "podium_brier_base", "dnf_brier"]
    print("\n=== GLOBAL ===")
    print(res[show].mean().round(4).to_string())
    print("\n=== CALIBRACION (ECE, menor es mejor) ===")
    print(pd.Series(calib).to_string())
    print("\n=== MODELO vs grid_pl (diff = modelo - grid_pl; IC95 bootstrap por carrera) ===")
    print(pd.DataFrame(tests).T.to_string())
    print("\n=== POR SEGMENTO ===")
    print(res.groupby("segment")[show].mean().round(4).T.to_string())
    print("\n=== POR TEMPORADA ===")
    print(res.groupby("season")[show].mean().round(4).T.to_string())
    print(f"\n✓ Detalle: {C.DATA_DIR / 'backtest_races.csv'} | Resumen: {C.METRICS_JSON}")


if __name__ == "__main__":
    main()
