"""Prediccion PRE-clasificacion: modelo de clasificacion + incertidumbre de grilla propagada a la carrera.

1. Un ranker propio predice el orden de clasificacion con features que NO dependen de la grilla.
2. Se muestrean K grillas (Plackett-Luce sobre esos puntajes); para cada una se recalculan las features de
   grilla y se simula la carrera con el modelo normal. Las probabilidades finales promedian sobre las K grillas.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

import config as C
from features import FEATURE_COLS
from model import MONOTONE, fit_tau, race_weights, score, simulate

GRID_COLS = {"grid_adj", "grid_rel", "quali_adj", "grid_pen", "grid_minus_car", "quali_minus_car", "sprint_pos"}
QUALI_COLS = [c for c in FEATURE_COLS if c not in GRID_COLS]


def _fit_quali_ranker(train: pd.DataFrame) -> xgb.XGBRanker:
    train = train.sort_values(["t", "quali_adj"]).reset_index(drop=True)
    y = (train["n_entries"] - train["quali_adj"] + 1).clip(lower=0).astype(int)
    gw = race_weights(train).groupby(train["t"], sort=True).first().to_numpy()
    m = xgb.XGBRanker(
        objective="rank:pairwise", n_estimators=250, learning_rate=0.05, max_depth=3, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist", random_state=C.RANDOM_STATE,
        monotone_constraints={c: v for c, v in MONOTONE.items() if c in QUALI_COLS})
    m.fit(train[QUALI_COLS], y, qid=train["t"], sample_weight=gw)
    return m


def fit_quali(table: pd.DataFrame, cal_races: int = C.CAL_RACES) -> dict:
    table = table.sort_values(["t", "quali_adj"]).reset_index(drop=True)
    ts = sorted(table["t"].unique())
    if len(ts) >= cal_races + 25:
        cal_t = set(ts[-cal_races:])
        r0 = _fit_quali_ranker(table[~table["t"].isin(cal_t)])
        cal = table[table["t"].isin(cal_t)]
        tau = fit_tau([(r0.predict(g[QUALI_COLS]), g["quali_adj"].to_numpy()) for _, g in cal.groupby("t")])
    else:
        r0 = _fit_quali_ranker(table)
        tau = 1.5 * fit_tau([(r0.predict(g[QUALI_COLS]), g["quali_adj"].to_numpy()) for _, g in table.groupby("t")])
    return {"ranker": _fit_quali_ranker(table), "tau": tau}


def regrid(feats: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Recalcula las columnas que dependen de la grilla para una grilla hipotetica (1..n, sin penalizaciones)."""
    n = len(feats)
    rel = (grid - (n + 1) / 2) / (n / 2)
    return feats.assign(grid_adj=grid, grid_rel=rel, quali_adj=grid, grid_pen=0.0,
                        grid_minus_car=rel - feats["car_q"], quali_minus_car=rel - feats["car_q"])


def sample_grids(qb: dict, feats: pd.DataFrame, k: int, rng) -> np.ndarray:
    """(k, n) posiciones de salida muestreadas del modelo de clasificacion."""
    s = qb["ranker"].predict(feats[QUALI_COLS])
    util = s / qb["tau"] + rng.gumbel(size=(k, len(s)))
    order = np.argsort(-util, axis=1)
    grid = np.empty_like(order)
    grid[np.arange(k)[:, None], order] = np.arange(1, len(s) + 1)
    return grid.astype(float)


def predict_prequali(bundle: dict, qb: dict, feats: pd.DataFrame, n_grids: int = 300, sims_per: int = 40,
                     seed: int = 0) -> pd.DataFrame:
    """feats: features de inferencia (la grilla que traigan se ignora). Mismo formato de salida que predict_race."""
    rng = np.random.default_rng(seed)
    grids = sample_grids(qb, feats, n_grids, rng)
    pos_all, grid_all, dnf_all = [], [], []
    for g in grids:
        f = regrid(feats, g)
        X = f[bundle["feature_cols"]]
        p_dnf = bundle["dnf"].predict_proba(X)[:, 1]
        pos_all.append(simulate(score(bundle["ranker"], X, bundle["grid_beta"]), bundle["tau"], p_dnf, sims_per, rng))
        dnf_all.append(p_dnf)
        grid_all.append(g)
    pos = np.vstack(pos_all)
    out = feats[["driver_id"]].copy()
    out["grid_esperada"] = np.mean(grid_all, axis=0)
    out["pos_esperada"] = pos.mean(axis=0)
    out["prob_ganar"] = (pos == 1).mean(axis=0)
    out["prob_podio"] = (pos <= 3).mean(axis=0)
    out["prob_top10"] = (pos <= 10).mean(axis=0)
    out["prob_dnf"] = np.mean(dnf_all, axis=0)
    out["pos_p10"] = np.percentile(pos, 10, axis=0)
    out["pos_p90"] = np.percentile(pos, 90, axis=0)
    return out.sort_values("pos_esperada").reset_index(drop=True)
