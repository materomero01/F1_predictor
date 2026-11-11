"""Modelo v2: ranker por carrera + Plackett-Luce (calibrado) + prob. de abandono.

Por qué así:
  * Una carrera es un ordenamiento, no 20 regresiones independientes -> XGBRanker
    entrenado por carrera (grupos).
  * Las probabilidades salen de simular carreras con Plackett-Luce (ruido Gumbel):
    las prob. de ganar suman 100%, y podio / top 10 / posición esperada son
    coherentes entre sí. La "temperatura" tau se ajusta con carreras reservadas.
  * El abandono (DNF) se modela aparte y manda al fondo al piloto en la simulación.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

import config as C
from features import FEATURE_COLS


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------

# Restricciones de dominio (score alto = mejor resultado): largar más adelante, tener
# mejor auto o ser mejor que el compañero nunca debería empeorar la predicción.
# Estabiliza el modelo con pocos datos y evita que "invente" relaciones al revés.
MONOTONE = {c: -1 for c in ("grid_adj", "grid_rel", "quali_adj", "car_q", "car_f", "car_q_gap",
                            "car_f_gap", "pu_car_q", "pu_car_f", "drv_race_delta", "drv_quali_delta")}

def race_weights(table: pd.DataFrame) -> pd.Series:
    latest = table["season"].max()
    return 0.5 ** ((latest - table["season"]) / C.RECENCY_HALF_LIFE_SEASONS)


def _fit_ranker(train: pd.DataFrame) -> xgb.XGBRanker:
    train = train.sort_values(["t", "pos"]).reset_index(drop=True)
    y = (train["n_entries"] - train["pos"] + 1).clip(lower=0).astype(int)
    gw = race_weights(train).groupby(train["t"], sort=True).first().to_numpy()
    model = xgb.XGBRanker(
        objective="rank:pairwise", n_estimators=250, learning_rate=0.05, max_depth=3,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
        tree_method="hist", random_state=C.RANDOM_STATE, monotone_constraints=MONOTONE)
    model.fit(train[FEATURE_COLS], y, qid=train["t"], sample_weight=gw)
    return model


def _fit_dnf(train: pd.DataFrame) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=2, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist",
        random_state=C.RANDOM_STATE)
    clf.fit(train[FEATURE_COLS], train["dnf"].astype(int), sample_weight=race_weights(train))
    return clf


def score(ranker, X: pd.DataFrame, beta: float = C.GRID_BLEND_BETA) -> np.ndarray:
    """Puntaje del ranker + termino de grilla: el pairwise subpondera la punta (ver config.GRID_BLEND_BETA)."""
    return ranker.predict(X) - beta * np.log(X["grid_adj"].to_numpy())


def _pl_neg_loglik(log_tau: float, races: list[tuple[np.ndarray, np.ndarray]]) -> float:
    tau = np.exp(log_tau)
    total, cnt = 0.0, 0
    for scores, pos in races:
        s = scores[np.argsort(pos)] / tau               # mejor -> peor
        for k in range(min(len(s) - 1, C.TAU_TOP_K or len(s))):
            total += s[k] - logsumexp(s[k:])
            cnt += 1
    return -total / max(cnt, 1)


def fit_tau(scores_pos: list[tuple[np.ndarray, np.ndarray]]) -> float:
    res = minimize_scalar(_pl_neg_loglik, bounds=(-4, 4), args=(scores_pos,), method="bounded")
    return float(np.exp(res.x))


def fit_bundle(table: pd.DataFrame, cal_races: int = C.CAL_RACES) -> dict:
    table = table.sort_values(["t", "pos"]).reset_index(drop=True)
    ts = sorted(table["t"].unique())
    if len(ts) >= cal_races + 25:
        cal_t = set(ts[-cal_races:])
        tr, cal = table[~table["t"].isin(cal_t)], table[table["t"].isin(cal_t)]
        r0 = _fit_ranker(tr)
        sp = [(score(r0, g[FEATURE_COLS]), g["pos"].to_numpy()) for _, g in cal.groupby("t")]
        tau = fit_tau(sp)
    else:                                   # pocos datos: tau en-muestra, inflado por seguridad
        r0 = _fit_ranker(table)
        sp = [(score(r0, g[FEATURE_COLS]), g["pos"].to_numpy()) for _, g in table.groupby("t")]
        tau = fit_tau(sp) * 1.5
    ranker = _fit_ranker(table)             # reentrena con TODO (incluye las carreras de calibración)
    dnf = _fit_dnf(table)
    last = table.sort_values("t").iloc[-1]
    return {
        "ranker": ranker, "dnf": dnf, "tau": tau, "grid_beta": C.GRID_BLEND_BETA, "feature_cols": FEATURE_COLS,
        "trained_through": f"{int(last['season'])} R{int(last['round'])} {last['race_name']}",
        "n_races": int(table["t"].nunique()), "xgboost": xgb.__version__,
    }


# --------------------------------------------------------------------------
# Predicción
# --------------------------------------------------------------------------

def simulate(scores: np.ndarray, tau: float, p_dnf: np.ndarray, n_sims: int, rng) -> np.ndarray:
    n = len(scores)
    util = scores / tau + rng.gumbel(size=(n_sims, n))
    dnf = rng.random((n_sims, n)) < p_dnf
    util = np.where(dnf, util - 1e3, util)
    order = np.argsort(-util, axis=1)
    pos = np.empty_like(order)
    pos[np.arange(n_sims)[:, None], order] = np.arange(1, n + 1)
    return pos


def predict_race(bundle: dict, feats: pd.DataFrame, n_sims: int = C.N_SIMS, seed: int = 0) -> pd.DataFrame:
    X = feats[bundle["feature_cols"]]
    scores = score(bundle["ranker"], X, bundle["grid_beta"])
    p_dnf = bundle["dnf"].predict_proba(X)[:, 1]
    pos = simulate(scores, bundle["tau"], p_dnf, n_sims, np.random.default_rng(seed))
    out = feats[["driver_id"]].copy()
    out["score"] = scores
    out["pos_esperada"] = pos.mean(axis=0)
    out["prob_ganar"] = (pos == 1).mean(axis=0)
    out["prob_podio"] = (pos <= 3).mean(axis=0)
    out["prob_top10"] = (pos <= 10).mean(axis=0)
    out["prob_dnf"] = p_dnf
    out["pos_p10"] = np.percentile(pos, 10, axis=0)
    out["pos_p90"] = np.percentile(pos, 90, axis=0)
    return out.sort_values("pos_esperada").reset_index(drop=True)
