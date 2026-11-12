"""Evaluacion honesta del ajuste de parametros: se ELIGE con carreras 2024-2025 y se MIDE en 2026.

Grilla de configuraciones (los 4 parametros que ajustamos mirando el backtest):
  CAR_WINDOW x prior_races(reglamento nuevo) x GRID_BLEND_BETA x TAU_TOP_K
Referencia "original" = lo que habia antes de ajustar: window 5, prior_races 1.5, beta 0, tau con todo el campo.
    python sweep_honest.py
"""
import itertools

import numpy as np
import pandas as pd

import config as C
import evaluate as E
import model as M
from features import build_training_table, prepare_results

raw = pd.read_csv(C.RESULTS_CSV)
_cache, _orig = {}, (M._fit_ranker, M._fit_dnf)
cfg_key = [None]


def cached(fn, tag):
    def wrapper(train):
        k = (tag, cfg_key[0], len(train), int(train["t"].max()))
        if k not in _cache:
            _cache[k] = fn(train)
        return _cache[k]
    return wrapper


M._fit_ranker, M._fit_dnf = cached(_orig[0], "r"), cached(_orig[1], "d")
SHOW = ["rps", "gridpl_rps", "pos_logloss", "winner_logloss", "spearman", "mae"]
rows = []
for window, prior in itertools.product([5, 8], [1.5, 3.0]):
    C.CAR_WINDOW, C.PRIOR_NEW_REGIME = window, {"shrink": 0.75, "prior_races": prior}
    cfg_key[0] = (window, prior)
    T = build_training_table(prepare_results(raw))
    ts = sorted(T.loc[T["season"] >= 2024, "t"].unique())
    for beta, topk in itertools.product([0.0, 1.0], [None, 3]):
        C.GRID_BLEND_BETA, C.TAU_TOP_K = beta, topk
        M.score.__defaults__ = (beta,)
        per = []
        for i in range(0, len(ts), 6):
            ch = ts[i:i + 6]
            b = M.fit_bundle(T[T["t"] < ch[0]])
            for t in ch:
                g = T[T["t"] == t]
                per.append({"season": int(g["season"].iloc[0]), **E._race_metrics(b, g)})
        d = pd.DataFrame(per)
        sel, test = d[d["season"] <= 2025][SHOW].mean(), d[d["season"] == 2026][SHOW].mean()
        rows.append({"window": window, "prior": prior, "beta": beta, "topk": topk,
                     **{f"sel_{k}": round(v, 3) for k, v in sel.items()}, **{f"test_{k}": round(v, 3) for k, v in test.items()}})
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv(C.DATA_DIR / "sweep_honest.csv", index=False)
best = df.sort_values("sel_rps").iloc[0]      # RPS usa toda la distribucion de posiciones; el ganador es 1 evento/carrera (ruidoso)
orig = df[(df.window == 5) & (df.prior == 1.5) & (df.beta == 0.0) & (df.topk.isna())].iloc[0]
cur = df[(df.window == 8) & (df.prior == 3.0) & (df.beta == 1.0) & (df.topk == 3)].iloc[0]
cols = ["window", "prior", "beta", "topk"] + [f"{p}_{m}" for p in ("sel", "test") for m in SHOW]
print("\n=== sel = 2024-2025 (donde se ELIGE) | test = 2026 (donde se MIDE) ===")
print(pd.DataFrame([orig, cur, best], index=["original (sin ajustar)", "actual (ajustado en todo)", "elegido SOLO con 2024-25"])[cols].T.to_string())
