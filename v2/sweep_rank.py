"""Objetivo del ranker: pairwise vs. ndcg enfocado en la punta (backtest walk-forward desde 2024)."""
import pandas as pd
import xgboost as xgb
import config as C
import model as M
from evaluate import _race_metrics
from features import build_training_table, prepare_results

T = build_training_table(prepare_results(pd.read_csv(C.RESULTS_CSV)))
ts = sorted(T.loc[T.season >= 2024, "t"].unique())
SHOW = ["spearman", "winner_hit_prob", "winner_hit_grid", "winner_logloss", "podium_brier", "top3_overlap"]
BASE = dict(n_estimators=250, learning_rate=0.05, max_depth=3, min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, tree_method="hist", random_state=C.RANDOM_STATE, monotone_constraints=M.MONOTONE)
VARIANTS = {
    "pairwise": dict(objective="rank:pairwise"),
    "pairwise_100": dict(objective="rank:pairwise", n_estimators=100),
    "ndcg_topk3": dict(objective="rank:ndcg", lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=3, ndcg_exp_gain=False),
    "ndcg_topk6": dict(objective="rank:ndcg", lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=6, ndcg_exp_gain=False),
}
for name, extra in VARIANTS.items():
    def fit(train, extra=extra):
        train = train.sort_values(["t", "pos"]).reset_index(drop=True)
        y = (train["n_entries"] - train["pos"] + 1).clip(lower=0).astype(int)
        gw = M.race_weights(train).groupby(train["t"], sort=True).first().to_numpy()
        m = xgb.XGBRanker(**{**BASE, **extra})
        m.fit(train[M.FEATURE_COLS], y, qid=train["t"], sample_weight=gw)
        return m
    M._fit_ranker = fit
    rows = []
    for i in range(0, len(ts), 6):
        ch = ts[i:i + 6]
        b = M.fit_bundle(T[T.t < ch[0]])
        rows += [_race_metrics(b, T[T.t == t]) for t in ch]
    print(name, pd.DataFrame(rows)[SHOW].mean().round(3).to_dict(), flush=True)
