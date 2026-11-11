"""Predicción de una carrera.

Ejemplos:
    # próxima carrera del calendario, grilla estimada (pre-clasificación)
    python predict.py

    # próxima carrera con la grilla real (códigos en orden de largada, pole primero)
    python predict.py --grid RUS,ANT,NOR,PIA,LEC,HAM,...

    # sanity check: "predecir" una carrera pasada usando solo datos anteriores
    python predict.py --backtest 2026 18
"""
import argparse

import joblib
import numpy as np
import pandas as pd

import config as C
from features import FeatureBuilder, build_inference_features, canonical_team, prepare_results
from model import predict_race


def load_all():
    raw = pd.read_csv(C.RESULTS_CSV)
    sched = pd.read_csv(C.SCHEDULE_CSV)
    bundle = joblib.load(C.MODEL_PATH)
    return raw, sched, bundle


def next_race(raw: pd.DataFrame, sched: pd.DataFrame) -> pd.Series:
    done = set(zip(raw["season"], raw["round"]))
    pend = sched[[(s, r) not in done for s, r in zip(sched["season"], sched["round"])]]
    pend = pend[pend["season"] >= raw["season"].max()].sort_values(["season", "round"])
    if pend.empty:
        raise SystemExit("No hay carreras pendientes en el calendario cargado (¿temporada terminada?).")
    return pend.iloc[0]


def default_lineup(raw: pd.DataFrame) -> pd.DataFrame:
    """Alineación = pilotos de la última carrera disputada."""
    last = raw[(raw["season"] == raw["season"].max())]
    last = last[last["round"] == last["round"].max()]
    return last[["driver_id", "code", "driver_name", "constructor_id", "constructor_name"]].reset_index(drop=True)


def estimate_grid(raw: pd.DataFrame, season: int, rnd: int, circuit_id: str, lineup: pd.DataFrame) -> pd.DataFrame:
    """Grilla estimada = ranking por (rendimiento del auto + delta de clasificación del piloto)."""
    prepared = prepare_results(raw)
    e = lineup.copy()
    e["grid"] = np.arange(1, len(e) + 1, dtype=float)       # placeholder, no entra en car/drv
    f = build_inference_features(prepared, season, rnd, circuit_id, e)
    n = len(e)
    score = f["car_q"] * (n / 2) + f["drv_quali_delta"]
    est = f.assign(score=score).sort_values("score")
    pos = {d: i + 1 for i, d in enumerate(est["driver_id"])}
    e["grid"] = e["driver_id"].map(pos).astype(float)
    return e


def run(raw, bundle, season, rnd, circuit_id, entries) -> pd.DataFrame:
    prepared = prepare_results(raw)
    feats = build_inference_features(prepared, season, rnd, circuit_id, entries)
    pred = predict_race(bundle, feats)
    meta = entries[["driver_id"] + [c for c in ("code", "constructor_name", "grid") if c in entries]]
    return pred.merge(meta, on="driver_id", how="left")


def show(df: pd.DataFrame):
    cols = [c for c in ("code", "constructor_name", "grid", "pos_esperada", "pos_p10", "pos_p90",
                        "prob_ganar", "prob_podio", "prob_top10", "prob_dnf", "pos_real") if c in df]
    out = df[cols].copy()
    for c in ("prob_ganar", "prob_podio", "prob_top10", "prob_dnf"):
        out[c] = (out[c] * 100).round(1)
    out["pos_esperada"] = out["pos_esperada"].round(1)
    out.index = out.index + 1
    print(out.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", help="códigos separados por coma, pole primero")
    ap.add_argument("--backtest", nargs=2, type=int, metavar=("SEASON", "ROUND"))
    a = ap.parse_args()
    raw, sched, bundle = load_all()
    print(f"Modelo entrenado hasta: {bundle['trained_through']}\n")

    if a.backtest:
        s, r = a.backtest
        race = raw[(raw["season"] == s) & (raw["round"] == r)]
        hist = raw[(raw["season"] < s) | ((raw["season"] == s) & (raw["round"] < r))]
        entries = race[["driver_id", "code", "driver_name", "constructor_id", "constructor_name",
                        "grid", "quali_pos", "sprint_pos", "is_sprint_weekend"]].copy()
        entries["is_sprint"] = entries["is_sprint_weekend"]
        print(f"Predicción retroactiva: {race.iloc[0]['race_name']} {s} (solo datos previos; "
              f"ojo: el modelo final igual se entrenó con esta carrera, usá evaluate.py para métricas limpias)\n")
        out = run(hist, bundle, s, r, race.iloc[0]["circuit_id"], entries)
        out = out.merge(race[["driver_id", "position"]].rename(columns={"position": "pos_real"}), on="driver_id")
        show(out)
        return

    nr = next_race(raw, sched)
    print(f"Próxima carrera: {nr['season']} R{nr['round']} – {nr['race_name']} ({nr['date']})\n")
    lineup = default_lineup(raw)
    if a.grid:
        codes = [c.strip().upper() for c in a.grid.split(",")]
        order = {c: i + 1 for i, c in enumerate(codes)}
        missing = set(lineup["code"]) - set(order)
        if missing:
            raise SystemExit(f"Faltan pilotos en --grid: {sorted(missing)}")
        entries = lineup.assign(grid=lineup["code"].map(order).astype(float))
    else:
        entries = estimate_grid(raw, int(nr["season"]), int(nr["round"]), nr["circuit_id"], lineup)
        print("(grilla ESTIMADA: pasá --grid con la real después de clasificar)\n")
    entries["is_sprint"] = int(nr["is_sprint_weekend"])
    show(run(raw, bundle, int(nr["season"]), int(nr["round"]), nr["circuit_id"], entries))


if __name__ == "__main__":
    main()
