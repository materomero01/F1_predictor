"""Tests con datos sintéticos (no necesitan internet).

    python tests/test_pipeline.py

Verifica: (1) no hay fuga de datos futuros, (2) entrenamiento / predicción
end-to-end, (3) las probabilidades suman 1, (4) mismo código en train e inferencia.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from features import FEATURE_COLS, build_inference_features, build_training_table, prepare_results  # noqa: E402
from model import fit_bundle, predict_race  # noqa: E402


def make_synthetic(seed=0, seasons=range(2018, 2027), races=18):
    rng = np.random.default_rng(seed)
    teams = ["red_bull", "mercedes", "ferrari", "mclaren", "alpine", "aston_martin",
             "williams", "haas", "sauber", "rb"]
    drivers = [(f"drv{i}", teams[i // 2]) for i in range(20)]
    skill = {d: rng.normal(0, 1.0) for d, _ in drivers}
    rows = []
    strength = {t: rng.normal(0, 2) for t in teams}
    for s in seasons:
        if s in C.REGIME_CHANGE_SEASONS:      # reinicio parcial del orden de autos
            strength = {t: 0.3 * strength[t] + rng.normal(0, 2) for t in teams}
        for r in range(1, races + 1):
            circuit = f"circ{r % 12}"
            perf = {d: -(strength[t] + skill[d]) + rng.normal(0, 1.5) for d, t in drivers}
            qual = {d: v + rng.normal(0, 0.5) for d, v in perf.items()}
            q_order = sorted(qual, key=qual.get)
            order = sorted(perf, key=perf.get)
            dnf = {d: rng.random() < 0.1 for d, _ in drivers}
            fin = [d for d in order if not dnf[d]] + [d for d in order if dnf[d]]
            sprint = r % 6 == 0
            for pos, d in enumerate(fin, 1):
                t = dict(drivers)[d]
                rows.append({
                    "season": s, "round": r, "race_name": f"GP{r}", "circuit_id": circuit,
                    "date": f"{s}-{(r % 12) + 1:02d}-15", "driver_id": d, "code": d.upper(),
                    "driver_name": d, "constructor_id": t, "constructor_name": t,
                    "grid": q_order.index(d) + 1, "quali_pos": q_order.index(d) + 1,
                    "sprint_pos": float(q_order.index(d) + 1) if sprint else np.nan,
                    "is_sprint_weekend": int(sprint), "position": float(pos), "position_text": str(pos),
                    "points": [25, 18, 15, 12, 10, 8, 6, 4, 2, 1][pos - 1] if pos <= 10 else 0.0,
                    "laps": 50.0, "status": "Finished" if not dnf[d] else "Engine",
                })
    return pd.DataFrame(rows)


def test_no_leakage(raw):
    """Cambiar el resultado de una carrera NO debe alterar sus features ni las anteriores."""
    prepared = prepare_results(raw)
    base = build_training_table(prepared)
    t_mid = int(base["t"].quantile(0.7))
    tampered = raw.copy()
    mask = (tampered["season"] == base[base["t"] == t_mid]["season"].iloc[0]) & \
           (tampered["round"] == base[base["t"] == t_mid]["round"].iloc[0])
    tampered.loc[mask, "position"] = tampered.loc[mask, "position"].iloc[::-1].to_numpy()
    tampered.loc[mask, "status"] = "Accident"
    new = build_training_table(prepare_results(tampered))
    a = base[base["t"] <= t_mid].sort_values(["t", "driver_id"])[FEATURE_COLS].reset_index(drop=True)
    b = new[new["t"] <= t_mid].sort_values(["t", "driver_id"])[FEATURE_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    after = base[base["t"] > t_mid].sort_values(["t", "driver_id"])[FEATURE_COLS].reset_index(drop=True)
    after2 = new[new["t"] > t_mid].sort_values(["t", "driver_id"])[FEATURE_COLS].reset_index(drop=True)
    assert not np.allclose(after.fillna(0).to_numpy(), after2.fillna(0).to_numpy()), "las futuras sí deben cambiar"
    print("✓ sin fuga de datos futuros")


def test_train_infer_consistency(raw):
    """Features de inferencia == features de entrenamiento para la misma carrera."""
    prepared = prepare_results(raw)
    table = build_training_table(prepared)
    last = raw[(raw["season"] == raw["season"].max()) & (raw["round"] == raw["round"].max())]
    hist = raw.drop(last.index)
    entries = last[["driver_id", "constructor_id", "grid", "quali_pos", "sprint_pos", "is_sprint_weekend"]].copy()
    entries["is_sprint"] = entries["is_sprint_weekend"]
    inf = build_inference_features(prepare_results(hist), int(last.iloc[0]["season"]),
                                   int(last.iloc[0]["round"]), last.iloc[0]["circuit_id"], entries)
    tr = table[table["t"] == table["t"].max()]
    a = inf.sort_values("driver_id")[FEATURE_COLS].reset_index(drop=True)
    b = tr.sort_values("driver_id")[FEATURE_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    print("✓ features de train == inferencia")


def test_model(raw):
    table = build_training_table(prepare_results(raw))
    cut = table["t"].max() - 5
    bundle = fit_bundle(table[table["t"] < cut])
    sp = []
    for t in range(cut, cut + 5):
        g = table[table["t"] == t]
        p = predict_race(bundle, g, n_sims=3000).merge(g[["driver_id", "pos", "grid_adj"]], on="driver_id")
        assert abs(p["prob_ganar"].sum() - 1) < 1e-6, p["prob_ganar"].sum()
        assert (p["prob_podio"].sum() - 3) < 1e-6
        sp.append((p["pos_esperada"].corr(p["pos"], method="spearman"),
                   p["grid_adj"].corr(p["pos"], method="spearman")))
    m = np.mean(sp, axis=0)
    print(f"✓ modelo OK | tau={bundle['tau']:.2f} | spearman modelo={m[0]:.2f} vs grilla={m[1]:.2f}")
    assert m[0] > 0.3


def test_regime_prior(raw):
    """En un año de cambio de reglamento el prior del auto debe pesar menos."""
    prepared = prepare_results(raw)
    table = build_training_table(prepared)
    r22 = table[(table["season"] == 2022) & (table["round"] == 1)]
    r21 = table[(table["season"] == 2021) & (table["round"] == 1)]
    assert (r22["regime_new"] == 1).all() and (r21["regime_new"] == 0).all()
    print("✓ flag de cambio de reglamento")


if __name__ == "__main__":
    raw = make_synthetic()
    test_no_leakage(raw)
    test_train_infer_consistency(raw)
    test_regime_prior(raw)
    test_model(raw)
    print("\nTodos los tests pasaron.")
