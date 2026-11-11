"""Construye features desde data/results.csv y entrena el modelo final con TODO el historial.

Uso:  python train.py
"""
import joblib
import pandas as pd

import config as C
from features import FEATURE_COLS, build_training_table, prepare_results
from model import fit_bundle


def main():
    raw = pd.read_csv(C.RESULTS_CSV)
    print(f"Resultados: {len(raw)} filas, {raw.groupby(['season', 'round']).ngroups} carreras "
          f"({raw['season'].min()}–{raw['season'].max()})")
    prepared = prepare_results(raw)
    table = build_training_table(prepared)
    table.to_csv(C.FEATURES_CSV, index=False)
    print(f"Tabla de entrenamiento: {len(table)} filas, {table['t'].nunique()} carreras")

    bundle = fit_bundle(table)
    joblib.dump(bundle, C.MODEL_PATH)
    print(f"\n✓ Modelo guardado en {C.MODEL_PATH}")
    print(f"  Entrenado hasta: {bundle['trained_through']}")
    print(f"  Temperatura Plackett-Luce (tau): {bundle['tau']:.3f}")

    imp = pd.Series(bundle["ranker"].feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nImportancia de features (ranker):")
    print(imp.head(15).round(3).to_string())


if __name__ == "__main__":
    main()
