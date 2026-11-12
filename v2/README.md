# 🏎️ F1 Race Predictor v2

Predice el resultado de una carrera de F1 (posición esperada, % de ganar / podio / puntos / abandono)
a partir de la grilla, el rendimiento reciente del auto y del piloto, y el circuito.

## Qué cambió respecto a la v1

| v1 | v2 |
|---|---|
| Datos 2021–2025 vía `session.load()` (pesado), sin sprints | 2018–año actual vía Jolpica-F1 (liviano), con sprints y clasificación |
| Features distintas en train y en predicción (el año en curso se ignoraba) | **Un solo `FeatureBuilder`** para ambos, estrictamente causal |
| Circuito calculado pero no usado | Features de circuito (adelantamientos, abandonos, historial del piloto) |
| Rookies con penalizaciones a mano | Shrinkage estadístico (sin parches) |
| Un regresor por piloto, `scale_pos_weight` | XGBRanker por carrera + simulación Plackett-Luce calibrada |
| Split "temporal" alfabético, accuracy | Backtest walk-forward, Spearman, podio, log-loss, Brier, vs baseline de la grilla |

## Cambios de reglamento (2022, 2026)

`config.REGIME_CHANGE_SEASONS` marca los años donde el orden de los autos se reinicia. En esos años:
- el prior del auto (año anterior) se encoge ~75% hacia la media y pierde peso rápido (`PRIOR_NEW_REGIME`);
- el rendimiento del auto se mide con una ventana de las últimas 5 carreras de la temporada en curso;
- se pooleaan equipos con el mismo motor (`config.power_unit`), útil cuando un cliente rinde igual que el oficial;
- el piloto se mide **contra su compañero de equipo**, así su habilidad se arrastra aunque cambie el auto;
- el modelo ve 2022 y 2026 en el entrenamiento, y `evaluate.py` reporta aparte las primeras 6 carreras de esos años.

Equipos nuevos (ej. Cadillac) no tienen prior: arrancan en la media y aprenden rápido.
Si Jolpica usa un `constructorId` que no está en `config.TEAM_LINEAGE`, `collect_data.py` te avisa.

## Uso

```bash
pip install -r requirements.txt
python collect_data.py        # baja resultados (cachea años cerrados)
python train.py               # features + modelo final
python evaluate.py            # backtest walk-forward (desde 2024; --from-season 2022 para ver el arranque del reglamento anterior)
python predict.py             # próxima carrera, grilla estimada
python predict.py --grid RUS,ANT,NOR,...   # con la grilla real
python predict.py --backtest 2026 18       # carrera pasada, solo con datos previos
streamlit run front.py
python tests/test_pipeline.py # tests con datos sintéticos (sin internet)
```

Después de cada carrera: `python collect_data.py && python train.py`.

## Limitaciones honestas

- La grilla es el predictor más fuerte; antes de clasificar el modelo estima la grilla y la precisión baja.
- No modela clima, estrategia, safety cars ni mejoras entre carreras más allá de lo que se refleja en resultados.
- Con pocas carreras del nuevo reglamento los intervalos P10–P90 son anchos a propósito.
- Las métricas reales dependen de los datos: mirá la salida de `evaluate.py` antes de confiar en las probabilidades.

## Resultados (backtest walk-forward, 2026-11-11)

Datos: Jolpica-F1, 2018–2026 R20 (193 carreras). Evaluación: 68 carreras desde 2024, reentrenando cada 6; el modelo de
cada fold solo ve carreras anteriores. `regime_early` = primeras 6 carreras de 2022/2026.

| Métrica | Modelo | Grilla (baseline) |
|---|---|---|
| Spearman | 0.703 | 0.677 |
| MAE de posición | 3.17 | 3.25 |
| Podio acertado (de 3) | 2.02 | 2.03 |
| Ganador acertado (pick = mayor prob.) | 0.60 | 0.60 (pole) |
| Log-loss del ganador | 1.29 | ln 20 ≈ 3.0 |
| Brier de podio | 0.064 | 0.125 (tasa base) |

Por segmento (Spearman): `regular` 0.717 vs grilla 0.687; `regime_early` 0.558 vs grilla 0.571.

Lectura honesta: ordenando, el modelo mejora un poco a la grilla en carreras normales, y su valor principal está en las
**probabilidades calibradas** (`TAU_TOP_K`, `GRID_BLEND_BETA`). Para elegir al ganador hoy coincide con la pole (0.60 = 0.60):
sin el término de grilla era 0.44. En el arranque de un reglamento nuevo no supera a la grilla (Spearman y podio);
con 6 carreras por segmento y un solo cambio de reglamento histórico, esas cifras son ruidosas.
Experimentos: `sweep_regime.py` (priors de reglamento), `sweep_tau.py` (tau), `sweep_rank.py` (objetivo del ranker, sin mejora),
`sweep_gridblend.py` (peso de la grilla: meseta en β 0.8–1.6).

### Pre-clasificación (grilla estimada, `backtest_pre_quali.py`, últimas 30 carreras)
Spearman 0.600 (grilla estimada sola: 0.586; con grilla real el modelo llega a ~0.70) · ganador acertado 0.33 (igual que la grilla estimada) ·
log-loss del ganador 2.44 · Brier de podio 0.095 (tasa base 0.121). Útil como adelanto probabilístico, pero **mucho menos preciso
que después de clasificar**: conviene volver a predecir con `--grid` cuando haya clasificación.

### Modelo de clasificación + incertidumbre de grilla (`prequali.py`, `eval_prequali.py`)
Antes de clasificar, `predict.py` y el front (modo "Estimada") ya no tratan una grilla estimada como cierta: un ranker propio predice la
clasificación, se muestrean 300 grillas posibles y la carrera se promedia sobre ellas. Walk-forward, 68 carreras desde 2024:

| | heurística (grilla única) | modelo + muestreo |
|---|---|---|
| Spearman del orden de clasificación | 0.721 | 0.739 |
| Log-loss del ganador | 2.15 | 2.06 |
| Brier de podio | 0.094 | 0.087 |
| Spearman de carrera | 0.612 | 0.601 |

La mejora es en **probabilidades** (menos sobreconfianza: antes Verstappen salía con 71% de ganar en Las Vegas, ahora ~20%); el orden de
carrera no mejora. Con 40 grillas el resultado variaba demasiado entre corridas; con 300 queda en ±1–1.5 puntos.
