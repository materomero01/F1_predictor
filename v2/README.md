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

Datos: Jolpica-F1, 2018–2026 R20. Cada carrera se predice con un modelo entrenado solo con carreras anteriores (reentrena cada 6).
Con **grilla real**. El baseline relevante es `grid_pl`: un Plackett-Luce que usa *solo* la grilla (con su tau y el mismo modelo de abandono).
Comparar contra ln 20 o contra la tasa base sería engañoso: esos baselines son muy débiles. Detalle completo: `python evaluate.py`.

| 68 carreras (2024+) | Modelo | Solo grilla (`grid_pl`) | Diferencia (IC95 bootstrap) |
|---|---|---|---|
| RPS de la posición final (menor = mejor) | 0.1089 | 0.1157 | −0.0067 [−0.0096, −0.0040] |
| Log-loss de la posición exacta | 2.557 | 2.624 | −0.067 [−0.098, −0.035] |
| Log-loss del ganador | 1.307 | 1.370 | −0.062 [−0.164, +0.033] (no significativo) |
| Spearman | 0.709 | 0.677 | +0.032 [+0.014, +0.051] |
| Error medio de posición (MAE) | 3.14 | 3.25 | |
| Ganador acertado (pick = mayor prob.) | 0.62 | 0.60 (pole) | |
| Podio acertado (de 3) | 2.00 | 2.03 | |

Con 112 carreras (2022+) la ventaja es más clara: RPS 0.1139 vs 0.1227, log-loss del ganador 1.328 vs 1.526 (−0.20, significativo), Spearman 0.675 vs 0.630.

**Calibración (ECE, 2024+):** ganar 0.015 · podio 0.025 · puntos 0.018 · abandono 0.011. Las probabilidades están bien calibradas.

**Dónde NO supera a la grilla:** en las primeras 6 carreras de un reglamento nuevo (2026 en este corte; 2022 y 2026 en el corte 2022+) el modelo
no mejora a la grilla (Spearman 0.555 vs 0.571, RPS 0.1395 vs 0.1361). Son 6–12 carreras: no se puede concluir más que "no hay evidencia de mejora".
El log-loss del ganador tiene mucho ruido (un evento por carrera): conviene decidir con RPS y log-loss de posición.

### Antes de clasificar (grilla desconocida, `prequali.py`, `eval_prequali.py`)
Un ranker propio predice la clasificación, se muestrean 300 grillas posibles y la carrera se promedia sobre ellas (68 carreras, 2024+):

| | grilla estimada por heurística | modelo + muestreo |
|---|---|---|
| Spearman del orden de clasificación | 0.721 | 0.746 |
| Spearman de carrera | 0.613 | 0.621 |
| Log-loss del ganador | 2.34 | 1.96 |
| Brier de podio | 0.098 | 0.085 |

Sigue siendo mucho menos preciso que con la clasificación real (ganador acertado ~0.25–0.30). La mejora está en las probabilidades (menos sobreconfianza).

### Revisión crítica y correcciones
Una revisión externa (2026-11-11) encontró y se corrigió: (1) los pilotos doblados (`Lapped`, desde 2023) se contaban como abandono: la tasa de DNF "vista" era
31–49% contra 10–18% real; ahora la etiqueta usa `position_text`; (2) el clasificador de abandono XGBoost no discriminaba mejor que la posición de largada
(AUC 0.55 vs 0.61) y se reemplazó por una logística sobre log(grilla); (3) el ranker se entrena solo con pilotos clasificados; (4) el tau de clasificación
usa los primeros 3 puestos como el de carrera; (5) se agregó el baseline `grid_pl`, RPS, ECE y tests pareados.

**Limitaciones que siguen vigentes**
- `CAR_WINDOW`, `prior_races`, `GRID_BLEND_BETA` y `TAU_TOP_K` se eligieron mirando el backtest 2024+. La verificación `sweep_honest.py` se hizo con la etiqueta de abandono
  rota (resultados no válidos) y no se repitió. En 2022–23 el término de grilla (β=1) rindió peor que β=0 en log-loss del ganador: el valor óptimo cambia con el período.
  Los hiperparámetros de XGBoost y las ventanas están en una meseta (|ΔRPS| < 0.001): no hay ganancia tuneándolos.
- Un solo cambio de reglamento comparable (2022): los priors de reglamento no son validables estadísticamente.
- Sin usar: tiempos de vuelta (ritmo del auto), práctica libre y clima (FastF1). Son las mejoras con mayor potencial, no medidas.
- `--grid` no recibe `quali_pos` ni `sprint_pos` (`grid_pen` queda en 0); no afecta a las carreras sin sprint.
- Las predicciones de Las Vegas / Qatar / Abu Dhabi son la única validación realmente fuera de muestra.

Experimentos: `sweep_regime.py`, `sweep_tau.py`, `sweep_rank.py`, `sweep_gridblend.py`, `sweep_honest.py` (ver limitación arriba), `backtest_pre_quali.py`, `eval_prequali.py`.
