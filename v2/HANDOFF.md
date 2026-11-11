# HANDOFF – F1 Race Predictor v2 (para continuar con Claude Code)

Dueño: Mateo (habla español). Preferencias: archivos completos (no snippets), implementación concisa,
logs en procesos largos, "hacelo como mejor te parezca" en detalles de implementación.
Carpeta del proyecto: `M:\projects\F1_ML_Race_Result_Predictor` (v1 en la raíz, **v2 en `v2/`**). No tocar v1.
Fecha de este handoff: 2026-11-10 (temporada 2026 casi terminada).

## 1. Qué es y por qué existe v2
La v1 (hecha hace ~1 año, FastF1 + XGBoost, datos 2021–2025) tenía problemas de validez:
- Desfase train/serve: al predecir se ignoraba la temporada en curso (`año < año_actual`), en train no.
- Split "temporal" ordenado alfabéticamente por nombre de carrera.
- El circuito no entraba al modelo (`feature_cols` no lo incluía); `podios_ultimas_5` tampoco.
- Rookies con penalizaciones a mano; probabilidades infladas por `scale_pos_weight`; accuracy como métrica.
- Se salteaban los fines de semana sprint.
- 2026 trae **cambio de reglamento + motores nuevos** → el orden de los autos se reinicia.

v2 reescribe todo en `v2/` (ver `README.md` para el resumen de usuario).

## 2. Estado actual (qué está hecho y qué NO)
Hecho y **probado solo con datos sintéticos** (`python tests/test_pipeline.py` pasa: sin fuga de datos futuros,
features de train == inferencia, probabilidades suman 1, flag de reglamento, modelo entrena y predice;
además train/evaluate/predict corrieron end-to-end sobre datos simulados). Entorno de prueba: Python 3.13,
pandas 3.0.5, xgboost 3.4.1, scikit-learn 1.9.

**NO verificado (primer trabajo de Claude Code):**
- `collect_data.py` contra la API real de Jolpica-F1 (nunca se ejecutó; el sandbox no tenía red a F1).
- Entrenamiento/backtest con datos reales → **no hay métricas reales todavía**.
- `front.py` (solo compila; no se levantó Streamlit).
- Versiones del entorno de Mateo (`.venv` de v1 en la raíz; pandas/xgboost pueden diferir).

## 3. Arquitectura (archivos de `v2/`)
| Archivo | Rol |
|---|---|
| `config.py` | Años, `REGIME_CHANGE_SEASONS={2022,2026}`, priors, ventanas, linaje de equipos (`TEAM_LINEAGE`), `power_unit()` por temporada |
| `collect_data.py` | Jolpica (`https://api.jolpi.ca/ergast/f1`): results + qualifying + sprint paginados → `data/results.csv`, `data/schedule.csv`; cachea años cerrados en `data/raw/` |
| `features.py` | `prepare_results`, `FeatureBuilder.race_features` (**único camino** para train e inferencia), `build_training_table`, `build_inference_features` |
| `model.py` | `XGBRanker` (rank:pairwise, grupos por carrera, monotone constraints) + `XGBClassifier` DNF + tau Plackett-Luce (fit en últimas `CAL_RACES`) + `simulate`/`predict_race` (Gumbel, 20k sims) |
| `train.py` | features → `data/features.csv`, modelo final → `models/model.joblib` |
| `evaluate.py` | Backtest walk-forward (reentrena cada 6 carreras), baseline "grilla", segmentos `regime_early`/`regular`, escribe `data/backtest_races.csv` y `models/backtest_metrics.json` |
| `predict.py` | CLI: próxima carrera (grilla estimada o `--grid CODES`), `--backtest SEASON ROUND`; funciones reutilizadas por el front |
| `front.py` | Streamlit ES/EN: carrera → grilla editable → tabla de probabilidades |
| `tests/test_pipeline.py` | Tests sintéticos |

Decisiones de diseño clave (no revertir sin motivo):
1. **Causalidad estricta**: features de la carrera `t` usan solo `t' < t`; el test de leakage lo garantiza.
2. **Piloto vs. auto separados**: piloto = delta contra compañero de equipo (race/quali), portable entre equipos;
   auto = rendimiento del equipo (rel. en [-1,1]) en ventana de últimas 5 carreras de la temporada.
3. **Prior del auto con consciencia de reglamento**: `blend = (n·cur + w·prior_shrunk)/(n+w)`, `w = prior_races·0.85^n`;
   en años de reglamento nuevo `shrink=0.75, prior_races=1.5`; estable `0.15 / 4.0`. Equipo sin historial → prior 0.
4. **Pooling por motor** (`pu_car_q/f`): promedio del auto blended de equipos con el mismo PU esa temporada.
5. Rookies/pilotos sin historial: shrinkage a 0 (`n/(n+k)`), sin parches manuales.
6. Probabilidades por **simulación Plackett-Luce** (ganar suma 100%) + DNF aparte (manda al fondo).
7. Sin identidad (piloto/equipo) como feature → generaliza a equipos nuevos (Cadillac, Audi).

## 4. Plan para Claude Code (en orden, con criterio de aceptación)

### Paso 0 – Entorno
- Crear venv (o reusar `.venv` de la raíz), `pip install -r v2/requirements.txt`. Correr `python v2/tests/test_pipeline.py`
  (nota: NO usar `python -I` con los scripts del proyecto; `-I` saca el directorio del script de `sys.path`).
- Agregar `.gitignore` en la raíz (si no existe): `.venv/`, `f1_cache/`, `__pycache__/`, `v2/data/`, `v2/models/`.
- ✅ Aceptación: tests pasan en el entorno de Mateo.

### Paso 1 – Datos reales (`collect_data.py`)
- `cd v2 && python collect_data.py`. Es el primer contacto con Jolpica: **validar supuestos del esquema**:
  - paginación (`limit=100`, `offset`, campo `MRData.total`), carreras partidas entre páginas (se juntan por `round`);
  - `Results[].grid` (0 = pit lane), `position`, `status` ("Finished", "+1 Lap", resto = DNF), `Driver.code`;
  - `QualifyingResults[].position`, `SprintResults[].position`; rate-limit 429 (hay backoff; Jolpica ~4 req/s, 500/h).
- Revisar la salida: carreras por año (≈ 17–24), 20–22 pilotos por carrera (11 equipos en 2026 → ~22), última carrera cargada
  = la última disputada de 2026; % de `quali_pos` nulo (esperable bajo; 2021 y sprint-shootouts pueden diferir); filas con `position` nula.
- **Revisar el warning de `constructorId` desconocidos** y completar `config.TEAM_LINEAGE` (ej. ids de Audi, Cadillac,
  "RB"/"Racing Bulls", Alpine/Renault) y `config.power_unit()` para 2026 (hoy cargado de memoria: Mercedes→Mercedes/McLaren/Williams/Alpine;
  Ferrari→Ferrari/Haas/Cadillac; Honda→Aston Martin; RBPT-Ford→Red Bull/RB; Audi→Audi). Contrastar con fuente oficial.
- ✅ Aceptación: `data/results.csv` y `schedule.csv` coherentes, sin ids sin mapear (o mapeados a propósito).

### Paso 2 – Entrenar y mirar features (`train.py`)
- `python train.py`. Verificar: nº de carreras de la tabla (≈ total − 8 de warmup), NaN solo en `sprint_pos`
  (y `pu_*` no deberían ser NaN), distribución de `car_q`/`drv_*` razonable, `tau` finito (ni en el borde ±e⁴).
- Revisar importancias: grilla/quali arriba es esperable; si `regime_new`/`team_races_season` dominan, sospechar sobreajuste.
- ✅ Aceptación: modelo guardado, sin errores/warnings graves (ojo versiones de pandas/xgboost).

### Paso 3 – Backtest (`evaluate.py`) y criterio de calidad
- `python evaluate.py` (desde 2024) y luego `python evaluate.py --from-season 2022` (valida el arranque del reglamento anterior).
- Comparar contra baseline "grilla" en el mismo output. Metas razonables (a confirmar con datos reales):
  - Spearman y top3_overlap **≥ baseline de la grilla** (el modelo debe agregar sobre la grilla, no empeorarla);
  - `winner_logloss` claramente < ln(20)≈3.0; `podium_brier` < `podium_brier_base`;
  - `regime_early` no debe ser mucho peor que `regular` (si lo es, el prior del reglamento está mal calibrado).
- Si el modelo no supera la grilla: probar (a) más regularización / `n_estimators` menor, (b) quitar features ruidosas
  (`drv_circuit_*`, `pu_*`), (c) `RECENCY_HALF_LIFE_SEASONS`, (d) monotone constraints, (e) calibración isotónica de probabilidades.
- ✅ Aceptación: tabla de métricas documentada en `README.md` (sección Resultados) con fecha y datos usados.

### Paso 4 – Ajuste del cambio de reglamento (tema central para Mateo)
- Barrido sobre `PRIOR_NEW_REGIME` (`shrink` ∈ {0.5, 0.75, 0.9}, `prior_races` ∈ {0.5, 1.5, 3}) y `CAR_WINDOW` ∈ {3,5,8},
  evaluando **solo** el segmento `regime_early` en 2022 (train hasta 2021) y luego mirar 2026 (primeras 6 carreras, ya con datos).
  Ojo: con una sola transición histórica (2022) hay riesgo de sobreajustar; preferir parámetros robustos sobre el óptimo exacto.
- Idea opcional: agregar `car_trend` (pendiente de mejora del auto dentro de la temporada) porque en reglamento nuevo los equipos desarrollan rápido.
- ✅ Aceptación: valores elegidos en `config.py` con comentario del experimento que los justifica.

### Paso 5 – Predicción y front
- `python predict.py` (próxima carrera, grilla estimada), `python predict.py --backtest 2026 <ronda>` (sanity),
  luego `streamlit run front.py`. Revisar: el lineup por defecto (`default_lineup`) es el de la última carrera → si hay suplentes o
  cambios, hay que poder editarlo (hoy solo se edita el orden de grilla; **mejora pendiente**: permitir reemplazar pilotos/equipos).
- Verificar que `next_race` funcione cuando termina la temporada (hoy hace `SystemExit` si no hay pendientes → mostrar mensaje amigable en el front).
- ✅ Aceptación: el front levanta, predice y muestra métricas del backtest.

### Paso 6 – Limpieza y entrega
- Actualizar `README.md` (v2 + resultados reales), mover v1 a `legacy/` **solo si Mateo lo aprueba**, commit.
- (Opcional) Deploy en Streamlit Community Cloud: ya hay un link placeholder en el README de v1; commitear un `models/model.joblib` chico
  y `data/results.csv`, o regenerar en CI.

## 5. Riesgos / cosas a vigilar
- **Esquema de Jolpica**: `collect_data.py` se escribió de memoria sin poder probarlo. Si falla, corregir ahí primero.
- **Linaje y motores 2026** cargados de memoria: contrastar. Un equipo mal mapeado pierde su prior (queda en la media) pero no rompe nada.
- **Pocos datos de reglamento nuevo**: P10–P90 anchos son correctos; no "arreglar" achicándolos.
- Fechas/horas: `schedule.csv` usa `date` (sin hora); `next_race` asume que las carreras sin resultados cargados son las pendientes.
- Pickles de XGBoost dependen de versión: `model.joblib` guarda `xgboost` version; reentrenar si cambia.
- El modo `--backtest` de `predict.py` usa el modelo final (entrenado con esa carrera); para métricas limpias usar `evaluate.py`.

## 6. Mejoras futuras (v3, por valor esperado)
1. **Ritmo de práctica libre** (FastF1, FP2 long-runs / gap a la mejor vuelta) → mejora mucho la predicción pre-clasificación y capta el desarrollo del auto en reglamento nuevo.
2. Modelo de **clasificación** (grilla) propio en vez de la heurística `estimate_grid`.
3. Clima (lluvia) y safety cars históricos por circuito; features de estrategia (stints) desde FastF1.
4. Modelo bayesiano/Elo de "auto + piloto" con varianza explícita para reglamentos nuevos.
5. Calibración isotónica de probabilidades con las predicciones del backtest.
6. Predicción de campeonato (simulación Monte Carlo del resto de la temporada).

## 7. Prompt de arranque sugerido para Claude Code
> Leé `v2/HANDOFF.md` y `v2/README.md`. Trabajá en `v2/` sin tocar los archivos de la raíz (v1). Seguí el plan del handoff desde el
> Paso 0: preparar entorno, correr los tests, ejecutar `collect_data.py` contra Jolpica real y arreglar lo que falle, entrenar, correr
> `evaluate.py` y mostrame las métricas contra el baseline de la grilla antes de seguir con el ajuste del cambio de reglamento (Paso 4).
> Archivos completos, logs en procesos largos, respuestas en español y concisas.
