"""Configuración central del pipeline v2.

Todo lo "ajustable" (años, cambios de reglamento, linaje de equipos, motores)
vive acá para no tocar el resto del código.
"""
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = BASE_DIR / "models"
for _d in (DATA_DIR, RAW_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = DATA_DIR / "results.csv"
SCHEDULE_CSV = DATA_DIR / "schedule.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_JSON = MODEL_DIR / "backtest_metrics.json"

# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
START_SEASON = 2018
END_SEASON = date.today().year
JOLPICA_URL = "https://api.jolpi.ca/ergast/f1"

# --------------------------------------------------------------------------
# Cambios de reglamento
# --------------------------------------------------------------------------
# Temporadas donde el rendimiento de los autos se "reinicia" (nuevo reglamento
# técnico). 2022: efecto suelo. 2026: nuevos motores + nuevo chasis/aero.
REGIME_CHANGE_SEASONS = {2022, 2026}

# Cuánto se "olvida" del año anterior al arrancar una temporada:
#   shrink = fracción que se tira hacia la media del campo
#   prior_races = peso del prior en "carreras equivalentes"
PRIOR_STABLE = {"shrink": 0.15, "prior_races": 4.0}
# Barrido (sweep_regime.py, primeras 6 carreras de 2022 y 2026): el efecto es chico (Spearman 0.50-0.54,
# ruido alto con 12 carreras); (0.75, 3.0) con CAR_WINDOW=8 fue el mejor y sus vecinos tambien.
PRIOR_NEW_REGIME = {"shrink": 0.75, "prior_races": 3.0}

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
CAR_WINDOW = 8              # carreras recientes de la temporada para medir el auto
CAR_HALF_LIFE = 3.0         # half-life (en carreras) dentro de esa ventana
DRIVER_WINDOW = 40          # carreras previas consideradas para el piloto
DRIVER_HALF_LIFE = 15.0
DNF_PRIOR_RATE, DNF_PRIOR_N = 0.12, 6.0

# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
RECENCY_HALF_LIFE_SEASONS = 2.5   # peso de cada carrera al entrenar
MIN_PRIOR_RACES_TO_TRAIN = 3      # descartar primeras carreras sin historial
CAL_RACES = 10                    # carreras reservadas para calibrar la temperatura
TAU_TOP_K = 3      # tau se ajusta con los primeros K puestos (None = todo el campo). Barrido desde 2024: K=None logloss ganador 1.90, K=3 1.45 (el orden del frente es mas predecible que el de la zona media)
N_SIMS = 20000
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Linaje de equipos: constructorId (Jolpica) -> nombre canónico
# Equipos sin entrada acá se tratan como equipos nuevos (sin prior).
# --------------------------------------------------------------------------
TEAM_LINEAGE = {
    "renault": "alpine", "alpine": "alpine",
    "force_india": "aston_martin", "racing_point": "aston_martin", "aston_martin": "aston_martin",
    "toro_rosso": "rb", "alphatauri": "rb", "rb": "rb",
    "alfa": "sauber", "sauber": "sauber", "audi": "sauber",
    "red_bull": "red_bull", "mercedes": "mercedes", "ferrari": "ferrari",
    "mclaren": "mclaren", "haas": "haas", "williams": "williams",
    "cadillac": "cadillac",
}

# Motor (power unit) por equipo canónico y temporada. Sirve para "pooling":
# si un equipo cliente rinde, informa sobre el motor de su proveedor.
# EDITAR si cambia algo (2026 está cargado con la parrilla anunciada).
def power_unit(team: str, season: int):
    if season >= 2026:
        return {"mercedes": "mercedes", "mclaren": "mercedes", "williams": "mercedes",
                "alpine": "mercedes", "ferrari": "ferrari", "haas": "ferrari",
                "cadillac": "ferrari", "aston_martin": "honda", "red_bull": "rbpt_ford",
                "rb": "rbpt_ford", "sauber": "audi"}.get(team)
    if season >= 2021:
        return {"mercedes": "mercedes", "aston_martin": "mercedes", "williams": "mercedes",
                "mclaren": "mercedes", "ferrari": "ferrari", "haas": "ferrari",
                "sauber": "ferrari", "alpine": "renault", "red_bull": "honda",
                "rb": "honda"}.get(team)
    if season == 2019 or season == 2020:
        return {"mercedes": "mercedes", "aston_martin": "mercedes", "williams": "mercedes",
                "ferrari": "ferrari", "haas": "ferrari", "sauber": "ferrari",
                "alpine": "renault", "mclaren": "renault", "red_bull": "honda",
                "rb": "honda"}.get(team)
    return {"mercedes": "mercedes", "aston_martin": "mercedes", "williams": "mercedes",
            "ferrari": "ferrari", "haas": "ferrari", "sauber": "ferrari",
            "alpine": "renault", "mclaren": "renault", "red_bull": "renault",
            "rb": "honda"}.get(team)
GRID_BLEND_BETA = 1.0   # puntaje += -beta*log(grid_adj). sweep_gridblend.py (desde 2024): beta 0 -> acierto ganador 0.44, logloss 1.45;
                        # meseta en beta 0.8-1.6 -> 0.59-0.60 y 1.29
