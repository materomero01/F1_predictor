"""Front Streamlit v2.   streamlit run front.py

Requiere haber corrido antes: collect_data.py -> train.py
"""
import json

import joblib
import pandas as pd
import streamlit as st

import config as C
from predict import default_lineup, estimate_grid, next_race, run, run_prequali

st.set_page_config(page_title="F1 Predictor v2", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(C.MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(C.RESULTS_CSV), pd.read_csv(C.SCHEDULE_CSV)


if not (C.MODEL_PATH.exists() and C.RESULTS_CSV.exists()):
    st.error("Faltan datos o modelo. Corré `python collect_data.py` y luego `python train.py`.")
    st.stop()

bundle = load_bundle()
raw, sched = load_data()

col_t, col_l = st.columns([0.85, 0.15])
col_t.title("🏁 F1 Race Predictor")
EN = col_l.radio("Idioma", ["ES", "EN"], horizontal=True, label_visibility="collapsed") == "EN"
T = lambda es, en: en if EN else es  # noqa: E731

st.caption(T(f"Modelo entrenado hasta: {bundle['trained_through']} · {bundle['n_races']} carreras",
             f"Model trained through: {bundle['trained_through']} · {bundle['n_races']} races"))
if C.METRICS_JSON.exists():
    m = json.loads(C.METRICS_JSON.read_text())["overall"]
    st.caption(T(f"Backtest walk-forward: Spearman {m['spearman']:.2f} (grilla: {m['spearman_grid']:.2f}) · "
                 f"podio ∩ {m['top3_overlap']:.2f}/3 (grilla: {m['top3_overlap_grid']:.2f}/3)",
                 f"Walk-forward backtest: Spearman {m['spearman']:.2f} (grid: {m['spearman_grid']:.2f}) · "
                 f"podium overlap {m['top3_overlap']:.2f}/3 (grid: {m['top3_overlap_grid']:.2f}/3)"))

# --- 1. carrera ---
try:
    nr = next_race(raw, sched)
except SystemExit:
    st.info(T("No hay carreras pendientes en el calendario cargado (temporada terminada). "
              "Cuando salga el calendario nuevo, corré `python collect_data.py`.",
              "No pending races in the loaded calendar (season is over). "
              "Once the new calendar is out, run `python collect_data.py`."))
    st.stop()
st.subheader(T("1️⃣ Carrera", "1️⃣ Race"))
races = sched[(sched["season"] == nr["season"]) & (sched["round"] >= nr["round"])]
label = lambda r: f"R{r.round} · {r.race_name} ({r.date})"  # noqa: E731
choice = st.selectbox(T("Carrera a predecir", "Race to predict"),
                      options=list(races.itertuples()), format_func=label)

# --- 2. grilla ---
st.subheader(T("2️⃣ Grilla de largada", "2️⃣ Starting grid"))
lineup = default_lineup(raw)
mode = st.radio(T("Origen de la grilla", "Grid source"),
                [T("Estimada (antes de clasificar)", "Estimated (pre-qualifying)"),
                 T("Manual (resultado real de clasificación)", "Manual (actual qualifying)")], horizontal=True)

key = f"grid_{choice.season}_{choice.round}_{mode}"
if key not in st.session_state:
    if mode.startswith(("Estim")):
        g = estimate_grid(raw, int(choice.season), int(choice.round), choice.circuit_id, lineup)
    else:
        g = lineup.assign(grid=range(1, len(lineup) + 1))
    st.session_state[key] = g.sort_values("grid").reset_index(drop=True)[
        ["grid", "code", "driver_name", "constructor_name", "driver_id", "constructor_id"]]

known = raw.sort_values(["season", "round"]).drop_duplicates("code", keep="last").set_index("code")
teams = (raw[raw["season"] == raw["season"].max()].drop_duplicates("constructor_id")
         .set_index("constructor_name")["constructor_id"].to_dict())
edited = st.data_editor(
    st.session_state[key], hide_index=True, width="stretch", key=key + "_ed",
    column_order=["grid", "code", "constructor_name"],
    column_config={
        "grid": st.column_config.NumberColumn(T("Largada", "Grid"), min_value=1, max_value=len(lineup), step=1),
        "code": st.column_config.TextColumn(T("Piloto (código)", "Driver (code)"), max_chars=3),
        "constructor_name": st.column_config.SelectboxColumn(T("Equipo", "Team"), options=list(teams), required=True)})
st.caption(T("Editá Largada (1 = pole; cada número una sola vez; penalizados / largada desde boxes ponen el puesto real), "
             "y si hay suplentes o cambios, el código (3 letras) y el equipo. Un código sin historial se trata como piloto nuevo.",
             "Edit Grid (1 = pole; each number once; for penalties / pit-lane starts enter the actual slot), and for "
             "substitutes or swaps, the 3-letter code and team. A code with no history is treated as a new driver."))


def resolve(df: pd.DataFrame) -> pd.DataFrame:
    """Código/equipo editados -> driver_id, constructor_id, nombre."""
    df = df.assign(code=df["code"].astype(str).str.strip().str.upper())
    ids = df["code"].map(known["driver_id"]).fillna(df["code"].str.lower())
    names = df["code"].map(known["driver_name"]).fillna(df["code"])
    return df.assign(driver_id=ids, driver_name=names, constructor_id=df["constructor_name"].map(teams))


# --- 3. predicción ---
st.subheader(T("3️⃣ Predicción", "3️⃣ Prediction"))
if st.button(T("🔮 Generar predicción", "🔮 Generate prediction"), type="primary"):
    pre = mode.startswith("Estim")
    if not pre and sorted(edited["grid"].astype(int)) != list(range(1, len(edited) + 1)):
        st.error(T("La grilla debe ser una permutación de 1..N sin repetidos.",
                   "The grid must be a permutation of 1..N with no repeats."))
        st.stop()
    if edited["code"].astype(str).str.strip().eq("").any() or edited["code"].astype(str).str.upper().duplicated().any():
        st.error(T("Cada piloto debe tener un código único.", "Each driver needs a unique code."))
        st.stop()
    entries = resolve(edited).assign(grid=edited["grid"].astype(float), is_sprint=int(choice.is_sprint_weekend))
    if pre:   # sin clasificación: se promedia sobre grillas posibles, la columna Largada se ignora
        out = run_prequali(raw, bundle, int(choice.season), int(choice.round), choice.circuit_id,
                           entries[["driver_id", "code", "driver_name", "constructor_id", "constructor_name"]])
        out = out.assign(grid=out["grid_esperada"].round(1))
    else:
        out = run(raw, bundle, int(choice.season), int(choice.round), choice.circuit_id, entries)
    out = out.assign(**{c: (out[c] * 100).round(1) for c in ("prob_ganar", "prob_podio", "prob_top10", "prob_dnf")})
    out["pos_esperada"] = out["pos_esperada"].round(1)
    out.index = out.index + 1
    names = {"code": T("Piloto", "Driver"), "constructor_name": T("Equipo", "Team"), "grid": T("Largada (esp.)" if pre else "Largada", "Grid (exp.)" if pre else "Grid"),
             "pos_esperada": T("Pos. esperada", "Exp. pos"), "pos_p10": "P10", "pos_p90": "P90",
             "prob_ganar": T("Ganar %", "Win %"), "prob_podio": T("Podio %", "Podium %"),
             "prob_top10": T("Puntos %", "Points %"), "prob_dnf": T("Abandono %", "DNF %")}
    st.dataframe(out[list(names)].rename(columns=names), width="stretch")
    if pre:
        st.info(T("Predicción PRE-clasificación: promedia sobre muchas grillas posibles (la columna Largada del paso 2 no se usa). "
                  "Es bastante menos precisa que con la clasificación real: cargala en modo Manual cuando la tengas.",
                  "PRE-qualifying prediction: averaged over many possible grids (the Grid column in step 2 is not used). "
                  "Much less precise than with the real qualifying: enter it in Manual mode once available."))
    st.caption(T("P10/P90: rango de posiciones en el 80% de las simulaciones. "
                 "Las probabilidades de ganar suman 100% (simulación Plackett-Luce).",
                 "P10/P90: finishing-position range in 80% of simulations. "
                 "Win probabilities sum to 100% (Plackett-Luce simulation)."))
