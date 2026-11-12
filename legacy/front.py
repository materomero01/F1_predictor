import streamlit as st
from get_current_drivers import obtener_pilotos_actuales
from race_predictor import PredictorF1
import fastf1
import pandas as pd
from datetime import date
import random
import numpy as np
import os

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------
st.set_page_config(page_title="F1 Predictor", layout="wide")

# Crear carpeta de cache si no existe
cache_dir = os.path.join(os.path.dirname(__file__), "f1_cache")
os.makedirs(cache_dir, exist_ok=True)

# Inicializar cache
fastf1.Cache.enable_cache(cache_dir)

# ------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------

@st.cache_data(ttl=60*60*24)
def get_circuits(year=None):
    if year is None:
        year = date.today().year
    try:
        schedule = fastf1.get_event_schedule(year)
        races = schedule[schedule["EventFormat"] != "Test"]
        circuits = races["Location"].dropna().unique().tolist()
        if not circuits:
            circuits = races["EventName"].dropna().unique().tolist()
        return circuits
    except:
        return []

@st.cache_data(ttl=60*60*24)
def get_drivers():
    return obtener_pilotos_actuales()

# ------------------------------------------------------------
# TRADUCCIONES
# ------------------------------------------------------------

lang_col1, lang_col2 = st.columns([0.85, 0.15])
with lang_col1:
    st.markdown("<h1 style='margin-bottom:0'>🏁 F1 Race Predictor</h1>", unsafe_allow_html=True)
with lang_col2:
    LANG = st.radio("", options=["ES", "EN"], horizontal=True, label_visibility="collapsed")

L = LANG == "EN"

TXT = {
    "intro": {
        False: "Este modelo de Machine Learning predice los resultados de una carrera de F1 basándose en la parrilla de salida, circuito y datos históricos.",
        True: "This Machine Learning model predicts F1 race results based on the starting grid, circuit, and historical data."
    },
    "load_drivers": {False: "1️⃣ Cargar pilotos", True: "1️⃣ Load drivers"},
    "load_desc": {
        False: "Los pilotos se cargan automáticamente desde la última carrera disponible. Ahora solo arrastralos para ordenarlos según la posición de largada.",
        True: "Drivers are automatically fetched. Just drag rows to set the starting grid order."
    },
    "select_track": {False: "2️⃣ Seleccionar circuito", True: "2️⃣ Select circuit"},
    "track_desc": {
        False: "Elegí el circuito para realizar la predicción.",
        True: "Choose the circuit to run the prediction."
    },
    "generate": {False: "3️⃣ Generar predicción", True: "3️⃣ Generate prediction"},
    "btn_predict": {False: "🔮 Generar Predicción", True: "🔮 Generate Prediction"},
    "err_positions": {
        False: "Debés ordenar todos los pilotos antes de continuar.",
        True: "Please order all drivers before continuing."
    },
    "table_headers": {
        False: {"piloto": "Piloto", "nombre": "Nombre", "equipo": "Equipo"},
        True: {"piloto": "Driver", "nombre": "Name", "equipo": "Team"}
    },
    "table_result_labels": {
        False: {
            "piloto": "Piloto", "equipo": "Equipo",
            "posicion_parrilla": "Largada", "posicion_predicha": "Predicción",
            "prob_ganar_%": "Prob. Ganar (%)", "prob_podio_%": "Prob. Podio (%)"
        },
        True: {
            "piloto": "Driver", "equipo": "Team",
            "posicion_parrilla": "Grid", "posicion_predicha": "Prediction",
            "prob_ganar_%": "Win Prob (%)", "prob_podio_%": "Podium Prob (%)"
        }
    }
}

# ------------------------------------------------------------
# FRONTEND
# ------------------------------------------------------------

# --- Paso 1: cargar pilotos ---
st.subheader(TXT["load_drivers"][L])
st.write(TXT["load_desc"][L])

df_pilotos = get_drivers()
if df_pilotos.empty:
    st.error("❌ No se pudieron cargar pilotos automáticamente.")
    st.stop()

headers = TXT["table_headers"][L]

# Inicializar session_state con una columna auxiliar ORDEN
if "grid_data" not in st.session_state:
    base_df = df_pilotos.copy()
    base_df["ORDEN"] = list(range(1, len(base_df) + 1))  # Orden inicial
    st.session_state.grid_data = base_df

# --- Editor con drag & drop ---
df_visual = st.session_state.grid_data.rename(columns=headers)

edited_df = st.data_editor(
    df_visual,
    key="parrilla_editor",
    width="stretch",
    hide_index=True,
    column_order=["ORDEN"] + list(headers.values()),
    disabled=list(headers.values()),  # sólo ORDEN es editable
)

# Al mover filas, Streamlit reasigna ORDEN. Actualizamos la tabla real:
df_back = edited_df.rename(columns={v: k for k, v in headers.items()})
df_back = df_back.sort_values("ORDEN")  # orden del usuario
df_back["posicion_parrilla"] = range(1, len(df_back) + 1)
st.session_state.grid_data = df_back

# --- Paso 2: seleccionar circuito ---
st.subheader(TXT["select_track"][L])
st.write(TXT["track_desc"][L])
circuitos = get_circuits()
circuito = st.selectbox("🌍 Circuito / Circuit", options=circuitos)

# --- Paso 3: predicción ---
st.subheader(TXT["generate"][L])

if st.button(TXT["btn_predict"][L]):
    df_actual = st.session_state.grid_data.copy()

    parrilla_input = [
        {
            "piloto": row["piloto"],
            "equipo": row["equipo"],
            "posicion_parrilla": int(row["posicion_parrilla"])
        }
        for _, row in df_actual.iterrows()
    ]

    predictor = PredictorF1()
    resultado = predictor.predecir_carrera(
        parrilla=parrilla_input,
        circuito=circuito,
        año=date.today().year,
    )

    ocultar = {"es_rookie", "posicion_real", "experiencia", "forma_reciente"}
    visible_cols = [c for c in resultado.columns if c not in ocultar]
    resultado_visual = resultado[visible_cols].rename(columns=TXT["table_result_labels"][L])
    st.dataframe(resultado_visual, width="stretch")
