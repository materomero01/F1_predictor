import streamlit as st
from get_current_drivers import obtener_pilotos_actuales
from race_predictor import PredictorF1
import fastf1
import pandas as pd
from datetime import date
import random
import numpy as np

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
    """Devuelve la lista de circuitos disponibles en FastF1."""
    if year is None:
        year = date.today().year
    try:
        schedule = fastf1.get_event_schedule(year)
        races = schedule[schedule["EventFormat"] != "Test"]
        circuits = races["Location"].dropna().unique().tolist()
        if not circuits:
            circuits = races["EventName"].dropna().unique().tolist()
        return circuits
    except Exception as e:
        print(f"Error al obtener circuitos: {e}")
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
        False: "Los pilotos se cargan automáticamente desde la última carrera disponible. Solo debés ingresar la posición de salida de cada uno.",
        True: "Drivers are automatically fetched from the most recent race. You only need to fill their starting positions."
    },
    "select_track": {False: "2️⃣ Seleccionar circuito", True: "2️⃣ Select circuit"},
    "track_desc": {
        False: "Elegí el circuito para realizar la predicción.",
        True: "Choose the circuit to run the prediction."
    },
    "generate": {False: "3️⃣ Generar predicción", True: "3️⃣ Generate prediction"},
    "gen_desc": {
        False: "Completá todas las posiciones antes de generar la predicción.",
        True: "Fill in all starting positions before generating the prediction."
    },
    "btn_predict": {False: "🔮 Generar Predicción", True: "🔮 Generate Prediction"},
    "err_positions": {
        False: "Completá todas las posiciones antes de continuar.",
        True: "Please fill all positions before continuing."
    },
    # Etiquetas de columnas visibles (solo visual, no cambian internamente)
    "table_headers": {
        False: {"piloto": "Piloto", "nombre": "Nombre", "equipo": "Equipo", "posicion_parrilla": "Posición de Largada"},
        True: {"piloto": "Driver", "nombre": "Name", "equipo": "Team", "posicion_parrilla": "Starting Grid"}
    },
    "table_result_labels": {
        False: {"piloto": "Piloto", "equipo": "Equipo", "posicion_parrilla": "Largada", "posicion_predicha": "Predicción", "prob_ganar_%": "Prob. Ganar (%)", "prob_podio_%": "Prob. Podio (%)"},
        True: {"piloto": "Driver", "equipo": "Team", "posicion_parrilla": "Grid", "posicion_predicha": "Prediction", "prob_ganar_%": "Win Prob (%)", "prob_podio_%": "Podium Prob (%)"}
    }
}

# ------------------------------------------------------------
# FRONTEND
# ------------------------------------------------------------

# --- Paso 1: cargar pilotos ---
st.subheader(TXT["load_drivers"][L])
st.write(TXT["load_desc"][L])

# Obtener pilotos del año actual o último con datos
df_pilotos = get_drivers()
if df_pilotos.empty:
    st.error("❌ No se pudieron cargar pilotos automáticamente.")
    st.stop()

# Columnas traducidas visualmente
headers = TXT["table_headers"][L]

# Inicializar DataFrame persistente en session_state
if "grid_data" not in st.session_state:
    base_df = df_pilotos.copy()
    base_df["posicion_parrilla"] = [""] * len(base_df)
    st.session_state.grid_data = base_df

# --- Botones arriba del editor ---
# --- Botones de acción alineados a la derecha ---
spacer_col, colA, colB = st.columns([0.6, 0.2, 0.2])
with colA:
    random_btn = st.button(
        "🎲 Generar Aleatorio" if not L else "🎲 Random Grid",
        use_container_width=True
    )

with colB:
    st.button(
        "🏎️ Simular Qualy (Próximamente)" if not L else "🏎️ Simulate Qualy (Coming Soon)",
        disabled=True,
        use_container_width=True
    )
# --- Generar posiciones aleatorias si se presionó el botón ---
if random_btn:
    df_pilotos = st.session_state.grid_data.copy()
    if "experiencia" in df_pilotos.columns:
        exp = df_pilotos["experiencia"].fillna(df_pilotos["experiencia"].mean())
        prob = np.exp(exp / exp.max())
        prob /= prob.sum()
        posiciones = np.random.choice(
            range(1, len(df_pilotos) + 1),
            size=len(df_pilotos),
            replace=False,
            p=prob
        )
    else:
        posiciones = random.sample(range(1, len(df_pilotos) + 1), len(df_pilotos))

    df_pilotos["posicion_parrilla"] = posiciones
    st.session_state.grid_data = df_pilotos

# --- Editor de la parrilla ---
df_visual = st.session_state.grid_data.rename(columns=headers)
edited_df = st.data_editor(
    df_visual,
    key="parrilla_editor",
    width="stretch",
    num_rows="fixed",
)

# --- Guardar cambios del editor ---
# (para que si el usuario modifica manualmente algo, también se conserve)
st.session_state.grid_data = edited_df.rename(columns={v: k for k, v in headers.items()})


# --- Paso 2: seleccionar circuito ---
st.subheader(TXT["select_track"][L])
st.write(TXT["track_desc"][L])

circuitos = get_circuits()
circuito = st.selectbox("🌍 Circuito / Circuit", options=circuitos)

# --- Paso 3: generar predicción ---
st.subheader(TXT["generate"][L])
st.write(TXT["gen_desc"][L])

if st.button(TXT["btn_predict"][L]):
    # Recuperamos la tabla "real" (sin traducciones)
    df_actual = st.session_state.grid_data.copy()

    # Validamos que todas las posiciones estén completas
    if df_actual["posicion_parrilla"].isnull().any() or (df_actual["posicion_parrilla"] == "").any():
        st.error(TXT["err_positions"][L])
    else:
        parrilla_input = []
        for _, row in df_actual.iterrows():
            parrilla_input.append({
                "piloto": row["piloto"],
                "equipo": row["equipo"],
                "posicion_parrilla": int(row["posicion_parrilla"]),
            })

        predictor = PredictorF1()
        resultado = predictor.predecir_carrera(
            parrilla=parrilla_input,
            circuito=circuito,
            año=date.today().year,
        )

        # Filtramos columnas visibles
        ocultar = {"es_rookie", "posicion_real", "experiencia", "forma_reciente"}
        visible_cols = [c for c in resultado.columns if c not in ocultar]

        # Renombramos para visualización (solo estética)
        resultado_visual = resultado[visible_cols].rename(columns=TXT["table_result_labels"][L])
        st.dataframe(resultado_visual, width="stretch")
