# get_current_drivers.py
import fastf1
import pandas as pd
from datetime import datetime
import os

fastf1.Cache.enable_cache('f1_cache')

def _parse_event_date(cal_df):
    # Intentamos varias columnas posibles que contienen fecha en distintos releases de fastf1
    for col in ("StartDate", "EventDate", "Date"):
        if col in cal_df.columns:
            try:
                return pd.to_datetime(cal_df[col], errors="coerce")
            except Exception:
                continue
    # Si no hay ninguna, devolvemos una serie NaT
    return pd.Series([pd.NaT] * len(cal_df))

def obtener_pilotos_actuales(año=None, use_local_cache=True, cache_path="pilotos_actuales.csv"):
    """
    Devuelve los pilotos y equipos activos buscando la última carrera con fecha < now
    y con resultados disponibles. Si use_local_cache=True y existe cache_path, lo usa primero.
    """
    # Si ya guardaste localmente antes, úsalo (útil para desarrollo offline)
    if use_local_cache and os.path.exists(cache_path):
        try:
            df_cache = pd.read_csv(cache_path)
            if not df_cache.empty:
                print(f"Usando cache local: {cache_path} ({len(df_cache)} pilotos)")
                return df_cache
        except Exception:
            pass

    if año is None:
        año = datetime.now().year

    now = datetime.now()

    # Probar desde el año actual hacia atrás
    for año_prueba in range(año, 2019, -1):
        try:
            calendario = fastf1.get_event_schedule(año_prueba)

            if calendario is None or calendario.empty:
                print(f"No hay calendario para {año_prueba}")
                continue

            # Asegurarnos de tener una columna de fecha parseada
            fechas = _parse_event_date(calendario)
            calendario = calendario.copy()
            calendario["_fecha_evento"] = fechas

            # Filtrar solo carreras convencionales y con fecha conocida
            calendario_filtrado = calendario[
                (calendario.get("EventFormat") == "conventional") &
                (calendario["_fecha_evento"].notna())
            ]

            if calendario_filtrado.empty:
                # Si no hay StartDate en este calendario, intentamos usar todos los eventos
                calendario_filtrado = calendario[
                    (calendario.get("EventFormat") == "conventional")
                ]

            if calendario_filtrado.empty:
                print(f"No hay carreras convencionales en {año_prueba}")
                continue

            # Tomar solo eventos con fecha anterior a 'now' (pasadas)
            calendario_pasadas = calendario_filtrado[calendario_filtrado["_fecha_evento"] < now]

            # Si no hay pasadas, tomar las que tengan fecha (ej. si el año es pasado pero no hay filtrado)
            if calendario_pasadas.empty:
                calendario_pasadas = calendario_filtrado

            # Ordenar por fecha descendente (la más reciente primero)
            calendario_ordenado = calendario_pasadas.sort_values("_fecha_evento", ascending=False)

            # Iterar eventos desde la más reciente hacia atrás hasta encontrar resultados válidos
            for _, evento in calendario_ordenado.iterrows():
                evento_nombre = evento.get("EventName") or evento.get("Event")
                if not evento_nombre:
                    continue

                print(f"Intentando sesión: {año_prueba} - {evento_nombre} (fecha: {evento.get('_fecha_evento')})")
                try:
                    sesion = fastf1.get_session(año_prueba, evento_nombre, 'R')
                    sesion.load(laps=False, telemetry=False)  # cargar sólo lo necesario
                    resultados = getattr(sesion, "results", None)

                    if resultados is None or resultados.empty:
                        print(f"  -> sin resultados, probando evento anterior...")
                        continue

                    pilotos = resultados[['Abbreviation', 'FullName', 'TeamName']].drop_duplicates()

                    df = pd.DataFrame({
                        'piloto': pilotos['Abbreviation'].astype(str),
                        'nombre': pilotos['FullName'].astype(str),
                        'equipo': pilotos['TeamName'].astype(str)
                    }).reset_index(drop=True)

                    # Guardar cache local para evitar futuras consultas
                    try:
                        df.to_csv(cache_path, index=False)
                        print(f"✓ {len(df)} pilotos cargados ({año_prueba} - {evento_nombre}), cache guardada en {cache_path}")
                    except Exception:
                        pass

                    return df

                except Exception as e_event:
                    print(f"  Error cargando sesión {evento_nombre}: {e_event}")
                    continue

        except Exception as e_year:
            print(f"Error en {año_prueba}: {e_year}")
            continue

    print("❌ No se encontraron pilotos en temporadas recientes.")
    return pd.DataFrame(columns=['piloto', 'nombre', 'equipo'])


if __name__ == "__main__":
    df = obtener_pilotos_actuales()
    print(df)
