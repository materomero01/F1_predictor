import fastf1 as ff1
from datetime import date

# 1. Habilitar la caché (opcional, pero recomendado para acelerar cargas repetidas)
ff1.Cache.enable_cache('f1_cache')

# 2. Obtener el año actual
current_year = date.today().year

# 3. Cargar el calendario (schedule) para el año actual
try:
    schedule = ff1.get_event_schedule(current_year)
except ff1.api.SessionNotAvailableError:
    # Si el calendario del año actual aún no está completamente disponible,
    # puedes intentar con un año anterior o manejar el error.
    print(f"⚠️ No se pudo cargar el calendario completo para el año {current_year}.")
    print("Por favor, verifica si la temporada de F1 ha comenzado o si hay un error en FastF1.")
    schedule = None


if schedule is not None:
    # 4. Filtrar solo los eventos que son "carreras" (IsRace == True)
    # y que no han sido cancelados (EventName no contiene 'Test', 'Practice', 'Pre-Season')
    
    # La columna 'EventName' en FastF1 generalmente tiene el nombre del Gran Premio.
    # El nombre del circuito está en la columna 'Location' o 'CircuitName' (que a veces no está presente).
    
    # Vamos a usar 'Location' (que es el nombre de la ciudad o el nombre corto del circuito)
    # o 'EventName' (que es el nombre del Gran Premio) y filtrar.
    
    # FastF1 a partir de la versión 3.x mejoró la estructura. Usaremos las columnas disponibles.
    
    # Filtrar solo los eventos principales (carreras) que tienen una 'Location' definida
    race_events = schedule[schedule['EventFormat'] != 'Test']
    
    # Obtenemos los nombres de los circuitos o ubicaciones
    circuitos = race_events['Location'].unique()

    # 5. Imprimir los resultados
    print(f"## 🏎️ Circuitos de la Temporada {current_year} (según 'Location'):")
    print("-" * 50)
    for nombre in circuitos:
        # Algunos eventos como 'Bahrain Test' pueden colarse si la API los incluye y no tienen un 'Location' único
        if nombre: # Asegura que el valor no esté vacío o None
            print(f"* {nombre}")
    print("-" * 50)

    # Si prefieres ver el nombre oficial del evento (Gran Premio):
    gp_names = race_events['EventName'].unique()
    print("\n## 🏁 Nombres de los Grandes Premios (EventName):")
    print("-" * 50)
    for nombre in gp_names:
        if nombre:
            print(f"* {nombre}")
    print("-" * 50)