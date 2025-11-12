import fastf1
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Habilitar cache para evitar descargas repetidas
fastf1.Cache.enable_cache('f1_cache')

def recolectar_datos_historicos(años=[2021,2022,2023,2024,2025]):
    """
    Recolecta datos históricos de F1 y los guarda localmente
    """
    datos_carreras = []
    
    for año in años:
        print(f"Procesando año {año}...")
        
        # Obtener calendario del año
        calendario = fastf1.get_event_schedule(año)
        
        for _, evento in calendario.iterrows():
            if evento['EventFormat'] != 'conventional':
                continue
                
            try:
                # Cargar sesión de carrera
                sesion = fastf1.get_session(año, evento['EventName'], 'R')
                sesion.load()
                
                resultados = sesion.results
                
                for _, piloto in resultados.iterrows():
                    datos_carreras.append({
                        'año': año,
                        'carrera': evento['EventName'],
                        'circuito': evento['Location'],
                        'piloto': piloto['Abbreviation'],
                        'equipo': piloto['TeamName'],
                        'posicion_parrilla': piloto['GridPosition'],
                        'posicion_final': piloto['Position'],
                        'puntos': piloto['Points'],
                        'estado': piloto['Status'],
                        'tiempo': piloto['Time'],
                    })
                
                print(f"  ✓ {evento['EventName']}")
                
            except Exception as e:
                print(f"  ✗ Error en {evento['EventName']}: {e}")
                continue
    
    df = pd.DataFrame(datos_carreras)
    df.to_csv('datos_f1_historicos.csv', index=False)
    print(f"\n✓ Datos guardados: {len(df)} registros")
    return df

if __name__ == "__main__":
    print("=== RECOLECCIÓN DE DATOS F1 ===\n")
    df = recolectar_datos_historicos()
    print("\n✓ Proceso completado!")
    print(f"Archivo generado: datos_f1_historicos.csv")