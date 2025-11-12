import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def crear_features_avanzadas(df):
    """
    Crea features ponderadas por experiencia y rendimiento
    """
    df = df.copy()
    
    # CRÍTICO: Asegurar que año sea numérico
    df['año'] = pd.to_numeric(df['año'], errors='coerce')
    
    # Crear fecha ordenable para ordenamiento correcto
    # Necesitamos ordenar por año Y por orden de carrera en el calendario
    df['fecha_orden'] = df['año'] * 100 + df.groupby('año').cumcount()
    
    # Ordenar cronológicamente
    df = df.sort_values(['fecha_orden', 'piloto']).reset_index(drop=True)
    
    print(f"Años únicos en datos de entrada: {sorted(df['año'].unique())}")
    print(f"Total de carreras: {df.groupby(['año', 'carrera']).ngroups}")
    
    # Convertir posiciones a numéricas
    df['posicion_final_num'] = pd.to_numeric(df['posicion_final'], errors='coerce')
    df['posicion_parrilla_num'] = pd.to_numeric(df['posicion_parrilla'], errors='coerce')
    
    features = []
    pilotos_procesados = 0
    registros_por_año = {}
    
    for piloto in df['piloto'].unique():
        df_piloto = df[df['piloto'] == piloto].sort_values('fecha_orden').reset_index(drop=True)
        
        for idx in range(len(df_piloto)):
            # Datos históricos hasta esta carrera (sin incluir la actual)
            historico = df_piloto.iloc[:idx]
            
            # Mínimo 3 carreras previas
            if len(historico) < 3:
                continue
            
            carrera_actual = df_piloto.iloc[idx]
            año_actual = int(carrera_actual['año'])
            
            # Contador por año
            if año_actual not in registros_por_año:
                registros_por_año[año_actual] = 0
            registros_por_año[año_actual] += 1
            
            # === EXPERIENCIA ===
            carreras_totales = len(historico)
            
            # === PONDERACIÓN EXPONENCIAL ===
            pesos = np.exp(np.linspace(-2, 0, len(historico)))
            pesos = pesos / pesos.sum()
            
            # === POSICIONES PONDERADAS ===
            posiciones = historico['posicion_final_num'].fillna(20)
            promedio_posicion_ponderado = np.average(posiciones, weights=pesos)
            
            # === PUNTOS PONDERADOS ===
            puntos = historico['puntos'].fillna(0)
            promedio_puntos_ponderado = np.average(puntos, weights=pesos)
            
            # === FORMA RECIENTE ===
            ultimas_3 = historico.tail(3)
            forma_reciente = ultimas_3['posicion_final_num'].fillna(20).mean()
            puntos_recientes = ultimas_3['puntos'].sum()
            
            ultimas_5 = historico.tail(5)
            podios_ultimas_5 = (ultimas_5['posicion_final_num'] <= 3).sum()
            
            # === CONSISTENCIA ===
            std_posiciones = historico['posicion_final_num'].fillna(20).std()
            
            # === TASA DE FINALIZACIÓN ===
            finalizadas = (historico['estado'] == 'Finished').sum()
            tasa_finalizacion = finalizadas / len(historico) if len(historico) > 0 else 0
            
            # === CLASIFICACIÓN ===
            promedio_parrilla = historico['posicion_parrilla_num'].fillna(20).mean()
            
            # === MEJORA PARRILLA -> CARRERA ===
            mejora_promedio = (historico['posicion_parrilla_num'].fillna(20) - 
                              historico['posicion_final_num'].fillna(20)).mean()
            
            # === CIRCUITO ESPECÍFICO ===
            circuito_actual = carrera_actual['circuito']
            historico_circuito = historico[historico['circuito'] == circuito_actual]
            
            if len(historico_circuito) > 0:
                promedio_posicion_circuito = historico_circuito['posicion_final_num'].fillna(20).mean()
                carreras_en_circuito = len(historico_circuito)
            else:
                promedio_posicion_circuito = promedio_posicion_ponderado
                carreras_en_circuito = 0
            
            features.append({
                'año': año_actual,
                'carrera': carrera_actual['carrera'],
                'circuito': carrera_actual['circuito'],
                'piloto': piloto,
                'equipo': carrera_actual['equipo'],
                'posicion_final': carrera_actual['posicion_final_num'],
                'gano': 1 if carrera_actual['posicion_final_num'] == 1 else 0,
                'podio': 1 if carrera_actual['posicion_final_num'] <= 3 else 0,
                'puntos_obtenidos': carrera_actual['puntos'],
                'experiencia_carreras': carreras_totales,
                'posicion_parrilla': carrera_actual['posicion_parrilla_num'],
                'promedio_posicion_ponderado': promedio_posicion_ponderado,
                'promedio_puntos_ponderado': promedio_puntos_ponderado,
                'forma_reciente_3_carreras': forma_reciente,
                'puntos_ultimas_3': puntos_recientes,
                'podios_ultimas_5': podios_ultimas_5,
                'consistencia_std': std_posiciones,
                'tasa_finalizacion': tasa_finalizacion,
                'promedio_parrilla': promedio_parrilla,
                'mejora_parrilla_carrera': mejora_promedio,
                'promedio_posicion_circuito': promedio_posicion_circuito,
                'experiencia_circuito': carreras_en_circuito,
            })
        
        pilotos_procesados += 1
    
    print(f"\n✓ {pilotos_procesados} pilotos procesados")
    print(f"\nRegistros generados por año:")
    for año in sorted(registros_por_año.keys()):
        print(f"  {año}: {registros_por_año[año]} registros")
    
    return pd.DataFrame(features)

if __name__ == "__main__":
    print("=== CREACIÓN DE FEATURES ===\n")
    
    # Cargar datos históricos
    print("Cargando datos históricos...")
    try:
        df_historico = pd.read_csv('datos_f1_historicos.csv')
        print(f"✓ {len(df_historico)} registros cargados")
        
        # VERIFICACIÓN CRÍTICA
        print(f"\nAños en datos históricos: {sorted(df_historico['año'].unique())}")
        carreras_por_año = df_historico.groupby('año')['carrera'].nunique()
        print(f"\nCarreras por año:")
        print(carreras_por_año)
        
    except FileNotFoundError:
        print("❌ Error: No se encontró 'datos_f1_historicos.csv'")
        print("   Ejecuta primero: python collect_data.py")
        exit(1)
    
    # Crear features
    print("\nGenerando features...")
    df_features = crear_features_avanzadas(df_historico)
    
    if len(df_features) == 0:
        print("\n❌ ERROR: No se generaron features!")
        print("Verifica que hay suficientes carreras por piloto (mínimo 3)")
        exit(1)
    
    # Guardar
    df_features.to_csv('datos_f1_features.csv', index=False)
    print(f"\n✓ Features guardadas: {len(df_features)} registros")
    
    # Resumen final
    print("\n=== RESUMEN FINAL ===")
    resumen = df_features.groupby('año').agg({
        'carrera': 'nunique',
        'piloto': 'nunique',
        'posicion_final': 'count'
    }).rename(columns={'carrera': 'carreras', 'piloto': 'pilotos', 'posicion_final': 'registros'})
    print(resumen)
    
    print("\n✓ Archivo generado: datos_f1_features.csv")