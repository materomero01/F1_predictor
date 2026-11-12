import fastf1
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

fastf1.Cache.enable_cache('f1_cache')

def recolectar_carreras_faltantes():
    """
    Recolecta las carreras del archivo carreras_faltantes.csv
    y las añade a datos_f1_historicos.csv
    """
    print("=== RECOLECCIÓN DE CARRERAS FALTANTES ===\n")
    
    # Cargar listado de faltantes
    try:
        df_faltantes = pd.read_csv('carreras_faltantes.csv')
        print(f"✓ {len(df_faltantes)} carreras faltantes a recolectar\n")
    except FileNotFoundError:
        print("❌ Error: No se encontró carreras_faltantes.csv")
        print("   Ejecuta primero: python listar_carreras_faltantes.py")
        return
    
    # Cargar datos existentes
    try:
        df_existente = pd.read_csv('datos_f1_historicos.csv')
        print(f"✓ Datos existentes: {len(df_existente)} registros\n")
    except FileNotFoundError:
        df_existente = pd.DataFrame()
        print("⚠ No hay datos existentes, creando archivo nuevo\n")
    
    datos_nuevos = []
    exitosas = 0
    fallidas = 0
    
    print(f"{'Año':<6} {'Carrera':<40} {'Estado'}")
    print("-" * 70)
    
    for _, faltante in df_faltantes.iterrows():
        año = int(faltante['año'])
        carrera = faltante['carrera']
        
        try:
            # Cargar sesión
            sesion = fastf1.get_session(año, carrera, 'R')
            sesion.load()
            
            resultados = sesion.results
            
            if resultados is None or len(resultados) == 0:
                print(f"{año:<6} {carrera:<40} ✗ Sin resultados")
                fallidas += 1
                continue
            
            # Extraer datos
            for _, piloto in resultados.iterrows():
                datos_nuevos.append({
                    'año': año,
                    'carrera': carrera,
                    'circuito': faltante.get('circuito', 'Unknown'),
                    'pais': faltante.get('pais', 'Unknown'),
                    'formato': faltante.get('formato', 'conventional'),
                    'piloto': piloto['Abbreviation'],
                    'equipo': piloto['TeamName'],
                    'posicion_parrilla': piloto['GridPosition'],
                    'posicion_final': piloto['Position'],
                    'puntos': piloto['Points'],
                    'estado': piloto['Status'],
                    'tiempo': piloto['Time'],
                })
            
            print(f"{año:<6} {carrera:<40} ✓ {len(resultados)} pilotos")
            exitosas += 1
            
        except Exception as e:
            error_msg = str(e)[:30]
            print(f"{año:<6} {carrera:<40} ✗ {error_msg}")
            fallidas += 1
    
    # Combinar y guardar
    if len(datos_nuevos) > 0:
        df_nuevos = pd.DataFrame(datos_nuevos)
        
        # Combinar con existentes
        if len(df_existente) > 0:
            df_total = pd.concat([df_existente, df_nuevos], ignore_index=True)
        else:
            df_total = df_nuevos
        
        # Guardar
        df_total.to_csv('datos_f1_historicos.csv', index=False)
        
        print(f"\n{'='*70}")
        print("RESUMEN")
        print(f"{'='*70}")
        print(f"✓ Carreras recolectadas:     {exitosas}")
        print(f"✗ Carreras fallidas:         {fallidas}")
        print(f"📊 Nuevos registros:          {len(df_nuevos)}")
        print(f"📊 Total registros ahora:     {len(df_total)}")
        print(f"\n✓ Archivo actualizado: datos_f1_historicos.csv")
        
        # Actualizar listado de faltantes
        if fallidas > 0:
            # Mantener solo las que fallaron
            df_faltantes_restantes = df_faltantes[
                df_faltantes.apply(
                    lambda x: (int(x['año']), x['carrera']) not in 
                    set(df_nuevos.groupby(['año', 'carrera']).size().index.map(lambda i: (i[0], i[1]))),
                    axis=1
                )
            ]
            
            if len(df_faltantes_restantes) > 0:
                df_faltantes_restantes.to_csv('carreras_faltantes.csv', index=False)
                print(f"\n⚠ Quedan {len(df_faltantes_restantes)} carreras sin recolectar")
                print("  (carreras_faltantes.csv actualizado)")
            else:
                import os
                os.remove('carreras_faltantes.csv')
                print("\n🎉 ¡Todas las carreras recolectadas exitosamente!")
        else:
            import os
            try:
                os.remove('carreras_faltantes.csv')
                print("\n🎉 ¡Todas las carreras recolectadas exitosamente!")
            except:
                pass
        
        return df_total
    else:
        print("\n❌ No se pudieron recolectar nuevos datos")
        return None

if __name__ == "__main__":
    recolectar_carreras_faltantes()