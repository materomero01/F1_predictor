import pandas as pd
import pickle
import numpy as np
from datetime import datetime

class PredictorF1:
    def __init__(self):
        """Cargar modelos y encoders"""
        print("Cargando modelos...")
        
        with open('modelo_regresion.pkl', 'rb') as f:
            self.modelo_regresion = pickle.load(f)
        
        with open('modelo_ganador.pkl', 'rb') as f:
            self.modelo_ganador = pickle.load(f)
            
        with open('modelo_podio.pkl', 'rb') as f:
            self.modelo_podio = pickle.load(f)
        
        with open('label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
            self.le_equipo = encoders['equipo']
            self.le_piloto = encoders['piloto']
        
        with open('feature_cols.pkl', 'rb') as f:
            self.feature_cols = pickle.load(f)
        
        # Cargar datos históricos
        self.df_historico = pd.read_csv('datos_f1_historicos.csv')
        self.df_historico['año'] = pd.to_numeric(self.df_historico['año'])
        self.df_historico['posicion_final_num'] = pd.to_numeric(
            self.df_historico['posicion_final'], errors='coerce'
        )
        self.df_historico['posicion_parrilla_num'] = pd.to_numeric(
            self.df_historico['posicion_parrilla'], errors='coerce'
        )
        
        # Calcular promedios por equipo (para rookies)
        self.promedios_equipo = self._calcular_promedios_equipo()
        
        print("✓ Modelos cargados correctamente\n")
    
    def _calcular_promedios_equipo(self):
        """Calcula estadísticas promedio por equipo para usar con rookies"""
        promedios = {}
        
        for equipo in self.df_historico['equipo'].unique():
            df_equipo = self.df_historico[self.df_historico['equipo'] == equipo]
            
            promedios[equipo] = {
                'promedio_posicion': df_equipo['posicion_final_num'].fillna(20).mean(),
                'promedio_puntos': df_equipo['puntos'].fillna(0).mean(),
                'tasa_finalizacion': (df_equipo['estado'] == 'Finished').mean(),
                'promedio_parrilla': df_equipo['posicion_parrilla_num'].fillna(20).mean(),
            }
        
        return promedios
    
    def calcular_features_piloto(self, piloto, equipo, circuito, posicion_parrilla, fecha_limite=None):
        """
        Calcula features para un piloto basándose en su historia
        Si es rookie, usa estadísticas del equipo
        """
        # Filtrar histórico del piloto
        df_piloto = self.df_historico[self.df_historico['piloto'] == piloto].copy()
        
        if fecha_limite:
            df_piloto = df_piloto[df_piloto['año'] < fecha_limite['año']]
        
        es_rookie = len(df_piloto) < 3
        
        if es_rookie:
            # ROOKIE: Usar estadísticas del equipo + penalización
            return self._features_rookie(equipo, circuito, posicion_parrilla)
        else:
            # PILOTO CON HISTORIAL
            return self._features_veterano(df_piloto, circuito, posicion_parrilla)
    
    def _features_rookie(self, equipo, circuito, posicion_parrilla):
        """Features para rookies basadas en el equipo"""
        # Buscar equipo con nombre similar si no existe exactamente
        equipo_stats = None
        
        # Intentar match exacto
        if equipo in self.promedios_equipo:
            equipo_stats = self.promedios_equipo[equipo]
        else:
            # Buscar match parcial (ej: "Aston Martin" en "Aston Martin Aramco")
            for eq_conocido in self.promedios_equipo.keys():
                if equipo.split()[0] in eq_conocido or eq_conocido.split()[0] in equipo:
                    equipo_stats = self.promedios_equipo[eq_conocido]
                    break
        
        # Si no hay match, usar promedios generales de la parrilla
        if equipo_stats is None:
            equipo_stats = {
                'promedio_posicion': 10.5,
                'promedio_puntos': 5.0,
                'tasa_finalizacion': 0.75,
                'promedio_parrilla': 10.5,
            }
        
        # Features con penalización por inexperiencia
        features = {
            'experiencia_carreras': 0,  # Rookie
            'posicion_parrilla': posicion_parrilla,
            'promedio_posicion_ponderado': equipo_stats['promedio_posicion'] + 2,  # Penalización
            'promedio_puntos_ponderado': max(0, equipo_stats['promedio_puntos'] - 3),  # Penalización
            'forma_reciente_3_carreras': equipo_stats['promedio_posicion'] + 3,  # Penalización rookie
            'puntos_ultimas_3': 0,
            'podios_ultimas_5': 0,
            'consistencia_std': 5.0,  # Alta variabilidad (inseguridad)
            'tasa_finalizacion': max(0.5, equipo_stats['tasa_finalizacion'] - 0.15),  # Penalización
            'promedio_parrilla': equipo_stats['promedio_parrilla'] + 2,
            'mejora_parrilla_carrera': -1.0,  # Asumimos que pierde posiciones
            'promedio_posicion_circuito': equipo_stats['promedio_posicion'] + 2,
            'experiencia_circuito': 0,
        }
        
        return features
    
    def _features_veterano(self, df_piloto, circuito, posicion_parrilla):
        """Features para pilotos con historial"""
        # Ordenar cronológicamente
        df_piloto = df_piloto.sort_values(['año', 'carrera'])
        
        # Ponderación exponencial
        pesos = np.exp(np.linspace(-2, 0, len(df_piloto)))
        pesos = pesos / pesos.sum()
        
        # Calcular features
        posiciones = df_piloto['posicion_final_num'].fillna(20)
        puntos = df_piloto['puntos'].fillna(0)
        
        ultimas_3 = df_piloto.tail(3)
        ultimas_5 = df_piloto.tail(5)
        
        # Features del circuito específico
        df_circuito = df_piloto[df_piloto['circuito'] == circuito]
        if len(df_circuito) > 0:
            promedio_posicion_circuito = df_circuito['posicion_final_num'].fillna(20).mean()
            experiencia_circuito = len(df_circuito)
        else:
            promedio_posicion_circuito = np.average(posiciones, weights=pesos)
            experiencia_circuito = 0
        
        features = {
            'experiencia_carreras': len(df_piloto),
            'posicion_parrilla': posicion_parrilla,
            'promedio_posicion_ponderado': np.average(posiciones, weights=pesos),
            'promedio_puntos_ponderado': np.average(puntos, weights=pesos),
            'forma_reciente_3_carreras': ultimas_3['posicion_final_num'].fillna(20).mean(),
            'puntos_ultimas_3': ultimas_3['puntos'].sum(),
            'podios_ultimas_5': (ultimas_5['posicion_final_num'] <= 3).sum(),
            'consistencia_std': df_piloto['posicion_final_num'].fillna(20).std(),
            'tasa_finalizacion': (df_piloto['estado'] == 'Finished').sum() / len(df_piloto),
            'promedio_parrilla': df_piloto['posicion_parrilla_num'].fillna(20).mean(),
            'mejora_parrilla_carrera': (df_piloto['posicion_parrilla_num'].fillna(20) - 
                                        df_piloto['posicion_final_num'].fillna(20)).mean(),
            'promedio_posicion_circuito': promedio_posicion_circuito,
            'experiencia_circuito': experiencia_circuito,
        }
        
        return features
    
    def predecir_carrera(self, parrilla, circuito=None, año=None, incluir_rookies=True, posiciones_reales=None):
        """
        Predice el resultado de una carrera

        Parameters:
        -----------
        parrilla : list of dict
            [{'piloto': 'VER', 'equipo': 'Red Bull Racing', 'posicion_parrilla': 1}, ...]
        circuito : str, optional
            Nombre del circuito
        año : int, optional
            Año de la carrera
        incluir_rookies : bool
            Si True, predice rookies usando estadísticas del equipo
        posiciones_reales : dict, optional
            Diccionario con posiciones reales, ej: {'VER': 3, 'NOR': 1, ...}
        """
        if año is None:
            año = datetime.now().year

        print(f"\n{'='*60}")
        print(f"PREDICCIÓN DE CARRERA")
        print(f"{'='*60}")
        if circuito:
            print(f"Circuito: {circuito}")
        print(f"Año: {año}")
        print(f"Pilotos en parrilla: {len(parrilla)}\n")

        predicciones = []
        pilotos_rookie = []

        for entrada in parrilla:
            piloto = entrada['piloto']
            equipo = entrada['equipo']
            pos_parrilla = entrada['posicion_parrilla']

            # Calcular features (maneja rookies automáticamente)
            features = self.calcular_features_piloto(
                piloto, equipo, circuito or 'Unknown', pos_parrilla, fecha_limite={'año': año}
            )

            if features is None:
                continue

            es_rookie = features['experiencia_carreras'] == 0
            if es_rookie:
                pilotos_rookie.append(piloto)

            # Codificar equipo
            if equipo in self.le_equipo.classes_:
                equipo_encoded = self.le_equipo.transform([equipo])[0]
            else:
                equipo_encoded = 0

            features['equipo_encoded'] = equipo_encoded
            X = pd.DataFrame([features])[self.feature_cols]

            # Predicciones
            posicion_pred = self.modelo_regresion.predict(X)[0]
            prob_ganador = self.modelo_ganador.predict_proba(X)[0, 1]
            prob_podio = self.modelo_podio.predict_proba(X)[0, 1]

            predicciones.append({
                'piloto': piloto,
                'equipo': equipo,
                'posicion_parrilla': pos_parrilla,
                'posicion_predicha': posicion_pred,
                'prob_ganar_%': prob_ganador * 100,
                'prob_podio_%': prob_podio * 100,
                'posicion_real': posiciones_reales.get(piloto) if posiciones_reales else None,
                'experiencia': features['experiencia_carreras'],
                'forma_reciente': features['forma_reciente_3_carreras'],
                'es_rookie': '🆕' if es_rookie else '',
            })

        df_pred = pd.DataFrame(predicciones)
        df_pred = df_pred.sort_values('posicion_predicha').reset_index(drop=True)
        df_pred.index = df_pred.index + 1

        if pilotos_rookie:
            print(f"🆕 Rookies predichos (basado en equipo): {', '.join(pilotos_rookie)}\n")

        return df_pred


    def mostrar_prediccion(self, df_pred):
        """Muestra la predicción incluyendo (opcionalmente) posición real"""
        print(f"\n{'='*105}")
        print(f"{'PREDICCIÓN DE RESULTADOS':^105}")
        print(f"{'='*105}\n")

        # Determinar si hay columna de posición real
        mostrar_real = 'posicion_real' in df_pred.columns and df_pred['posicion_real'].notna().any()

        if mostrar_real:
            print(f"{'Pos':>3} {'Piloto':<6} {'Equipo':<25} {'Parrilla':>8} {'Pred':>6} {'Real':>6} {'Ganar%':>8} {'Podio%':>8} {'':>3}")
            print("-" * 105)
            for idx, row in df_pred.iterrows():
                real = int(row['posicion_real']) if not pd.isna(row['posicion_real']) else '-'
                print(f"{idx:>3} {row['piloto']:<6} {row['equipo']:<25} "
                    f"{row['posicion_parrilla']:>8.0f} {row['posicion_predicha']:>6.1f} "
                    f"{real:>6} {row['prob_ganar_%']:>8.1f} {row['prob_podio_%']:>8.1f} {row['es_rookie']:>3}")
        else:
            print(f"{'Pos':>3} {'Piloto':<6} {'Equipo':<25} {'Parrilla':>8} {'Pred':>6} {'Ganar%':>8} {'Podio%':>8} {'':>3}")
            print("-" * 85)
            for idx, row in df_pred.iterrows():
                print(f"{idx:>3} {row['piloto']:<6} {row['equipo']:<25} "
                    f"{row['posicion_parrilla']:>8.0f} {row['posicion_predicha']:>6.1f} "
                    f"{row['prob_ganar_%']:>8.1f} {row['prob_podio_%']:>8.1f} {row['es_rookie']:>3}")
        print("-" * 105)

        # Mostrar podio
        print(f"\n🏆 PODIO PREDICHO:")
        for i in range(min(3, len(df_pred))):
            medalla = ['🥇', '🥈', '🥉'][i]
            row = df_pred.iloc[i]
            print(f"  {medalla} {row['piloto']} ({row['equipo']}) - {row['prob_ganar_%']:.1f}% ganar")


# === EJEMPLO DE USO ===
if __name__ == "__main__":
    # Inicializar predictor
    predictor = PredictorF1()
    
    # Parrilla 2025 con rookies
    parrilla_ejemplo = [
        # Red Bull Racing
        {'piloto': 'VER', 'equipo': 'Red Bull Racing', 'posicion_parrilla': 19},
        {'piloto': 'TSU', 'equipo': 'Red Bull Racing', 'posicion_parrilla': 17},

        # Mercedes
        {'piloto': 'RUS', 'equipo': 'Mercedes', 'posicion_parrilla': 6},
        {'piloto': 'ANT', 'equipo': 'Mercedes', 'posicion_parrilla': 2},  # Rookie

        # Ferrari
        {'piloto': 'LEC', 'equipo': 'Ferrari', 'posicion_parrilla': 3},
        {'piloto': 'HAM', 'equipo': 'Ferrari', 'posicion_parrilla': 13},

        # McLaren
        {'piloto': 'NOR', 'equipo': 'McLaren', 'posicion_parrilla': 1},
        {'piloto': 'PIA', 'equipo': 'McLaren', 'posicion_parrilla': 5},

        # Aston Martin
        {'piloto': 'ALO', 'equipo': 'Aston Martin', 'posicion_parrilla': 11},
        {'piloto': 'STR', 'equipo': 'Aston Martin', 'posicion_parrilla': 14},

        # Alpine
        {'piloto': 'GAS', 'equipo': 'Alpine F1 Team', 'posicion_parrilla': 9},
        {'piloto': 'COL', 'equipo': 'Alpine F1 Team', 'posicion_parrilla': 16},  # Rookie

        # Williams
        {'piloto': 'ALB', 'equipo': 'Williams', 'posicion_parrilla': 12},
        {'piloto': 'SAI', 'equipo': 'Williams', 'posicion_parrilla': 15},

        # RB
        {'piloto': 'LAW', 'equipo': 'RB', 'posicion_parrilla': 7},
        {'piloto': 'HAD', 'equipo': 'RB', 'posicion_parrilla': 5},  # Rookie

        # Sauber
        {'piloto': 'HUL', 'equipo': 'Kick Sauber', 'posicion_parrilla': 10},
        {'piloto': 'BOR', 'equipo': 'Kick Sauber', 'posicion_parrilla': 18},  # Rookie

        # Haas
        {'piloto': 'OCO', 'equipo': 'Haas F1 Team', 'posicion_parrilla': 20},
        {'piloto': 'BEA', 'equipo': 'Haas F1 Team', 'posicion_parrilla': 8},  # Rookie
    ]
    
    # Posiciones reales (solo si las tenés)
    posiciones_reales = {
        'NOR': 1,
        'ANT': 2,
        'VER': 3,
        'RUS': 4,
        'PIA': 5,
        'BEA': 6,
        'LAW': 7,
        'HAD': 8,
        'HUL': 9,
        'GAS': 10,
        'ALB': 11,
        'OCO': 12,
        'SAI': 13,
        'ALO': 14,
        'COL': 15,
        'STR': 16,
        'TSU': 17,
        'HAM': None,  # Abandono
        'LEC': None,  # Abandono
        'BOR': None,  # Abandono
    }

    # Predecir
    resultado = predictor.predecir_carrera(
        parrilla=parrilla_ejemplo,
        circuito='São Paulo Grand Prix',
        año=2025,
        posiciones_reales=posiciones_reales,
    )
    
    # Mostrar
    predictor.mostrar_prediccion(resultado)
    
    # Guardar
    resultado.to_csv('prediccion_carrera.csv', index=False)
    print(f"\n✓ Predicción guardada en: prediccion_carrera.csv")