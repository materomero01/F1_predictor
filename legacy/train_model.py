import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print("=== ENTRENAMIENTO MODELO F1 ===\n")

# Cargar datos
print("Cargando features...")
df = pd.read_csv('datos_f1_features.csv')
print(f"✓ {len(df)} registros cargados\n")

print("Años en el dataset:")
print(df['año'].value_counts().sort_index())

# === LIMPIEZA DE DATOS ===
print("\n=== LIMPIEZA DE DATOS ===")
print(f"Registros antes de limpieza: {len(df)}")

# Verificar NaN en columnas críticas
print("\nValores nulos por columna:")
nulos = df.isnull().sum()
if nulos.sum() > 0:
    print(nulos[nulos > 0])

# Eliminar filas con NaN en posicion_final (target)
df = df.dropna(subset=['posicion_final'])
print(f"Registros después de eliminar NaN en target: {len(df)}")

# Rellenar NaN en features numéricas con valores sensatos
features_numericas = [
    'experiencia_carreras', 'posicion_parrilla', 'promedio_posicion_ponderado',
    'promedio_puntos_ponderado', 'forma_reciente_3_carreras', 'puntos_ultimas_3',
    'consistencia_std', 'tasa_finalizacion', 'promedio_parrilla',
    'mejora_parrilla_carrera', 'promedio_posicion_circuito', 'experiencia_circuito'
]

for col in features_numericas:
    if col in df.columns and df[col].isnull().any():
        # Rellenar con mediana para valores numéricos
        mediana = df[col].median()
        df[col].fillna(mediana, inplace=True)
        print(f"  Rellenados {col} con mediana: {mediana:.2f}")

# Codificar variables categóricas
print("\nCodificando variables...")
le_equipo = LabelEncoder()
df['equipo_encoded'] = le_equipo.fit_transform(df['equipo'])

le_piloto = LabelEncoder()
df['piloto_encoded'] = le_piloto.fit_transform(df['piloto'])

# Features
feature_cols = [
    'experiencia_carreras',
    'posicion_parrilla',
    'promedio_posicion_ponderado',
    'promedio_puntos_ponderado',
    'forma_reciente_3_carreras',
    'puntos_ultimas_3',
    'consistencia_std',
    'tasa_finalizacion',
    'promedio_parrilla',
    'mejora_parrilla_carrera',
    'equipo_encoded',
]

# Verificar que todas las features existen
features_faltantes = [f for f in feature_cols if f not in df.columns]
if features_faltantes:
    print(f"\n⚠ Features faltantes: {features_faltantes}")
    print("Usando solo features disponibles...")
    feature_cols = [f for f in feature_cols if f in df.columns]

X = df[feature_cols]
y = df['posicion_final']

# Verificación final de NaN
print("\n=== VERIFICACIÓN FINAL ===")
print(f"NaN en X: {X.isnull().sum().sum()}")
print(f"NaN en y: {y.isnull().sum()}")
print(f"Inf en X: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
print(f"Inf en y: {np.isinf(y).sum()}")

if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print("\n❌ ERROR: Aún hay valores nulos después de la limpieza")
    print("\nFilas con NaN en X:")
    print(X[X.isnull().any(axis=1)])
    exit(1)

# === ESTRATEGIA DE SPLIT TEMPORAL ===
años_unicos = sorted(df['año'].unique())
print(f"\nAños disponibles: {años_unicos}")

if len(años_unicos) == 1:
    print("\n⚠ Solo hay un año de datos, usando split aleatorio (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    df_train = df.iloc[X_train.index]
    df_test = df.iloc[X_test.index]
else:
    # Split temporal: últimas N carreras para test
    carreras_unicas = df[['año', 'carrera']].drop_duplicates().sort_values(['año', 'carrera'])
    n_carreras_test = max(5, int(len(carreras_unicas) * 0.2))
    
    carreras_test = carreras_unicas.tail(n_carreras_test)
    
    test_mask = df.set_index(['año', 'carrera']).index.isin(
        carreras_test.set_index(['año', 'carrera']).index
    )
    train_mask = ~test_mask
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    df_train = df[train_mask]
    df_test = df[test_mask]
    
    print(f"\n✓ Split temporal: últimas {n_carreras_test} carreras para test")

print(f"✓ Train: {len(X_train)} registros")
print(f"✓ Test: {len(X_test)} registros")

# Verificar que hay datos
if len(X_train) == 0:
    print("\n❌ ERROR: No hay datos de entrenamiento")
    exit(1)

# === MODELO REGRESIÓN ===
print("\n=== ENTRENANDO MODELO REGRESIÓN ===")
modelo_regresion = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'  # Más estable con datos problemáticos
)

modelo_regresion.fit(X_train, y_train)
predicciones = modelo_regresion.predict(X_test)

mae = mean_absolute_error(y_test, predicciones)
print(f"\n✓ Modelo entrenado")
print(f"MAE (Error promedio de posiciones): {mae:.2f}")

# Estadísticas adicionales
print("\nEstadísticas de predicción:")
print(f"  Error ±3 posiciones: {(abs(predicciones - y_test) <= 3).mean()*100:.1f}%")
print(f"  Error ±5 posiciones: {(abs(predicciones - y_test) <= 5).mean()*100:.1f}%")

# === MODELO CLASIFICACIÓN GANADOR ===
print("\n=== ENTRENANDO MODELO GANADOR ===")
y_train_gano = df_train['gano']
y_test_gano = df_test['gano']

n_ganadores_train = y_train_gano.sum()
print(f"Ganadores en train: {n_ganadores_train}")

if n_ganadores_train > 0:
    scale_pos = len(y_train_gano) / n_ganadores_train
else:
    scale_pos = 1.0
    print("⚠ No hay ganadores en train, usando scale_pos=1.0")

modelo_ganador = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=scale_pos,
    random_state=42,
    tree_method='hist'
)

modelo_ganador.fit(X_train, y_train_gano)
pred_ganador = modelo_ganador.predict(X_test)
prob_ganador = modelo_ganador.predict_proba(X_test)[:, 1]

print(f"\n✓ Modelo entrenado")
print(f"Accuracy: {accuracy_score(y_test_gano, pred_ganador):.3f}")

# === MODELO PODIO ===
print("\n=== ENTRENANDO MODELO PODIO ===")
y_train_podio = df_train['podio']
y_test_podio = df_test['podio']

n_podios_train = y_train_podio.sum()
scale_pos_podio = len(y_train_podio) / n_podios_train if n_podios_train > 0 else 1.0

modelo_podio = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=scale_pos_podio,
    random_state=42,
    tree_method='hist'
)

modelo_podio.fit(X_train, y_train_podio)
pred_podio = modelo_podio.predict(X_test)

print(f"\n✓ Modelo entrenado")
print(f"Accuracy: {accuracy_score(y_test_podio, pred_podio):.3f}")

# === IMPORTANCIA DE FEATURES ===
importancias = pd.DataFrame({
    'feature': feature_cols,
    'importancia': modelo_regresion.feature_importances_
}).sort_values('importancia', ascending=False)

print("\n=== IMPORTANCIA DE FEATURES ===")
print(importancias.to_string(index=False))

# Visualización
try:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Importancia
    sns.barplot(data=importancias, x='importancia', y='feature', ax=axes[0])
    axes[0].set_title('Importancia de Features', fontsize=14, fontweight='bold')
    
    # Distribución de errores
    errores = abs(predicciones - y_test)
    axes[1].hist(errores, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(mae, color='red', linestyle='--', linewidth=2, label=f'MAE: {mae:.2f}')
    axes[1].set_xlabel('Error Absoluto (posiciones)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Errores', fontsize=14, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('importancia_features.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado: importancia_features.png")
except Exception as e:
    print(f"\n⚠ No se pudo generar gráfico: {e}")

# === ANÁLISIS DE PREDICCIONES ===
print("\n=== EJEMPLO DE PREDICCIONES ===")
resultados_test = df_test.copy()
resultados_test['posicion_predicha'] = predicciones
resultados_test['error'] = abs(resultados_test['posicion_final'] - resultados_test['posicion_predicha'])

print("\nMejores predicciones:")
mejores = resultados_test.nsmallest(5, 'error')[['carrera', 'piloto', 'posicion_final', 'posicion_predicha', 'error']]
print(mejores.to_string(index=False))

print("\nPeores predicciones:")
peores = resultados_test.nlargest(5, 'error')[['carrera', 'piloto', 'posicion_final', 'posicion_predicha', 'error']]
print(peores.to_string(index=False))

# === GUARDAR MODELOS ===
print("\n=== GUARDANDO MODELOS ===")

with open('modelo_regresion.pkl', 'wb') as f:
    pickle.dump(modelo_regresion, f)
    
with open('modelo_ganador.pkl', 'wb') as f:
    pickle.dump(modelo_ganador, f)

with open('modelo_podio.pkl', 'wb') as f:
    pickle.dump(modelo_podio, f)
    
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({'equipo': le_equipo, 'piloto': le_piloto}, f)

with open('feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("✓ modelo_regresion.pkl")
print("✓ modelo_ganador.pkl")
print("✓ modelo_podio.pkl")
print("✓ label_encoders.pkl")
print("✓ feature_cols.pkl")

# Guardar predicciones para análisis
resultados_test.to_csv('predicciones_test.csv', index=False)
print("✓ predicciones_test.csv")

print("\n=== PROCESO COMPLETADO ===")
print(f"\n📊 RESUMEN:")
print(f"   MAE: {mae:.2f} posiciones")
print(f"   Accuracy Ganador: {accuracy_score(y_test_gano, pred_ganador):.1%}")
print(f"   Accuracy Podio: {accuracy_score(y_test_podio, pred_podio):.1%}")