# Fase 2: Scripts, Versionamiento, Pruebas y Docker - Predicción de Tarifas (Uber/Lyft)

Este directorio contiene la implementación de la **Fase 2** del Proyecto Integrador del curso **Modelos y Simulación de Sistemas I** (2026-II).

## 1. Autor y Relación con la Fase 1
- **Integrante:** Jorge Iván Vélez Quintero (`IvanVelezQu`)
- **Problema:** Regresión para estimar el precio (`price` en USD) de viajes de transporte compartido (Uber y Lyft) en Boston a partir de variables operativas y meteorológicas (`distance`, `surge_multiplier`, `temperature`, `cab_type`, `name`).
- **Continuidad con Fase 1:** Se conserva la carpeta `fase-1/` y se transforma el flujo del notebook en una aplicación modular ejecutable desde línea de comandos (`train.py` y `predict.py`) y contenedor Docker.

## 2. Correcciones Aplicadas Respecto a la Entrega 1
1. **Análisis Exploratorio (EDA) y Correlaciones:** Se amplió el notebook de la Fase 1 con estadísticas descriptivas, distribución de `price`, gráficos de dispersión y matriz de correlación.
2. **Aclaración y Justificación de Valores Faltantes:** Tras depurar las filas sin variable objetivo (`price`), el dataset original no tenía nulos en las predictoras. Para cumplir el requisito del curso (0.1 % a 2 % de nulos), el código de entrenamiento inyecta de forma reproducible (`seed=42`) un **1.5 % de valores faltantes artificiales** en `temperature`, imputados mediante la media (`SimpleImputer(strategy='mean')`) dentro del pipeline de entrenamiento para evitar fuga de información.
3. **Inclusión del Modelo Base (Baseline):** Se incorporó `DummyRegressor(strategy="mean")` evaluado sobre la misma partición de prueba.
4. **Rutas Reproducibles:** Se eliminaron las rutas personales de Google Drive del código, parametrizando todas las rutas mediante argumentos CLI (`argparse`).

## 3. Entorno y Dependencias
- **Versión de Python:** Python 3.11+ (probado en 3.11 / 3.12).
- **Instalación local (desde la carpeta `fase-2`):**

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Acceso a los Datos y Esquema de Entrada
- **Dataset de entrenamiento:** Descargar `rideshare_kaggle.csv` desde [Kaggle: Uber and Lyft Dataset Boston, MA](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma) y ubicarlo en `fase-2/data/rideshare_kaggle.csv`.
- **Archivo de ejemplo para predicción:** Incluido en `fase-2/data/ejemplo_prediccion.csv` (no requiere la columna objetivo `price`).
- **Esquema requerido para predicción:**

| Columna | Tipo esperado | Descripción | Tratamiento de nulos / no vistos |
| :--- | :--- | :--- | :--- |
| `distance` | `float64` | Distancia del viaje en millas | Imputación con la media de entrenamiento |
| `surge_multiplier` | `float64` | Multiplicador de tarifa dinámica | Imputación con la media de entrenamiento |
| `temperature` | `float64` | Temperatura en °F | Imputación con la media de entrenamiento |
| `cab_type` | `string` | Plataforma (`Uber` o `Lyft`) | Moda + `OneHotEncoder(handle_unknown='ignore')` |
| `name` | `string` | Categoría del vehículo (`UberX`, `Lyft`, etc.) | Moda + `OneHotEncoder(handle_unknown='ignore')` |

*(Las columnas adicionales se ignoran en el modelo pero se conservan en el CSV de salida junto con la columna `prediccion` manteniendo el orden original de las filas).*

## 5. Ejecución desde Línea de Comandos (Local)
Ubicado en la terminal dentro de la carpeta `fase-2/`:

### Consultar ayuda (`--help`)
```bash
python train.py --help
python predict.py --help
```

### Entrenar versiones del modelo (`train.py`)
```bash
# Versión v1 (configuración base Fase 1: 50 árboles)
python train.py --data data/rideshare_kaggle.csv --target price --output-dir models --version v1 --seed 42 --n-estimators 50

# Versión v2 (30 árboles y profundidad máxima 15)
python train.py --data data/rideshare_kaggle.csv --target price --output-dir models --version v2 --seed 42 --n-estimators 30 --max-depth 15
```

### Generar predicciones (`predict.py`)
```bash
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_v1.csv
python predict.py --model models/v2/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_v2.csv
```

## 6. Ubicación de Versiones, Artefactos y Enlace Externo
Cada versión genera en `fase-2/models/<version>/`:
- `modelo.joblib`: Pipeline completo (`ColumnTransformer` + `RandomForestRegressor`).
- `metrics.json`: Métricas de evaluación en prueba del modelo y del baseline.
- `metadata.json`: Metadatos de versión, fecha UTC, columnas, tipos, hiperparámetros, filas de entrenamiento/prueba y hash del commit.

Los archivos `metrics.json` y `metadata.json` de `v1` y `v2` están incluidos directamente en el repositorio (`fase-2/models/v1/` y `fase-2/models/v2/`). Debido al límite de tamaño de GitHub, los binarios `modelo.joblib` pueden descargarse desde Google Drive:
- **Enlace a los artefactos `modelo.joblib` (`v1` y `v2`):** https://drive.google.com/drive/folders/1jTNGCb77BmaIJBxe4DCePaY-6ul8L-ni?usp=drive_link

## 7. Ejecución con Docker y Persistencia
Desde la carpeta `fase-2/`:

### Construcción de la imagen
```bash
docker build -t modelos-fase2:1.0 .
```

### Ejecución en Terminal Bash (Linux / macOS / Git Bash)
```bash
mkdir -p models outputs

docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" modelos-fase2:1.0 python train.py --data data/rideshare_kaggle.csv --target price --output-dir models --version v-docker --seed 42 --n-estimators 30 --max-depth 15

docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```

### Ejecución en Windows PowerShell
```powershell
mkdir -Force models, outputs

docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models" modelos-fase2:1.0 python train.py --data data/rideshare_kaggle.csv --target price --output-dir models --version v-docker --seed 42 --n-estimators 30 --max-depth 15

docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models:ro" -v "${PWD}/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```

## 8. Pruebas Automatizadas con `pytest`
Desde `fase-2/`, ejecutar:
```bash
python -m pytest -v
```
El archivo `tests/test_aplicacion.py` implementa las 5 pruebas requeridas usando conjuntos sintéticos pequeños y directorios temporales (`tmp_path`):
1. `test_entrenamiento_y_almacenamiento`: Comprueba la creación y carga de `modelo.joblib`, `metrics.json` y `metadata.json`.
2. `test_carga_y_prediccion`: Comprueba que el modelo cargado genera una predicción por fila idéntica al modelo en memoria.
3. `test_valores_faltantes`: Comprueba la imputación de nulos en `temperature` y el manejo de categorías no vistas sin eliminar filas.
4. `test_entrada_invalida`: Comprueba que la ausencia de una columna obligatoria (`distance`) lanza error y finaliza con código de salida distinto de cero (`returncode != 0`).
5. `test_versionamiento`: Comprueba la coexistencia de `v1` y `v2` y que un identificador repetido lanza `FileExistsError` sin sobrescribir archivos.

## 9. Métricas Obtenidas, Comparación con Baseline y Limitaciones
Partición: 510.380 observaciones de entrenamiento y 127.596 de prueba (`test_size=0.2`, `seed=42`).

| Versión / Modelo | RMSE (USD) | MAE (USD) | R² |
| :--- | :---: | :---: | :---: |
| **Baseline (`DummyRegressor` - media)** | 9.3370 | 7.5598 | -0.0000 |
| **Versión `v1` (`RandomForestRegressor`, `n_estimators=50`, `max_depth=None`)** | 1.9632 | 1.2767 | 0.9558 |
| **Versión `v2` (`RandomForestRegressor`, `n_estimators=30`, `max_depth=15`)** | **1.7582** | **1.1520** | **0.9645** |

- **Interpretación:** La versión `v2` reduce el error promedio (RMSE) a **1.7582 USD** (frente a 9.3370 USD del modelo base), explicando el 96.45 % de la varianza del precio.
- **Limitaciones conocidas:** Los árboles de decisión no extrapolan tendencias fuera del rango observado de distancia o multiplicador dinámico, y el modelo está entrenado únicamente con datos de Boston sin variables de tráfico en tiempo real.

## 10. Identificación de la Entrega
- **Etiqueta de Git (Tag):** `entrega-2` (integrada en la rama `main` desde `develop`).
