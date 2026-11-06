# Fase 1: Modelo Predictivo - Predicción de Tarifas (Uber/Lyft)

# Fase 1: Modelo Predictivo - Predicción de Tarifas (Uber/Lyft)

Este directorio contiene los entregables corregidos de la **Fase 1** del Proyecto Integrador del curso Modelos y Simulación de Sistemas I.

## Autor
- Jorge Iván Vélez Quintero (`IvanVelezQu`)

## Descripción del Problema y Datos
El objetivo es predecir el precio (`price`) de viajes en plataformas de movilidad (Uber y Lyft) en Boston mediante un modelo de **regresión** utilizando las variables predictoras `distance`, `surge_multiplier`, `temperature`, `cab_type` y `name`.

- **Fuente del dataset:** [Kaggle - Uber and Lyft Dataset Boston, MA](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma)
- **Ubicación esperada:** Descargar el archivo `rideshare_kaggle.csv` y ubicarlo en `data/rideshare_kaggle.csv`.
- **Aclaración sobre valores faltantes:** Originalmente la variable `temperature` no presentaba valores faltantes tras limpiar los nulos de la variable objetivo (`price`). Para cumplir con el requisito académico (0.1 % a 2 % de nulos en una variable predictora), **estos nulos se generan artificialmente en el código** inyectando un **1.5 %** de valores faltantes aleatorios (`seed=42`), los cuales se imputan con la media dentro del pipeline de entrenamiento.

## Estructura de Archivos Corregida
- `Fase_1_integrador_modelos.ipynb`: Notebook ejecutable con el análisis exploratorio (EDA), inyección e imputación de nulos, comparación con modelo base y serialización.
- `modelo_rideshare.pkl`: Artefacto del pipeline entrenado (alojado en enlace externo documentado debido a su tamaño).

## Dependencias
Python 3.11+ con los paquetes: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` y `joblib`.

## Resultados y Comparación con Baseline
| Modelo | RMSE (USD) | R² |
| :--- | :---: | :---: |
| **Baseline (`DummyRegressor` - media)** | ~9.32 | 0.0000 |
| **Random Forest Regressor (`n_estimators=50`)** | **1.963** | **0.9558** |

Un RMSE de 1.963 USD indica que las predicciones se desvían en promedio ~1.96 dólares respecto a la tarifa real.

Puedes descargarlo desde el siguiente enlace:
[Descargar modelo_rideshare.pkl desde Google Drive](https://drive.google.com/drive/folders/1jTNGCb77BmaIJBxe4DCePaY-6ul8L-ni?usp=drive_link)

