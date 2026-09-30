# Fase 1: Modelo Predictivo - Predicción de Tarifas (Uber/Lyft)

Este directorio contiene los entregables correspondientes a la **Fase 1** del Proyecto Integrador del curso Modelos y Simulación de Sistemas I.

## Equipo de Trabajo
- Jorge Iván Vélez Quintero (GitHub: IvanVelezQu)

##  Descripción del Problema
El objetivo de este proyecto es predecir el precio (`price`) de los servicios de transporte compartido (Uber y Lyft) en la ciudad de Boston. Para ello, se ha planteado un problema de **regresión** utilizando características como la distancia del viaje, el multiplicador de tarifa dinámica, las condiciones meteorológicas (temperatura) y el tipo de servicio.

El conjunto de datos original fue obtenido desde [Kaggle: Uber and Lyft Dataset Boston, MA](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma).

## Cumplimiento de Requisitos del Dataset
- **Observaciones:** > 1.000 registros.
- **Variables predictoras:** > 5 (incluyendo `distance`, `surge_multiplier`, `temperature`, `cab_type`, `name`).
- **Valores Nulos:** La variable predictora `temperature` contiene un **1.4999%** de valores faltantes (cumpliendo el rango exigido del 0.1% al 2%).
- **Series de tiempo:** El enfoque es transversal, no secuencial.

## Estructura del Directorio
- `modelo.ipynb`: Notebook de Jupyter completamente ejecutable que detalla el proceso de exploración de datos (EDA), preprocesamiento, prevención de fuga de información y entrenamiento del modelo.
- `modelo_rideshare.pkl`: Archivo serializado que contiene el pipeline de preprocesamiento y el modelo de Machine Learning entrenado (Random Forest Regressor).

## Preprocesamiento y Prevención de Fuga de Información (Data Leakage)
Para garantizar la integridad del modelo y evitar fugas de información, se implementó la división de datos (`train_test_split`) **antes** de aplicar cualquier transformación. 

Posteriormente, se diseñó un `Pipeline` de Scikit-Learn que integra:
1. **Datos Numéricos:** Imputación de nulos con la media y estandarización (`StandardScaler`).
2. **Datos Categóricos:** Imputación con el valor más frecuente y codificación *One-Hot* (`OneHotEncoder`).

## Modelo y Evaluación
Se utilizó un modelo **Random Forest Regressor** ensamblado dentro del pipeline. 
*(Nota: Añadan aquí los resultados de las métricas RMSE y R2 que les arrojó el notebook en la celda de evaluación).*

## Instrucciones de Ejecución
Para reproducir el entrenamiento del modelo:
1. Clonar el repositorio.
2. Descargar el archivo de datos original desde Kaggle y ubicarlo en la ruta correspondiente (o conectar Google Drive si se usa Colab).
3. Instalar las dependencias (Scikit-learn, Pandas, Numpy).
4. Ejecutar todas las celdas del archivo `modelo.ipynb`.
