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
