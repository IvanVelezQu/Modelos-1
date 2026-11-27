# Fase 4: Monitoreo y Política de Reentrenamiento - Predicción de Tarifas (Uber/Lyft)

Este directorio contiene la implementación de la **Fase 4 (Monitoreo)** del Proyecto Integrador del curso **Modelos y Simulación de Sistemas I** (2026-II), conservando las carpetas `fase-1/`, `fase-2/` y `fase-3/`.

## 1. Registro Persistente de Solicitudes (`api.py`)
Cada solicitud enviada a `POST /predict` —incluyendo las rechazadas por validación mediante un manejador de `RequestValidationError`— agrega exactamente una fila en modo *append* al archivo persistente `logs/predicciones_log.csv` con las columnas:
- `fecha_hora` (UTC ISO-8601)
- `estado_http` (ej. `200`, `422`, `503`)
- `version` (identificador de la versión usada o vacío si no se invocó modelo)
- `prediccion` (valor estimado en USD o vacío si hubo error)

## 2. Inicio de la API con Registro Habilitado y Persistencia
### Ejecución local (desde `fase-4/`)
```bash
MODEL_VERSION=v2 PREDICTIONS_LOG_PATH=logs/predicciones_log.csv uvicorn api:app --host 0.0.0.0 --port 8000
```

### Ejecución en Docker montando modelos y carpeta de registro (`logs/`)
```bash
docker build -t modelos-fase4:1.0 .
mkdir -p logs

# En Bash (Linux / macOS / Git Bash)
docker run --rm -p 8000:8000 \
  -e MODEL_VERSION=v2 \
  -e PREDICTIONS_LOG_PATH=/app/logs/predicciones_log.csv \
  -v "$PWD/../fase-2/data:/app/data:ro" \
  -v "$PWD/../fase-2/models:/app/models" \
  -v "$PWD/logs:/app/logs" \
  modelos-fase4:1.0
```

```powershell
# En Windows PowerShell
docker run --rm -p 8000:8000 `
  -e MODEL_VERSION=v2 `
  -e PREDICTIONS_LOG_PATH=/app/logs/predicciones_log.csv `
  -v "${PWD}/../fase-2/data:/app/data:ro" `
  -v "${PWD}/../fase-2/models:/app/models" `
  -v "${PWD}/logs:/app/logs" `
  modelos-fase4:1.0
```

## 3. Cálculo de Métricas de Operación (`monitoreo.py`)
Para consultar los indicadores operativos sobre el registro persistente:
```bash
python monitoreo.py --log-file logs/predicciones_log.csv
```
- Cuenta las filas de datos como el **total de solicitudes**.
- Cuenta las filas con `estado_http` entre `400` y `599` para calcular el **porcentaje de solicitudes fallidas**.
- Si el registro está vacío, muestra `Total de solicitudes: 0` y `sin solicitudes` para el porcentaje.

## 4. Gestión de Versiones y Reversión (Rollback)
Se reutilizan las versiones entrenadas y documentadas en las fases anteriores (disponibles en `models/v1/` y `models/v2/`, además de `v3` generada al validar `POST /train` en la Fase 3):
- **Versión activa configurada:** `v2` (`RandomForestRegressor`, `n_estimators=30`, `max_depth=15`, `RMSE = 1.7582`, `R² = 0.9645`).
- **Versión anterior conservada:** `v1` (`RandomForestRegressor`, `n_estimators=50`, `max_depth=None`, `RMSE = 1.9632`, `R² = 0.9558`).
- **Cómo volver a la versión anterior (`v1`):** No se requiere modificar código; basta con reiniciar el servidor cambiando la variable de entorno `MODEL_VERSION` a `v1`:
  ```bash
  # En local:
  MODEL_VERSION=v1 uvicorn api:app --host 0.0.0.0 --port 8000
  # En Docker: reemplazar -e MODEL_VERSION=v2 por -e MODEL_VERSION=v1
  ```

## Reporte

### Solicitudes enviadas en el ensayo (sobre registro vacío)
1. **Solicitud 1 (válida):** `{"distance": 2.5, "surge_multiplier": 1.0, "temperature": 42.0, "cab_type": "Uber", "name": "UberX"}` → `HTTP 200` (versión `v2`).
2. **Solicitud 2 (válida):** `{"distance": 3.8, "surge_multiplier": 1.0, "temperature": 38.5, "cab_type": "Uber", "name": "Black SUV"}` → `HTTP 200` (versión `v2`).
3. **Solicitud 3 (inválida — falta campo obligatorio `distance`):** `{"surge_multiplier": 1.25, "temperature": 40.0, "cab_type": "Lyft", "name": "Lyft"}` → `HTTP 422`.

### Muestra del registro generado (`fase-4/logs/predicciones_log.csv`)
```csv
fecha_hora,estado_http,version,prediccion
2026-11-27T04:41:27,200,v2,10.5543
2026-11-27T04:41:28,200,v2,36.1654
2026-11-27T04:41:28,422,,
```

### Indicadores de operación obtenidos
- **Total de solicitudes:** 3
- **Porcentaje de solicitudes fallidas (4xx/5xx):** 33.3% (1/3)

**Interpretación:** La API procesó dos registros válidos utilizando la versión `v2` y rechazó el tercero con código `HTTP 422` porque faltaba la variable obligatoria `distance`, produciendo una tasa de fallos de 1/3 (33.3%), resultado esperado del control de validación de entradas.

### Versiones documentadas
- **Versión activa:** `v2`
- **Versión anterior disponible para reversión:** `v1` (y versión `v3` generada durante la validación de `POST /train` en la Fase 3).

### Política de reentrenamiento
Al acumular 500 viajes nuevos con su tarifa real cobrada (o al cumplirse un periodo mensual de operación), el responsable del proyecto ejecutará `POST /train` para generar una versión candidata sin sobrescribir `v2` ni `v1`. La candidata y la versión activa (`v2`) se compararán con la métrica **RMSE** sobre un mismo conjunto reservado de viajes recientes, y se activará la nueva versión mediante `MODEL_VERSION` únicamente si reduce el RMSE frente a la versión vigente.

## 5. Identificación de la Entrega
- **Etiqueta de Git (Tag):** `entrega-3-4` (integrada en la rama `main` desde `develop`).
