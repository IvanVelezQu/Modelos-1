# Fase 3: API REST - Predicción de Tarifas (Uber/Lyft)

Este directorio contiene la implementación de la **Fase 3 (API REST)** del Proyecto Integrador del curso **Modelos y Simulación de Sistemas I** (2026-II).

## 1. Descripción y Reutilización de la Fase 2
Se implementa una API REST con **FastAPI** (`api.py`) que reutiliza el esquema de variables, el pipeline de preprocesamiento (`src/utils.py`) y la lógica de entrenamiento (`train.py`) desarrollados en la Fase 2.

### Operaciones disponibles
- **`GET /health`**: Devuelve `HTTP 200` y un JSON confirmando que la API está en ejecución junto con la versión activa configurada.
- **`POST /predict`**: Recibe un registro JSON con las variables predictoras (`distance`, `surge_multiplier`, `temperature`, `cab_type`, `name`), carga la versión configurada, aplica las transformaciones guardadas y devuelve `HTTP 200` con la predicción y la versión utilizada (`{"prediccion": 10.7962, "version": "v1"}`).
  - Admite `null` en `temperature` aplicando la regla de imputación por la media almacenada en el pipeline.
  - Si falta una variable obligatoria o el tipo es incompatible, responde con `HTTP 422` identificando el campo.
  - Si el artefacto del modelo configurado no está disponible, responde con `HTTP 503`.
- **`POST /train`**: Entrena una nueva versión con los datos y configuración del proyecto sin sobrescribir las versiones anteriores (`v1`, `v2`, etc.), guarda `modelo.joblib`, `metrics.json` y `metadata.json`, y devuelve `HTTP 201` con el nuevo identificador (por ejemplo, `v3`).

## 2. Selección de Versión del Modelo
La versión utilizada por `POST /predict` se controla mediante la variable de entorno `MODEL_VERSION` (por defecto `v1`):
- **En local (Bash):** `MODEL_VERSION=v2 uvicorn api:app --host 0.0.0.0 --port 8000`
- **En Docker:** agregar `-e MODEL_VERSION=v2` al comando `docker run`.

## 3. Ejecución con Docker
Ubicado dentro de la carpeta `fase-3/`:

### Construir la imagen
```bash
docker build -t modelos-fase3:1.0 .
```

### Iniciar el contenedor publicando el puerto 8000 y montando datos y modelos
- **En Bash (Linux / macOS / Git Bash):**
```bash
docker run --rm -p 8000:8000 \
  -e MODEL_VERSION=v1 \
  -v "$PWD/../fase-2/data:/app/data:ro" \
  -v "$PWD/../fase-2/models:/app/models" \
  modelos-fase3:1.0
```

- **En Windows PowerShell:**
```powershell
docker run --rm -p 8000:8000 `
  -e MODEL_VERSION=v1 `
  -v "${PWD}/../fase-2/data:/app/data:ro" `
  -v "${PWD}/../fase-2/models:/app/models" `
  modelos-fase3:1.0
```

## 4. Cliente de Prueba y Ejemplo de Solicitud / Respuesta
Con el servidor en ejecución en `http://localhost:8000`, ejecutar el cliente incluido:
```bash
python cliente.py --url http://localhost:8000
```

También puede consultarse con `curl`:
```bash
# 1. Estado del servicio
curl http://localhost:8000/health

# 2. Predicción con un registro del proyecto
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance": 2.5, "surge_multiplier": 1.0, "temperature": 42.0, "cab_type": "Uber", "name": "UberX"}'

# 3. Entrenar una nueva versión
curl -X POST http://localhost:8000/train
```

### Ejemplo real de solicitud y respuesta en `POST /predict`
**Solicitud:**
```json
{
  "distance": 2.5,
  "surge_multiplier": 1.0,
  "temperature": 42.0,
  "cab_type": "Uber",
  "name": "UberX"
}
```

**Respuesta (`HTTP 200 OK`):**
```json
{
  "prediccion": 10.7962,
  "version": "v1"
}
```

## 5. Ejecución de Pruebas con `pytest`
Desde la carpeta `fase-3/`:
```bash
python -m pytest -v
```
El archivo `tests/test_api.py` verifica los tres casos requeridos:
1. `test_health_responde_200`: `GET /health` responde `200` e informa el estado `ok`.
2. `test_predict_entrada_valida`: `POST /predict` devuelve `200`, `prediccion` numérica y `version`.
3. `test_predict_campo_obligatorio_ausente`: `POST /predict` sin `distance` se rechaza con `422` indicando el campo faltante.
