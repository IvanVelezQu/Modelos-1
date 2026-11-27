import csv
import os
from datetime import datetime, timezone
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from src.utils import REQUIRED_FEATURES, validate_and_prepare_features
from train import run_training

app = FastAPI(
    title="API REST con Monitoreo - Predicción de Tarifas Uber/Lyft",
    description="Fase 4 del Proyecto Integrador - Modelos y Simulación de Sistemas I",
    version="2.0.0",
)

LOG_HEADERS = ["fecha_hora", "estado_http", "version", "prediccion"]


class RidePredictionInput(BaseModel):
    """Esquema de entrada con las 5 variables predictoras del proyecto."""
    model_config = ConfigDict(extra="ignore")

    distance: Union[StrictFloat, StrictInt] = Field(
        ..., description="Distancia del viaje en millas"
    )
    surge_multiplier: Union[StrictFloat, StrictInt] = Field(
        ..., description="Multiplicador de tarifa dinámica"
    )
    temperature: Optional[Union[StrictFloat, StrictInt]] = Field(
        ..., description="Temperatura en °F (admite null para imputación)"
    )
    cab_type: StrictStr = Field(
        ..., description="Plataforma de transporte (Uber o Lyft)"
    )
    name: StrictStr = Field(
        ..., description="Categoría del servicio (ej. UberX, Lyft, Black SUV)"
    )


def get_log_file_path() -> str:
    return os.getenv("PREDICTIONS_LOG_PATH", "logs/predicciones_log.csv")


def append_prediction_log(
    estado_http: int, version: str = "", prediccion: Union[float, str] = ""
) -> None:
    """Agrega una fila al archivo CSV de registro conservando el historial."""
    log_path = get_log_file_path()
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(LOG_HEADERS)
        writer.writerow([timestamp, estado_http, version, prediccion])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Registra en el CSV los rechazos de validación (HTTP 422) dirigidos a /predict."""
    if request.url.path == "/predict":
        append_prediction_log(estado_http=422, version="", prediccion="")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def get_next_available_version(models_dir: str) -> str:
    os.makedirs(models_dir, exist_ok=True)
    idx = 1
    while True:
        candidate = f"v{idx}"
        version_path = os.path.join(models_dir, candidate)
        if not os.path.exists(version_path) or not os.listdir(version_path):
            return candidate
        idx += 1


@app.get("/health")
def health_check():
    active_version = os.getenv("MODEL_VERSION", "v2")
    return {
        "status": "ok",
        "message": "API con monitoreo en ejecución",
        "configured_version": active_version,
        "log_file": get_log_file_path(),
    }


@app.post("/predict")
def predict_fare(payload: RidePredictionInput):
    models_dir = os.getenv("MODELS_DIR", "models")
    version = os.getenv("MODEL_VERSION", "v2")
    model_path = os.path.join(models_dir, version, "modelo.joblib")

    if not os.path.exists(model_path):
        append_prediction_log(estado_http=503, version="", prediccion="")
        raise HTTPException(
            status_code=503,
            detail=(
                f"El modelo configurado para la versión '{version}' no está disponible "
                f"en '{model_path}'."
            ),
        )

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        append_prediction_log(estado_http=503, version="", prediccion="")
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo cargar el artefacto del modelo '{version}': {exc}",
        ) from exc

    record = {
        "distance": float(payload.distance),
        "surge_multiplier": float(payload.surge_multiplier),
        "temperature": np.nan if payload.temperature is None else float(payload.temperature),
        "cab_type": payload.cab_type,
        "name": payload.name,
    }
    df_row = pd.DataFrame([record])[REQUIRED_FEATURES]

    try:
        X_valid = validate_and_prepare_features(df_row)
        pred_value = round(float(pipeline.predict(X_valid)[0]), 4)
    except (ValueError, TypeError) as exc:
        append_prediction_log(estado_http=400, version=version, prediccion="")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    append_prediction_log(estado_http=200, version=version, prediccion=pred_value)
    return {
        "prediccion": pred_value,
        "version": version,
    }


@app.post("/train", status_code=201)
def train_new_version():
    models_dir = os.getenv("MODELS_DIR", "models")
    data_path = os.getenv("TRAIN_DATA_PATH", "data/rideshare_kaggle.csv")
    target_col = os.getenv("TARGET_COL", "price")
    seed = int(os.getenv("TRAIN_SEED", "42"))
    n_estimators = int(os.getenv("TRAIN_N_ESTIMATORS", "25"))
    max_depth_env = os.getenv("TRAIN_MAX_DEPTH", "15")
    max_depth = int(max_depth_env) if max_depth_env and max_depth_env != "None" else None

    if not os.path.exists(data_path):
        raise HTTPException(
            status_code=400,
            detail=f"No se encontró el archivo de datos para entrenar en '{data_path}'.",
        )

    new_version = get_next_available_version(models_dir)
    result = run_training(
        data_path=data_path,
        target_col=target_col,
        output_dir=models_dir,
        version=new_version,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    return {
        "status": "trained",
        "version": new_version,
        "metrics": result["metrics"]["model"],
        "model_path": result["model_path"],
    }
