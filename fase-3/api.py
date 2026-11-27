import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr
from typing import Union

from src.utils import REQUIRED_FEATURES, validate_and_prepare_features
from train import run_training

app = FastAPI(
    title="API REST - Predicción de Tarifas Uber/Lyft",
    description="Fase 3 del Proyecto Integrador - Modelos y Simulación de Sistemas I",
    version="1.0.0",
)


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


def get_next_available_version(models_dir: str) -> str:
    """Genera un identificador de versión nuevo (v3, v4, ...) sin sobrescribir las anteriores."""
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
    """Devuelve HTTP 200 indicando que la API está en ejecución."""
    active_version = os.getenv("MODEL_VERSION", "v1")
    return {
        "status": "ok",
        "message": "API en ejecución",
        "configured_version": active_version,
    }


@app.post("/predict")
def predict_fare(payload: RidePredictionInput):
    """
    Carga la versión configurada del modelo, aplica las transformaciones guardadas
    y devuelve la predicción junto con la versión utilizada.
    """
    models_dir = os.getenv("MODELS_DIR", "models")
    version = os.getenv("MODEL_VERSION", "v1")
    model_path = os.path.join(models_dir, version, "modelo.joblib")

    if not os.path.exists(model_path):
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
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo cargar el artefacto del modelo '{version}': {exc}",
        ) from exc

    # Construir tabla de una fila con las columnas en el orden del esquema
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
        pred_value = float(pipeline.predict(X_valid)[0])
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "prediccion": round(pred_value, 4),
        "version": version,
    }


@app.post("/train", status_code=201)
def train_new_version():
    """
    Entrena una nueva versión con los datos y configuración de la Fase 2,
    conservando las versiones anteriores, y devuelve su identificador.
    """
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

    try:
        result = run_training(
            data_path=data_path,
            target_col=target_col,
            output_dir=models_dir,
            version=new_version,
            seed=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el entrenamiento de la nueva versión: {exc}",
        ) from exc

    return {
        "status": "trained",
        "version": new_version,
        "metrics": result["metrics"]["model"],
        "model_path": result["model_path"],
    }
