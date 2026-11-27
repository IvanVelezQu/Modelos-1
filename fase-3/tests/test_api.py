import os
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from train import run_training

client = TestClient(app)


@pytest.fixture()
def small_model_env(tmp_path, monkeypatch):
    """Crea y configura un modelo ligero en un directorio temporal para las pruebas."""
    data_csv = str(tmp_path / "datos_test.csv")
    models_dir = str(tmp_path / "models")

    rng = np.random.default_rng(42)
    n_rows = 40
    df = pd.DataFrame(
        {
            "distance": rng.uniform(0.5, 5.0, size=n_rows),
            "surge_multiplier": rng.choice([1.0, 1.25, 1.5], size=n_rows),
            "temperature": rng.uniform(32.0, 55.0, size=n_rows),
            "cab_type": rng.choice(["Uber", "Lyft"], size=n_rows),
            "name": rng.choice(["UberX", "UberXL", "Lyft", "Shared"], size=n_rows),
        }
    )
    df["price"] = 2.5 + 3.1 * df["distance"] * df["surge_multiplier"]
    df.to_csv(data_csv, index=False)

    run_training(
        data_path=data_csv,
        target_col="price",
        output_dir=models_dir,
        version="v1",
        seed=42,
        n_estimators=5,
    )

    monkeypatch.setenv("MODELS_DIR", models_dir)
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.setenv("TRAIN_DATA_PATH", data_csv)
    return models_dir


def test_health_responde_200():
    """1. Verifica que GET /health responde HTTP 200 e informa el estado."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_predict_entrada_valida(small_model_env):
    """2. Verifica que una entrada válida devuelve HTTP 200, predicción y versión."""
    registro_valido = {
        "distance": 2.5,
        "surge_multiplier": 1.0,
        "temperature": 42.0,
        "cab_type": "Uber",
        "name": "UberX",
    }
    response = client.post("/predict", json=registro_valido)
    assert response.status_code == 200
    body = response.json()
    assert "prediccion" in body
    assert isinstance(body["prediccion"], (int, float))
    assert body["version"] == "v1"


def test_predict_campo_obligatorio_ausente(small_model_env):
    """3. Verifica que una entrada sin un campo obligatorio se rechaza e identifica la clave."""
    registro_incompleto = {
        # Se omite la clave obligatoria 'distance'
        "surge_multiplier": 1.0,
        "temperature": 42.0,
        "cab_type": "Uber",
        "name": "UberX",
    }
    response = client.post("/predict", json=registro_incompleto)
    assert response.status_code in (400, 422)
    assert "distance" in response.text
