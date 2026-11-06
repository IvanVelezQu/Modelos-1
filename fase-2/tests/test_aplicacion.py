import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from predict import run_prediction
from train import run_training


def _create_synthetic_dataset(filepath: str, n_rows: int = 40) -> pd.DataFrame:
    """Crea un dataset sintético pequeño con el esquema real del proyecto."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "distance": rng.uniform(0.5, 6.0, size=n_rows),
            "surge_multiplier": rng.choice([1.0, 1.25, 1.5], size=n_rows),
            "temperature": rng.uniform(30.0, 55.0, size=n_rows),
            "cab_type": rng.choice(["Uber", "Lyft"], size=n_rows),
            "name": rng.choice(["UberX", "UberXL", "Lyft", "Shared"], size=n_rows),
        }
    )
    df["price"] = (
        2.5
        + 3.2 * df["distance"] * df["surge_multiplier"]
        + rng.normal(0, 0.2, size=n_rows)
    )
    df.to_csv(filepath, index=False)
    return df


def test_entrenamiento_y_almacenamiento(tmp_path):
    """1. Verifica que train genera un artefacto cargable, métricas y metadatos."""
    data_csv = str(tmp_path / "datos_train.csv")
    models_dir = str(tmp_path / "models")
    _create_synthetic_dataset(data_csv, n_rows=40)

    result = run_training(
        data_path=data_csv,
        target_col="price",
        output_dir=models_dir,
        version="v_test",
        seed=42,
        n_estimators=10,
    )

    assert os.path.exists(result["model_path"])
    assert os.path.exists(result["metrics_path"])
    assert os.path.exists(result["metadata_path"])

    loaded_model = joblib.load(result["model_path"])
    assert hasattr(loaded_model, "predict")

    with open(result["metrics_path"], "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "model" in metrics and "baseline" in metrics
    assert "rmse" in metrics["model"] and "r2" in metrics["model"]

    with open(result["metadata_path"], "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["version"] == "v_test"
    assert metadata["target_variable"] == "price"


def test_carga_y_prediccion(tmp_path):
    """2. Verifica que el artefacto guardado predice por fila y conserva resultados."""
    data_csv = str(tmp_path / "datos_train.csv")
    models_dir = str(tmp_path / "models")
    input_csv = str(tmp_path / "nuevos.csv")
    output_csv = str(tmp_path / "predicciones.csv")

    df_train = _create_synthetic_dataset(data_csv, n_rows=40)
    result = run_training(
        data_path=data_csv,
        target_col="price",
        output_dir=models_dir,
        version="v1",
        seed=42,
        n_estimators=10,
    )

    df_input = df_train.drop(columns=["price"]).iloc[:6].copy()
    df_input.to_csv(input_csv, index=False)

    # Predicción directa antes/después de cargar
    pipeline = joblib.load(result["model_path"])
    expected_preds = pipeline.predict(df_input)

    df_out = run_prediction(result["model_path"], input_csv, output_csv)

    assert os.path.exists(output_csv)
    assert len(df_out) == len(df_input)
    assert "prediccion" in df_out.columns
    assert np.allclose(df_out["prediccion"].values, expected_preds, rtol=1e-5)


def test_valores_faltantes(tmp_path):
    """3. Verifica que entradas válidas con nulos se procesan sin eliminar filas."""
    data_csv = str(tmp_path / "datos_train.csv")
    models_dir = str(tmp_path / "models")
    input_csv = str(tmp_path / "nuevos_con_nulos.csv")
    output_csv = str(tmp_path / "predicciones_nulos.csv")

    _create_synthetic_dataset(data_csv, n_rows=40)
    result = run_training(
        data_path=data_csv,
        target_col="price",
        output_dir=models_dir,
        version="v1",
        seed=42,
        n_estimators=10,
    )

    df_input = pd.DataFrame(
        {
            "distance": [2.0, 3.5, 1.1],
            "surge_multiplier": [1.0, 1.25, 1.0],
            "temperature": [40.0, np.nan, np.nan],
            "cab_type": ["Uber", "Lyft", "Uber"],
            "name": ["UberX", "Lyft", "CategoriaNoVista"],
        }
    )
    df_input.to_csv(input_csv, index=False)

    df_out = run_prediction(result["model_path"], input_csv, output_csv)
    assert len(df_out) == 3
    assert not df_out["prediccion"].isna().any()


def test_entrada_invalida(tmp_path):
    """4. Verifica que la ausencia de una columna requerida produce error claro y código != 0."""
    data_csv = str(tmp_path / "datos_train.csv")
    models_dir = str(tmp_path / "models")
    bad_input_csv = str(tmp_path / "invalido.csv")
    output_csv = str(tmp_path / "salida_fallida.csv")

    _create_synthetic_dataset(data_csv, n_rows=30)
    result = run_training(
        data_path=data_csv,
        target_col="price",
        output_dir=models_dir,
        version="v1",
        seed=42,
        n_estimators=10,
    )

    # CSV sin la columna obligatoria 'distance'
    df_bad = pd.DataFrame(
        {
            "surge_multiplier": [1.0],
            "temperature": [42.0],
            "cab_type": ["Uber"],
            "name": ["UberX"],
        }
    )
    df_bad.to_csv(bad_input_csv, index=False)

    with pytest.raises(ValueError, match="distance"):
        run_prediction(result["model_path"], bad_input_csv, output_csv)

    # Probar desde línea de comandos que el código de salida sea != 0 y no genere archivo falso
    proc = subprocess.run(
        [
            sys.executable,
            "predict.py",
            "--model",
            result["model_path"],
            "--input",
            bad_input_csv,
            "--output",
            output_csv,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert not os.path.exists(output_csv)


def test_versionamiento(tmp_path):
    """5. Verifica que crear v2 conserva v1 y que repetir versión lanza error sin sobrescribir."""
    data_csv = str(tmp_path / "datos_train.csv")
    models_dir = str(tmp_path / "models")
    _create_synthetic_dataset(data_csv, n_rows=35)

    res_v1 = run_training(
        data_path=data_csv,
        target_col="price",
        output_dir=models_dir,
        version="v1",
        seed=42,
        n_estimators=10,
    )
    res_v2 = run_training(
        data_path=data_csv,
        target_col="price",
        output_dir=models_dir,
        version="v2",
        seed=42,
        n_estimators=15,
    )

    assert os.path.exists(res_v1["model_path"])
    assert os.path.exists(res_v2["model_path"])
    assert joblib.load(res_v1["model_path"]) is not None
    assert joblib.load(res_v2["model_path"]) is not None

    # Intentar sobrescribir v1 debe fallar con FileExistsError
    with pytest.raises(FileExistsError):
        run_training(
            data_path=data_csv,
            target_col="price",
            output_dir=models_dir,
            version="v1",
            seed=99,
            n_estimators=5,
        )
