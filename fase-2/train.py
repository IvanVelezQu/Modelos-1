import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone

import joblib
from sklearn.model_selection import train_test_split

from src.utils import (
    CATEGORICAL_FEATURES,
    EXPECTED_TYPES,
    NUMERIC_FEATURES,
    REQUIRED_FEATURES,
    build_baseline_model,
    build_model_pipeline,
    compute_regression_metrics,
    get_git_commit_hash,
    inject_controlled_missing_values,
    load_csv_file,
    validate_and_prepare_features,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de predicción de tarifas de viajes y almacena una versión."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Ruta del archivo CSV con datos etiquetados.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="price",
        help="Nombre de la variable objetivo (por defecto: price).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directorio base donde se guardarán las versiones (por defecto: models).",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Identificador único de la versión del modelo (ej. v1, v2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (por defecto: 42).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción del conjunto de prueba entre 0 y 1 (por defecto: 0.2).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=50,
        help="Número de árboles en RandomForestRegressor (por defecto: 50).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Profundidad máxima de los árboles (opcional).",
    )
    return parser.parse_args(args)


def run_training(
    data_path: str,
    target_col: str,
    output_dir: str,
    version: str,
    seed: int = 42,
    test_size: float = 0.2,
    n_estimators: int = 50,
    max_depth: int = None,
) -> dict:
    version_dir = os.path.join(output_dir, version)
    if os.path.exists(version_dir) and os.listdir(version_dir):
        raise FileExistsError(
            f"La versión '{version}' ya existe en '{version_dir}'. "
            "No se permite sobrescribir versiones existentes. Use un identificador nuevo."
        )

    df = load_csv_file(data_path)

    if target_col not in df.columns:
        raise ValueError(
            f"La variable objetivo '{target_col}' no existe en el archivo '{data_path}'."
        )

    # Eliminar filas donde la variable objetivo sea nula
    df = df.dropna(subset=[target_col]).copy()
    if len(df) < 5:
        raise ValueError("El conjunto de datos tiene muy pocas observaciones válidas para entrenar.")

    # Asegurar el 1.5% de nulos controlados en temperature si no existen originalmente
    df = inject_controlled_missing_values(df, col="temperature", ratio=0.015, seed=seed)

    X = validate_and_prepare_features(df)
    y = df[target_col].astype(float)

    # Separar entrenamiento y prueba antes de ajustar cualquier transformación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # 1. Entrenar y evaluar el modelo base (Baseline)
    baseline = build_baseline_model()
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    baseline_metrics = compute_regression_metrics(y_test, y_pred_base)

    # 2. Entrenar y evaluar el pipeline principal (Random Forest)
    pipeline = build_model_pipeline(
        n_estimators=n_estimators, max_depth=max_depth, seed=seed
    )
    pipeline.fit(X_train, y_train)
    y_pred_model = pipeline.predict(X_test)
    model_metrics = compute_regression_metrics(y_test, y_pred_model)

    # Crear directorio de la versión y guardar artefactos
    os.makedirs(version_dir, exist_ok=True)
    model_filepath = os.path.join(version_dir, "modelo.joblib")
    metrics_filepath = os.path.join(version_dir, "metrics.json")
    metadata_filepath = os.path.join(version_dir, "metadata.json")

    joblib.dump(pipeline, model_filepath)

    metrics_payload = {
        "version": version,
        "evaluation_partition": "test",
        "model": model_metrics,
        "baseline": baseline_metrics,
    }
    with open(metrics_filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4, ensure_ascii=False)

    metadata_payload = {
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "problem_type": "regression",
        "target_variable": target_col,
        "algorithm": "RandomForestRegressor",
        "baseline_algorithm": "DummyRegressor(strategy='mean')",
        "input_features": {
            "ordered_columns": REQUIRED_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "expected_types": EXPECTED_TYPES,
        },
        "model_parameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": seed,
        },
        "missing_values_strategy": {
            "numeric": "SimpleImputer(strategy='mean')",
            "categorical": "SimpleImputer(strategy='most_frequent')",
            "note": "Incluye 1.5% de valores faltantes controlados en 'temperature' durante entrenamiento.",
        },
        "data_info": {
            "source": "Kaggle - Uber and Lyft Dataset Boston, MA",
            "file_path": data_path,
            "total_valid_observations": int(len(df)),
            "train_observations": int(len(X_train)),
            "test_observations": int(len(X_test)),
            "split_strategy": "train_test_split (random_state fijo)",
            "test_size_ratio": test_size,
        },
        "evaluation_metrics": ["rmse", "mae", "r2"],
        "environment": {
            "python_version": platform.python_version(),
            "dependencies_file": "requirements.txt",
            "git_commit_hash": get_git_commit_hash(),
        },
    }
    with open(metadata_filepath, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=4, ensure_ascii=False)

    print("=== RESUMEN DE ENTRENAMIENTO ===")
    print(f"Versión generada   : {version}")
    print(f"Filas entrenamiento: {len(X_train)} | Filas prueba: {len(X_test)}")
    print(
        f"Métricas Modelo    : RMSE={model_metrics['rmse']:.4f}, "
        f"MAE={model_metrics['mae']:.4f}, R2={model_metrics['r2']:.4f}"
    )
    print(
        f"Métricas Baseline  : RMSE={baseline_metrics['rmse']:.4f}, "
        f"MAE={baseline_metrics['mae']:.4f}, R2={baseline_metrics['r2']:.4f}"
    )
    print(f"Modelo guardado en : {model_filepath}")
    print(f"Métricas en        : {metrics_filepath}")
    print(f"Metadatos en       : {metadata_filepath}")

    return {
        "model_path": model_filepath,
        "metrics_path": metrics_filepath,
        "metadata_path": metadata_filepath,
        "metrics": metrics_payload,
        "metadata": metadata_payload,
    }


def main():
    args = parse_args()
    try:
        run_training(
            data_path=args.data,
            target_col=args.target,
            output_dir=args.output_dir,
            version=args.version,
            seed=args.seed,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
    except Exception as exc:
        print(f"ERROR en entrenamiento: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
