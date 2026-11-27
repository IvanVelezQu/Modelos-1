import argparse
import os
import sys

import joblib
import pandas as pd

from src.utils import load_csv_file, validate_and_prepare_features


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Genera predicciones de tarifas cargando una versión almacenada del modelo."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al archivo modelo.joblib o al directorio de la versión (ej. models/v1/modelo.joblib).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta del archivo CSV con las variables predictoras de entrada.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ruta donde se guardará el archivo CSV con las predicciones.",
    )
    return parser.parse_args(args)


def resolve_model_path(model_arg: str) -> str:
    """Permite recibir tanto la ruta directa a modelo.joblib como la carpeta de la versión."""
    if os.path.isdir(model_arg):
        candidate = os.path.join(model_arg, "modelo.joblib")
    else:
        candidate = model_arg

    if not os.path.exists(candidate):
        raise FileNotFoundError(f"No se encontró el artefacto del modelo en: {candidate}")
    return candidate


def run_prediction(model_path_or_dir: str, input_csv: str, output_csv: str) -> pd.DataFrame:
    model_file = resolve_model_path(model_path_or_dir)
    df_input = load_csv_file(input_csv)

    # Validar columnas requeridas, tipos de datos y ordenar según esquema de entrenamiento
    X_valid = validate_and_prepare_features(df_input)

    try:
        pipeline = joblib.load(model_file)
    except Exception as exc:
        raise ValueError(f"Error al cargar el modelo '{model_file}': {exc}") from exc

    predictions = pipeline.predict(X_valid)

    # Conservar las filas originales en el mismo orden y agregar la columna 'prediccion'
    df_output = df_input.copy()
    df_output["prediccion"] = predictions

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_output.to_csv(output_csv, index=False)
    print(f"Predicciones generadas exitosamente ({len(df_output)} filas) en: {output_csv}")
    return df_output


def main():
    args = parse_args()
    try:
        run_prediction(
            model_path_or_dir=args.model,
            input_csv=args.input,
            output_csv=args.output,
        )
    except Exception as exc:
        print(f"ERROR en predicción: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
