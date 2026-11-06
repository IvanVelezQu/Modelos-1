import os
import subprocess
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["distance", "surge_multiplier", "temperature"]
CATEGORICAL_FEATURES = ["cab_type", "name"]
REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXPECTED_TYPES = {
    "distance": "float64",
    "surge_multiplier": "float64",
    "temperature": "float64",
    "cab_type": "string/categorical",
    "name": "string/categorical",
}


def load_csv_file(filepath: str) -> pd.DataFrame:
    """Lee un archivo CSV verificando su existencia y formato."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo no existe: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        raise ValueError(f"No se pudo leer el archivo CSV '{filepath}': {exc}") from exc

    if df.empty:
        raise ValueError(f"El archivo CSV '{filepath}' está vacío.")
    return df


def validate_and_prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida que existan las columnas requeridas, verifica tipos compatibles
    y ordena las columnas según el esquema de entrenamiento.
    """
    missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Faltan columnas requeridas en el archivo de entrada: {missing_cols}. "
            f"Columnas esperadas: {REQUIRED_FEATURES}"
        )

    # Seleccionar y ordenar según el esquema del entrenamiento (ignora columnas extra)
    X = df[REQUIRED_FEATURES].copy()

    # Validar que las columnas numéricas contengan tipos numéricos compatibles
    for col in NUMERIC_FEATURES:
        non_null_series = X[col].dropna()
        if not non_null_series.empty:
            converted = pd.to_numeric(non_null_series, errors="coerce")
            if converted.isna().any():
                raise TypeError(
                    f"La columna numérica '{col}' contiene valores incompatibles no numéricos."
                )
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

    # Validar columnas categóricas
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].where(X[col].notna(), np.nan)
        non_null_cat = X[col].dropna()
        if not non_null_cat.empty:
            X.loc[non_null_cat.index, col] = non_null_cat.astype(str)

    return X


def inject_controlled_missing_values(
    df: pd.DataFrame, col: str = "temperature", ratio: float = 0.015, seed: int = 42
) -> pd.DataFrame:
    """
    Asegura que la columna indicada tenga el porcentaje requerido de valores nulos
    (1.5 % por defecto) de manera reproducible cuando el dataset original no los tiene.
    """
    df_out = df.copy()
    current_null_ratio = df_out[col].isna().mean()
    if current_null_ratio == 0.0 and len(df_out) > 0:
        rng = np.random.default_rng(seed)
        n_nulls = max(1, int(len(df_out) * ratio))
        null_positions = rng.choice(len(df_out), size=n_nulls, replace=False)
        df_out.iloc[null_positions, df_out.columns.get_loc(col)] = np.nan
    return df_out


def build_model_pipeline(
    n_estimators: int = 50, max_depth: int = None, seed: int = 42
) -> Pipeline:
    """Construye el Pipeline completo de preprocesamiento y RandomForestRegressor."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def build_baseline_model() -> DummyRegressor:
    """Construye el modelo base (DummyRegressor por la media)."""
    return DummyRegressor(strategy="mean")


def compute_regression_metrics(y_true, y_pred) -> dict:
    """Calcula RMSE, MAE y R2 para evaluación."""
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_git_commit_hash() -> str:
    """Obtiene el hash del commit actual de Git o 'unknown' si no está disponible."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
