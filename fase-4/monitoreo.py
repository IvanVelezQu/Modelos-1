import argparse
import csv
import os
import sys


def analyze_log(log_path: str) -> dict:
    """Calcula el total de solicitudes y el porcentaje de respuestas fallidas (4xx/5xx)."""
    if not os.path.exists(log_path):
        return {"total": 0, "fallidas": 0, "porcentaje_fallos": None}

    total = 0
    fallidas = 0

    with open(log_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            estado_raw = (row.get("estado_http") or "").strip()
            if not estado_raw:
                continue
            total += 1
            estado = int(estado_raw)
            if 400 <= estado <= 599:
                fallidas += 1

    if total == 0:
        return {"total": 0, "fallidas": 0, "porcentaje_fallos": None}

    porcentaje = (fallidas / total) * 100.0
    return {"total": total, "fallidas": fallidas, "porcentaje_fallos": porcentaje}


def main():
    parser = argparse.ArgumentParser(
        description="Calcula métricas de operación a partir del registro de solicitudes de /predict."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/predicciones_log.csv",
        help="Ruta al archivo CSV de registro (por defecto: logs/predicciones_log.csv).",
    )
    args = parser.parse_args()

    res = analyze_log(args.log_file)
    print("=== MÉTRICAS DE OPERACIÓN DE LA API ===")
    print(f"Archivo analizado                : {args.log_file}")
    print(f"Total de solicitudes             : {res['total']}")
    if res["total"] == 0:
        print("Porcentaje de solicitudes fallidas: sin solicitudes")
    else:
        print(f"Solicitudes fallidas (4xx/5xx)   : {res['fallidas']}")
        print(f"Porcentaje de solicitudes fallidas: {res['porcentaje_fallos']:.1f}%")


if __name__ == "__main__":
    main()
