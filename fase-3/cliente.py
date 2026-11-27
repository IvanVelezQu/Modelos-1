import argparse
import json
import requests


def main():
    parser = argparse.ArgumentParser(
        description="Cliente de prueba para consultar la API REST de tarifas Uber/Lyft."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="URL base de la API (por defecto: http://localhost:8000)",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    # 1. Consultar GET /health
    resp_health = requests.get(f"{base_url}/health", timeout=10)
    print(f"GET /health ({resp_health.status_code}):", resp_health.json())

    # 2. Enviar una solicitud de predicción con datos reales del proyecto
    payload = {
        "distance": 2.5,
        "surge_multiplier": 1.0,
        "temperature": 42.0,
        "cab_type": "Uber",
        "name": "UberX",
    }
    print("\nEnviando solicitud a POST /predict:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    resp_pred = requests.post(f"{base_url}/predict", json=payload, timeout=15)
    print(f"\nRespuesta POST /predict ({resp_pred.status_code}):")
    print(json.dumps(resp_pred.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
