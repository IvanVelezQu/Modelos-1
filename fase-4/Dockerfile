FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models \
    MODEL_VERSION=v2 \
    TRAIN_DATA_PATH=/app/data/rideshare_kaggle.csv \
    PREDICTIONS_LOG_PATH=/app/logs/predicciones_log.csv

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ejemplo_prediccion.csv ./data/ejemplo_prediccion.csv
COPY train.py predict.py api.py cliente.py monitoreo.py ./

RUN mkdir -p /app/models /app/logs

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
