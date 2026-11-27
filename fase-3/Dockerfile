FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models \
    MODEL_VERSION=v1 \
    TRAIN_DATA_PATH=/app/data/rideshare_kaggle.csv

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ejemplo_prediccion.csv ./data/ejemplo_prediccion.csv
COPY train.py predict.py api.py cliente.py ./

RUN mkdir -p /app/models

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
