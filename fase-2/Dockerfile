FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ejemplo_prediccion.csv ./data/ejemplo_prediccion.csv
COPY train.py predict.py ./

RUN mkdir -p /app/models /app/outputs

CMD ["python", "train.py", "--help"]
