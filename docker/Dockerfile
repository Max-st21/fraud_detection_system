FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY dependencies/service_requirements.txt .
RUN pip install --no-cache-dir -r service_requirements.txt

# Копирование исходного кода
COPY src/predict_service/ ./src/predict_service/
COPY src/model_scripts/load_model.py src/model_scripts/predict.py ./src/model_scripts/
COPY config/ ./config/
COPY models/ ./models/
COPY tests/ ./tests/

# Переменные окружения
ENV PYTHONPATH=/app

# Открываем порт
EXPOSE 8000

# Команда для запуска
CMD ["uvicorn", "src.predict_service.main:app", "--host", "0.0.0.0", "--port", "8000"]