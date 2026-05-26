from fastapi import FastAPI, HTTPException
import logging
from pathlib import Path
from src.model_scripts.load_model import load_model, load_model_from_mlflow
from src.model_scripts.predict import predict
from src.predict_service.models import HealthResponse, PredictRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API")

model_files = list(Path("/models").glob("*.pkl"))
model_path = max(model_files, key=lambda p: p.stat().st_mtime)
# Загрузка модели
model = load_model(model_path)
model_name = model_path.name


@app.get("/health", response_model=HealthResponse)
async def health():
    """Проверка работоспособности сервиса"""
    return HealthResponse(
        status="ok",
    )


@app.post(f"/predict/{model_name}")
async def predict(request: PredictRequest):
    """
    Предсказание по одной транзакции    
    Пример запроса:
    {
        "data": {
            "transaction_amount": 15000,
            "login_attempts": 3,
            "device_risk_score": 75.5,
            ...
        },
        "model_name": "FraudDetectionModel",
        "model_stage": "Production",
        "threshold": 0.5
    }
    """
    try:
        # Предсказание
        result = predict(model, request.data, request.threshold)
        return {
            "status": "success",
            **result
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))