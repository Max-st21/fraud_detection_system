from fastapi import FastAPI, HTTPException
import logging
from pathlib import Path
import pandas as pd
from src.model_scripts.load_model import load_model
from src.model_scripts.predict import predict
from src.predict_service.models import HealthResponse, PredictRequest
import hashlib
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API")

# Загрузка модели при старте
model_files = list(Path("models").glob("*.pkl"))
model = None
model_name = None

if not model_files:
    logger.error("No model files found in models/ directory")
else:
    model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Model file size: {model_path.stat().st_size} bytes")
    logger.info(f"Model file modified: {model_path.stat().st_mtime}")
    # Сохраняем хеш файла модели для сравнения
    with open(model_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    logger.info(f"Model file MD5: {file_hash}")
    
    model = load_model(model_path)
    model_name = model_path.name
    logger.info(f"Model loaded: {model_name}")

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")

@app.post("/predict")
async def predict_endpoint(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Преобразуем Pydantic модель в словарь
        data_dict = request.data.model_dump()
        
        # Преобразуем словарь в DataFrame
        input_df = pd.DataFrame([data_dict])
        
        # Вызываем существующую функцию predict, передавая DataFrame
        result = predict(model, input_df, request.threshold)
        
        # Извлекаем предсказание для первой (единственной) строки
        prediction = int(result['predictions'].iloc[0])
        probability = float(result['probabilities'].iloc[0])
        
        return {
            "status": "success",
            "prediction": prediction,
            "probability": probability
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))