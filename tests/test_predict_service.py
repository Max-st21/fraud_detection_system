# tests/test_api.py
from fastapi.testclient import TestClient
import numpy as np
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent))

from src.predict_service.main import app

client = TestClient(app)


def test_health_endpoint():
    """Проверка эндпоинта /health"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_exists():
    """Проверка, что эндпоинт /predict существует"""
    # Получаем имя модели из пути (имитируем)
    response = client.post("/predict/test_model.pkl", json={
        "data": {
            "transaction_amount": 15000,
            "login_attempts": 3,
            "device_risk_score": 75.5,
            "transfer_frequency": 10,
            "anomaly_score": 0.5,
            "account_age_days": 365,
            "transaction_time_hour": 14,
            "failed_transactions_last_30d": 2,
            "avg_monthly_balance": 50000,
            "daily_transaction_count": 5,
            "geo_distance_km": 100,
            "session_duration_minutes": 30,
            "transaction_velocity_score": 50,
            "payment_channel": "Web Banking",
            "authentication_type": "OTP",
            "card_present_flag": 1,
            "international_transaction_flag": 0,
            "suspicious_ip_flag": 0
        },
        "threshold": 0.5
    })
    
    # Даже если модели нет, эндпоинт должен возвращать 404 или 500, но не 405
    assert response.status_code in [200, 404, 500]


def test_predict_with_mock_model(monkeypatch):
    """Тест с мок-моделью (без реальной загрузки)"""
    # Мокаем модель
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])
    
    # Мокаем load_model
    def mock_load_model(*args, **kwargs):
        return MockModel()
    
    monkeypatch.setattr("src.predict_service.main.load_model", mock_load_model)
    
    # Создаём новый клиент с моками
    from src.predict_service.main import app as mocked_app
    test_client = TestClient(mocked_app)
    
    response = test_client.post("/predict/test_model.pkl", json={
        "data": {
            "transaction_amount": 1000,
            "login_attempts": 1,
            "device_risk_score": 50,
            "transfer_frequency": 5,
            "anomaly_score": 0.3,
            "account_age_days": 100,
            "transaction_time_hour": 12,
            "failed_transactions_last_30d": 0,
            "avg_monthly_balance": 10000,
            "daily_transaction_count": 1,
            "geo_distance_km": 10,
            "session_duration_minutes": 10,
            "transaction_velocity_score": 20,
            "payment_channel": "Mobile App",
            "authentication_type": "Biometric",
            "card_present_flag": 1,
            "international_transaction_flag": 0,
            "suspicious_ip_flag": 0
        },
        "threshold": 0.5
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)


def test_predict_invalid_data():
    """Тест с некорректными данными"""
    response = client.post("/predict/model.pkl", json={
        "data": {}  # Пустые данные
    })
    # Должна быть ошибка (500 или 422)
    assert response.status_code in [422, 500]