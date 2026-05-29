import requests
import os

# Базовый URL сервиса (можно переопределить через переменную окружения)
BASE_URL = os.getenv("API_URL", "http://localhost:8000")

def test_health_endpoint():
    """Проверка эндпоинта /health"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint_exists():
    """Проверка, что эндпоинт /predict существует"""
    response = requests.post(f"{BASE_URL}/predict", json={
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
    # Эндпоинт существует, ответ 200 (модель есть) или 503 (модель не загружена)
    assert response.status_code in [200, 503]

def test_predict_with_valid_data():
    """Тест с корректными данными (реальная модель в контейнере)"""
    response = requests.post(f"{BASE_URL}/predict", json={
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

def test_predict_missing_data():
    """Тест с отсутствующим полем data"""
    response = requests.post(f"{BASE_URL}/predict", json={
        "threshold": 0.5
    })
    assert response.status_code == 422

def test_predict_invalid_threshold():
    """Тест с некорректным порогом"""
    response = requests.post(f"{BASE_URL}/predict", json={
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
        "threshold": 1.5  # Некорректный порог (>1)
    })
    assert response.status_code == 422