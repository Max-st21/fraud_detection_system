from pydantic import BaseModel, Field
from typing import Optional, List

# pydantic models
class TransactionData(BaseModel):
    """Модель данных транзакции"""
    transaction_amount: Optional[float] = None
    login_attempts: Optional[int] = None
    device_risk_score: Optional[float] = None
    transfer_frequency: Optional[int] = None
    anomaly_score: Optional[float] = None
    account_age_days: Optional[int] = None
    transaction_time_hour: Optional[int] = None
    failed_transactions_last_30d: Optional[int] = None
    avg_monthly_balance: Optional[float] = None
    daily_transaction_count: Optional[int] = None
    geo_distance_km: Optional[float] = None
    session_duration_minutes: Optional[int] = None
    transaction_velocity_score: Optional[float] = None
    payment_channel: Optional[str] = None
    authentication_type: Optional[str] = None
    card_present_flag: Optional[int] = None
    international_transaction_flag: Optional[int] = None
    suspicious_ip_flag: Optional[int] = None


class PredictRequest(BaseModel):
    data: TransactionData
    threshold: Optional[float] = 0.5


class StatisticsResponse(BaseModel):
    """Статистика предсказаний"""
    total_rows: int = Field(..., description="Общее количество строк")
    fraud_predictions: int = Field(..., description="Количество предсказанных фродов")
    fraud_rate: float = Field(..., description="Доля фрода", ge=0, le=1)
    threshold_used: float = Field(..., description="Использованный порог", ge=0, le=1)
    prediction_time: str = Field(..., description="Время предсказания")


class PredictResponse(BaseModel):
    """Ответ предсказания (все поля опциональны)"""
    predictions: List[int] = Field(None, description="Список предсказанных классов")
    probabilities: List[float] = Field(None, description="Список вероятностей фрода")
    statistics: Optional[StatisticsResponse] = Field(None, description="Статистика")


class HealthResponse(BaseModel):
    """Модель ответа health check"""
    status: str