import logging
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any
from datetime import datetime
from src.model_scripts.load_model import load_model, load_model_from_mlflow
import yaml
import argparse
import json
import lightgbm as lgb

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# ЗАГРУЗКА КОНФИГУРАЦИИ
# ==========================================
def load_config(config_path: str = "config/data_config.yaml") -> Dict:
    """Загружает конфигурацию из YAML файла"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using default config")
        # Возвращаем дефолтную конфигурацию
        return {
            'features': {
                'feature_cols': [
                    'transaction_amount', 'login_attempts', 'device_risk_score',
                    'transfer_frequency', 'anomaly_score', 'account_age_days',
                    'transaction_time_hour', 'failed_transactions_last_30d',
                    'avg_monthly_balance', 'daily_transaction_count', 'geo_distance_km',
                    'session_duration_minutes', 'transaction_velocity_score',
                    'payment_channel', 'authentication_type', 'card_present_flag',
                    'international_transaction_flag', 'suspicious_ip_flag'
                ]
            },
            'categorical_features': ['payment_channel', 'authentication_type']
        }

# ==========================================
# ПОДГОТОВКА ВХОДНЫХ ДАННЫХ
# ==========================================
def prepare_input_data(
    data: Union[pd.DataFrame, str, Dict, List],
    feature_cols: List[str] = None,
    categorical_features: List[str] = None
    ) -> pd.DataFrame:
    """
    Подготавливает входные данные для предсказания 
    Args:
        data: входные данные (DataFrame, путь к файлу, словарь, список)
        feature_cols: список колонок для предсказания
        categorical_features: список категориальных колонок
    """
    config = load_config()
    
    if feature_cols is None:
        feature_cols = config['features']['feature_cols']
    
    if categorical_features is None:
        categorical_features = config.get('categorical_features', [])
    
    # Преобразование различных форматов в DataFrame
    if isinstance(data, pd.DataFrame):
        X = data.copy()
    elif isinstance(data, str):
        # Путь к файлу
        if data.endswith('.csv'):
            X = pd.read_csv(data)
        elif data.endswith('.parquet'):
            X = pd.read_parquet(data)
        elif data.endswith('.json'):
            X = pd.read_json(data)
        else:
            raise ValueError(f"Unsupported file format: {data}")
        logger.info(f"Loaded data from {data}")
    elif isinstance(data, dict):
        X = pd.DataFrame([data])
    elif isinstance(data, list):
        X = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    # Проверяем наличие необходимых колонок
    missing_cols = set(feature_cols) - set(X.columns)
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        # Добавляем недостающие колонки с NaN
        for col in missing_cols:
            X[col] = np.nan
    
    # Оставляем только нужные колонки в правильном порядке
    X = X[feature_cols]
    
    # Преобразуем категориальные колонки
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].fillna('missing')
            X[col] = X[col].astype(str).astype('category')
            logger.debug(f"Converted {col} to categorical")
    
    # Числовые колонки: заполняем пропуски
    numeric_cols = [col for col in feature_cols if col not in categorical_features]
    for col in numeric_cols:
        if col in X.columns and X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
        if col in X.columns:
            X[col] = X[col].fillna(0)
    
    logger.info(f"Prepared {len(X)} rows for prediction")
    return X

# ==========================================
# ПРЕДСКАЗАНИЕ
# ==========================================
def predict(
    model: Any,
    data: Union[pd.DataFrame, str, Dict, List],
    threshold: float = 0.5
    ) -> Dict:
    """
    Делает предсказание на основе загруженной модели
    Args:
        model: загруженная модель
        data: входные данные
        threshold: порог для бинарной классификации
    """
    logger.info("Starting prediction...")
    
    # Подготовка данных
    X = prepare_input_data(data)
    
    if len(X) == 0:
        raise ValueError("No data to predict")
    
    # Предсказание в зависимости от типа модели
    if isinstance(model, lgb.Booster):
        logger.info("Using LightGBM Booster for prediction")
        proba = model.predict(X)
        # Если proba 2D, берем вероятность второго класса
        if len(proba.shape) > 1:
            proba = proba[:, 1]
    elif hasattr(model, 'predict_proba'):
        logger.info("Using predict_proba method")
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'predict'):
        logger.info("Using predict method")
        proba = model.predict(X)
        if hasattr(proba, 'shape') and len(proba.shape) > 1:
            proba = proba[:, 1] if proba.shape[1] > 1 else proba.flatten()
    else:
        raise ValueError("Model does not have predict_proba or predict method")
    
    predictions = (proba >= threshold).astype(int)
    
    # Формирование отчета
    result = {
        'predictions': pd.Series(predictions, name='prediction'),
        'probabilities': pd.Series(proba, name='probability'),
        'statistics': {
            'total_rows': len(X),
            'fraud_predictions': int(predictions.sum()),
            'fraud_rate': float(predictions.mean()),
            'threshold_used': threshold,
            'prediction_time': datetime.now().isoformat()
        }
    }
    
    logger.info(f"Statistics: {result['statistics']}")
    return result

# ==========================================
# ОСНОВНАЯ ФУНКЦИЯ (ДЛЯ КОМАНДНОЙ СТРОКИ)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--input', type=str, required=True, help='Input file path or JSON string')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output file path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--model-path', type=str, default=None, help='Path to model .pkl file')
    parser.add_argument('--use-mlflow', action='store_true', help='Load model from MLflow')
    parser.add_argument('--mlflow-stage', type=str, default='Production', help='MLflow model stage')
    
    args = parser.parse_args()
    
    # Загрузка модели
    if args.use_mlflow:
        model = load_model_from_mlflow(stage=args.mlflow_stage)
    else:
        model = load_model(args.model_path)
    
    # Загрузка входных данных
    if args.input.endswith(('.csv', '.parquet', '.json')):
        X = prepare_input_data(args.input)
    else:
        # Пробуем парсить как JSON
        try:
            data = json.loads(args.input)
            X = prepare_input_data(data)
        except:
            raise ValueError("Input must be a file path or valid JSON")
    
    # Предсказание
    result = predict(model, X, threshold=args.threshold)
    
    # Сохранение результатов
    output_df = pd.DataFrame({
        'prediction': result['predictions'],
        'probability': result['probabilities']
    })
    
    # Добавляем исходные данные, если они есть в X
    for col in X.columns:
        output_df[col] = X[col].values
    
    output_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")
    
    # Вывод статистики
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(json.dumps(result['statistics'], indent=2))


if __name__ == "__main__":
    main()