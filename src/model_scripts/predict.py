import logging
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any
from datetime import datetime
from src.model_scripts.load_model import load_model, load_model_from_mlflow
import yaml
import argparse
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# ЗАГРУЗКА КОНФИГУРАЦИИ
# ==========================================
def load_config(config_path: str = "config/data_config.yaml") -> Dict:
    """Загружает конфигурацию из YAML файла"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config

# ==========================================
# ПОДГОТОВКА ВХОДНЫХ ДАННЫХ
# ==========================================
def prepare_input_data(
    data: Union[pd.DataFrame, str, Dict, List],
    feature_cols: List[str] = None
    ) -> pd.DataFrame:
    """
    Подготавливает входные данные для предсказания 
    Args:
        data: входные данные (DataFrame, путь к файлу, словарь, список)
        feature_cols: список колонок для предсказания
    """
    config = load_config()
    
    if feature_cols is None:
        feature_cols = config['features']['feature_cols']
    
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
    
    # Оставляем только нужные колонки
    available_cols = [col for col in feature_cols if col in X.columns]
    X = X[available_cols]
    
    # Преобразуем категориальные колонки
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype('category')
    
    logger.info(f"Prepared {len(X)} rows for prediction")
    return X

# ==========================================
# ПРЕДСКАЗАНИЕ
# ==========================================
def predict(
    model: Any,
    data: Union[pd.DataFrame, str, Dict, List],
    threshold: float = 0.5
    ) -> pd.Series:
    """
    Делает предсказание на основе загруженной модели
    Args:
        model: загруженная модель
        data: входные данные
        return_proba: если True, возвращает вероятности, иначе бинарные классы
        threshold: порог для бинарной классификации
    """
    logger.info("Starting prediction...")
    
    # Подготовка данных
    X = prepare_input_data(data)
    
    # Предсказание
    proba = model.predict_proba(X)[:, 1]
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
    print(f"{result}")


if __name__ == "__main__":
    main()