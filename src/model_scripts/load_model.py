import pickle
from pathlib import Path
from typing import Any, Dict
import logging
import yaml
import mlflow
import mlflow.lightgbm

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
# ЗАГРУЗКА МОДЕЛИ
# ==========================================
def load_model(model_path: str = None) -> Any:
    """
    Загружает модель из .pkl файла
    Args:
        model_path: путь к .pkl файлу. Если None, берется из конфига
    """
    if model_path is None:
        config = load_config()
        model_path = Path(config['models']['local_path']) / config['models']['model_filename']
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {model_path}")
    return model

# ==========================================
# ЗАГРУЗКА МОДЕЛИ ИЗ MLFLOW
# ==========================================
def load_model_from_mlflow(model_name: str = None, stage: str = "Production") -> Any:
    """
    Загружает модель из MLflow Model Registry
    Args:
        model_name: имя модели в MLflow
        stage: стадия модели (Production, Staging, Archived)
    """
    config = load_config()
    
    if model_name is None:
        model_name = config['mlflow']['model_name']
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.lightgbm.load_model(model_uri)
    
    logger.info(f"Model loaded from MLflow: {model_uri}")
    return model
