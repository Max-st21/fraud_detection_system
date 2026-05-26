import pickle
import logging
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score,
    classification_report,
    roc_curve,
    confusion_matrix
)
import mlflow
import mlflow.lightgbm
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

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
# ЗАГРУЗКА ДАННЫХ
# ==========================================
def load_all_processed_data(data_dir: str) -> pd.DataFrame:
    """
    Загружает все Parquet файлы из всех run_id папок
    Структура: data_dir/run_id=*/data.parquet
    """
    data_dir = Path(data_dir)
    all_dfs = []
    
    # Ищем все папки с маской run_id=*
    run_folders = list(data_dir.glob("run_id=*"))
    
    if not run_folders:
        raise ValueError(f"No run folders found in {data_dir}")
    
    logger.info(f"Found {len(run_folders)} run folders")
    
    for folder in sorted(run_folders):
        parquet_file = folder / "data.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            all_dfs.append(df)
            logger.info(f"  Loaded {len(df)} rows from {parquet_file}")
        else:
            logger.warning(f"No data.parquet found in {folder}")
    
    if not all_dfs:
        raise ValueError("No data files loaded")
    
    result_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total loaded: {len(result_df)} rows, {len(result_df.columns)} columns")
    return result_df

# ==========================================
# ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ
# ==========================================
def prepare_data(
    df: pd.DataFrame, 
    target_col: str, 
    feature_cols: list,
    exclude_cols: list = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Подготавливает данные для обучения
    """
    if exclude_cols is None:
        exclude_cols = []
    if feature_cols is None:
        feature_cols = config['features']['feature_cols']
    if target_col is None:
        target_col = config['data']['target_col']
    # Исключаем колонки
    exclude = exclude_cols + [target_col]
    available_features = [col for col in feature_cols if (col in df.columns) and (col not in exclude)]
    
    X = df[available_features].copy()
    y = df[target_col].copy()

    # Преобразуем категориальные признаки
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category').cat.codes

    # Проверяем наличие необходимых колонок
    missing_cols = set(feature_cols) - set(X.columns)
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        # Добавляем недостающие колонки с NaN
        for col in missing_cols:
            X[col] = np.nan

    logger.info(f"Prepared data: {X.shape[1]} features, {len(y)} samples")  
    return X, y

# ==========================================
# ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ С OPTUNA
# ==========================================
def objective(trial, X, y, n_folds=5):
    """Целевая функция для Optuna с кросс-валидацией"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 5),
        'is_unbalance': True
    }
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        auc_scores.append(roc_auc_score(y_val_fold, y_pred_proba))
    
    mean_auc = np.mean(auc_scores)
    return mean_auc

def optimize_hyperparameters(
    X, y,
    n_trials: int = 50, 
    model_name: str = "fraud_detection"
    ) -> Dict:
    """Оптимизация гиперпараметров с Optuna"""
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    project_root = Path(__file__).parent.parent
    db_path = project_root / "db" / "optuna.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction='maximize',
        study_name=model_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,  # Используем результаты предыдущих запусков
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best AUC: {study.best_value:.4f}")
    
    return study.best_trial.params

# ==========================================
# ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ
# ==========================================
def train_final_model(
    X_train, y_train, 
    X_test, y_test,
    params: Dict,
    ) -> lgb.LGBMClassifier:
    """Обучает финальную модель на всех train данных"""
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    return model

# ==========================================
# ВЫЧИСЛЕНИЕ МЕТРИК
# ==========================================
def calculate_metrics(y_true, y_pred, y_pred_proba) -> Dict:
    """Вычисляет все метрики"""
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'f1_score': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_fraud': precision_score(y_true, y_pred),
        'recall_fraud': recall_score(y_true, y_pred),
        'precision_normal': precision_score(y_true, y_pred, pos_label=0),
        'recall_normal': recall_score(y_true, y_pred, pos_label=0),
    }
    return metrics

# ==========================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ==========================================
def plot_roc_curve(y_true, y_pred_proba, save_path: Path = None):
    """Строит ROC-кривую и сохраняет в файл"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    return fpr, tpr


def plot_confusion_matrix(y_true, y_pred, save_path: Path = None):
    """Строит матрицу ошибок (confusion matrix) и сохраняет в файл"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Получаем значения TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (0)', 'Fraud (1)'],
                yticklabels=['Normal (0)', 'Fraud (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix\n\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


# ==========================================
# ОСНОВНАЯ ФУНКЦИЯ
# ==========================================
def main():
    # Загрузка конфигурации
    config = load_config()
    
    # Настройка MLflow. Запуск через ui: mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
    MLFLOW_DIR = Path(__file__).parent.parent / "mlruns"
    MLFLOW_DIR.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Загрузка данных
    logger.info("Loading processed data...")
    df = load_all_processed_data(config['data']['processed_dir'])
    

    # Подготовка данных
    X, y = prepare_data(
        df, 
        config['data']['target_col'], 
        config['features']['feature_cols']
    )
    
    # Разделение на train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Дополнительное разделение train на train/val для оптимизации
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    logger.info(f"Train size: {len(X_train_opt)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Логирование в MLflow
    with mlflow.start_run(run_name=config['mlflow']['run_name']) as run:
        # Оптимизация гиперпараметров
        best_params = optimize_hyperparameters(
            X_train_opt, y_train_opt,
            n_trials=config['training']['optuna_trials']
        )
        
        # Добавляем фиксированные параметры
        best_params.update({
            'verbosity': -1,
            'random_state': 42,
            'is_unbalance': True
        })
        
        # Обучение финальной модели
        logger.info("Training final model...")
        final_model = train_final_model(
            X_train, y_train, X_test, y_test,
            best_params
        )
        
        # Предсказания
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Метрики
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        # Создаём временную папку для графиков
        plots_dir = Path(__file__).parent.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Строим и сохраняем ROC-кривую
        roc_path = plots_dir / f"roc_curve_{run.info.run_id}.png"
        plot_roc_curve(y_test, y_pred_proba, save_path=roc_path)
        
        # Строим и сохраняем матрицу ошибок
        cm_path = plots_dir / f"confusion_matrix_{run.info.run_id}.png"
        cm_values = plot_confusion_matrix(y_test, y_pred, save_path=cm_path)

        # Логируем параметры
        mlflow.log_params(best_params)
        
        # Логируем метрики
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Логируем classification report
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'])
        mlflow.log_text(report, "classification_report.txt")

        # Логируем графики как артефакты
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(cm_path))
        
        # Сохраняем модель в MLflow
        mlflow.lightgbm.log_model(final_model, "model")
        
        # Регистрируем модель в Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, config['mlflow']['model_name'])
        
        logger.info(f"Model logged to MLflow with run_id: {run.info.run_id}")
        
        # Сохранение модели в .pkl
        model_path = Path(config['models']['local_path']) / f"{config['models']['model_filename']}_{run.info.run_id}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
        
        logger.info(f"Model saved to {model_path}")
    
    # Вывод результатов
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nPer-class metrics:")
    print(f"  Fraud     - Precision: {metrics['precision_fraud']:.4f}, Recall: {metrics['recall_fraud']:.4f}")
    print(f"  Normal    - Precision: {metrics['precision_normal']:.4f}, Recall: {metrics['recall_normal']:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

if __name__ == "__main__":
    main()