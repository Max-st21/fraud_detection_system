import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import boto3
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw_data(path: str = "data/raw/banking_transactions.csv") -> pd.DataFrame:
    """
    Загружает CSV файл и возвращает DataFrame.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")
    
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw data shape: {df.shape}")
    return df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая очистка данных.
    """
    # Проверка дубликатов
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    logger.info(f"Dropped {initial_rows - df.shape[0]} duplicates.")

    # Преобразование целевой переменной
    if df['fraud_flag'].dtype == 'object':
        df['fraud_flag'] = df['fraud_flag'].str.upper().map({'TRUE': 1, 'FALSE': 0})
    else:
        # Если уже число/булево
        df['fraud_flag'] = df['fraud_flag'].astype(int)
    
    # Проверка и заполнение NaN
    df['fraud_flag'] = df['fraud_flag'].fillna(0).astype(int)
    
    # Заполнение пропусков
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # Используем np.log1p для положительных асимметричных признаков
    df['transaction_amount_log'] = np.log1p(df['transaction_amount'])
    df['geo_distance_km_log'] = np.log1p(df['geo_distance_km'])

    # Задаём индекс
    df.set_index('transaction_id', inplace=True, verify_integrity=True)

    return df

def save_processed_data(df: pd.DataFrame, output_path: str = None, base_name: str = "run_id"):
    """
    Сохраняет DataFrame в Parquet для эффективности.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        # Создаем папку data/processed/run_id={timestamp}/
        output_path = f"data/processed/{base_name}={timestamp}/data.parquet"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Сохраняем файл
    df.to_parquet(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")

def main():
    # Загрузка данных
    df = load_raw_data('data/raw/banking_transactions.csv')

    # Трансформация
    df = transform_data(df)

    # Сохранение
    save_processed_data(df, base_name='run_id')
    print('ETL pipeline completed successfully')

    # Загружаем в S3 (постоянное хранение)
    if os.getenv('AWS_ACCESS_KEY_ID'):
        s3 = boto3.client('s3')
        bucket = 'your-bucket-name'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for file in Path('data/processed').glob('*.parquet'):
            s3_key = f'processed-data/run_id={timestamp}/{file.name}'
            s3.upload_file(str(file), bucket, s3_key)
            print(f'Uploaded to s3://{bucket}/{s3_key}')
    else:
        print('Skipping S3 upload (no AWS credentials)')

    
if __name__ == "__main__":
    main()