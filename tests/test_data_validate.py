import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Загрузка файлов с данными
def load_all_processed_files(data_dir: str = "data/processed") -> pd.DataFrame:
    """Загружает все Parquet файлы из каталога processed"""
    data_path = Path(data_dir)
    
    # Ищем все папки с маской run_id=*
    run_folders = list(data_path.glob("run_id=*"))
    
    if not run_folders:
        raise FileNotFoundError(f"No run folders found in {data_dir}")
    
    all_dfs = []
    for folder in sorted(run_folders):
        parquet_file = folder / "data.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            df['source_file'] = folder.name  # сохраняем run_id
            all_dfs.append(df)
            print(f"Loaded: {folder.name}/data.parquet ({len(df)} rows)")
        else:
            print(f"Warning: No data.parquet found in {folder}")
    
    if not all_dfs:
        raise ValueError("No data files loaded")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df



# Фикстура с данными
@pytest.fixture(scope="module")
def df():
    return load_all_processed_files()


# ==========================================
# ТЕСТЫ КАЧЕСТВА ДАННЫХ
# ==========================================

def test_no_duplicates(df):
    """Проверка: нет дубликатов строк"""
    assert df.duplicated().sum() == 0, f"Found {df.duplicated().sum()} duplicate rows"


def test_fraud_flag_binary(df):
    """Проверка: fraud_flag только 0 или 1"""
    assert df['fraud_flag'].isin([0, 1]).all(), "fraud_flag contains values other than 0/1"


def test_fraud_flag_balance(df):
    """Проверка: есть мошеннические транзакции"""
    fraud_count = df['fraud_flag'].sum()
    assert fraud_count > 0, "No fraud transactions in dataset"


def test_transaction_amount_positive(df):
    """Проверка: сумма транзакции положительная"""
    assert (df['transaction_amount'] > 0).all(), "Transaction amount contains zero or negative values"


def test_device_risk_score_range(df):
    """Проверка: device_risk_score в диапазоне 0-100"""
    assert df['device_risk_score'].between(0, 100).all(), "device_risk_score out of range [0,100]"


def test_login_attempts_non_negative(df):
    """Проверка: login_attempts не отрицательный"""
    assert (df['login_attempts'] >= 0).all(), "login_attempts contains negative values"


def test_transaction_time_hour_valid(df):
    """Проверка: час транзакции 0-23"""
    assert df['transaction_time_hour'].between(0, 23).all(), "transaction_time_hour out of range [0,23]"


def test_categorical_columns_known_values(df):
    """Проверка: категориальные колонки имеют допустимые значения"""   
    # Допустимые значения для payment_channel
    valid_channels = ['Web Banking', 'Mobile App', 'ATM', 'POS Terminal']
    assert df['payment_channel'].isin(valid_channels).all(), \
        f"Invalid payment_channel values: {df['payment_channel'].unique()}"
    
    # Допустимые значения для authentication_type
    valid_auth = ['OTP', 'Password Only', 'Two-Factor Authentication', 'Biometric']
    assert df['authentication_type'].isin(valid_auth).all(), \
        f"Invalid authentication_type values: {df['authentication_type'].unique()}"


def test_log_transformed_columns(df):
    """Проверка: log-трансформированные колонки созданы корректно"""
    assert 'transaction_amount_log' in df.columns, "Missing transaction_amount_log"
    assert 'geo_distance_km_log' in df.columns, "Missing geo_distance_km_log"
    assert (df['transaction_amount_log'] >= 0).all(), "transaction_amount_log has negative values"


def test_no_infinite_values(df):
    """Проверка: нет бесконечных значений"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert np.isfinite(df[col]).all(), f"Non-finite values found in {col}"