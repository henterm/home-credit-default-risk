"""Загрузка таблиц Home Credit."""
import numpy as np
import pandas as pd

from src.config import DATA_DIR, SEED, TARGET


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 и int64 -> int32: памяти нужно примерно вдвое меньше."""
    df = df.copy()
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include="int64").columns:
        if df[col].between(np.iinfo(np.int32).min, np.iinfo(np.int32).max).all():
            df[col] = df[col].astype(np.int32)
    return df


def read_table(name: str, nrows: int | None = None) -> pd.DataFrame:
    """Читает data/raw/<name>.csv, например read_table("bureau")."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла {path}. Переложите CSV из Kaggle в папку {DATA_DIR}."
        )
    return downcast(pd.read_csv(path, nrows=nrows))


def stratified_sample(df: pd.DataFrame, frac: float, seed: int = SEED) -> pd.DataFrame:
    """Подвыборка с той же долей дефолтов, что и в полном train. Для быстрых прогонов."""
    parts = [g.sample(frac=frac, random_state=seed) for _, g in df.groupby(TARGET)]
    return pd.concat(parts).sort_index()


def load_application(sample_frac: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (train, test) из application_train.csv и application_test.csv."""
    train = read_table("application_train")
    test = read_table("application_test")
    if sample_frac:
        train = stratified_sample(train, sample_frac)
    return train, test
