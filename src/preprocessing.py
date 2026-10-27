"""Очистка и признаки по application_train / application_test.

Правило: каждая функция применяется к train и test одинаково.
Здесь нет ничего, что "учится" на данных (медиан, кодировок, TARGET), поэтому
утечки нет. Всё, что учится, живёт внутри sklearn Pipeline в src/models.py.
"""
import numpy as np
import pandas as pd

from src.config import DAYS_EMPLOYED_ANOMALY

EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Флаг аномалии сохраняем: у "неработающих" дефолтов меньше (5.4% против 8.7%),
    # и это полезный сигнал. Само значение заменяем на пропуск
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype("int8")
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != DAYS_EMPLOYED_ANOMALY)
    # 4 строки с CODE_GENDER = "XNA": это пропуск, а не отдельная категория.
    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].mask(df["CODE_GENDER"] == "XNA")
    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]      # кредит / годовой доход
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]    # платёжная нагрузка
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]                   # ~ 1 / срок кредита
    df["EMPLOYED_TO_AGE"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]             # доля жизни в работе
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]        # цена товара / сумма кредита

    # Внешние скоринги: самые сильные признаки. Среднее по доступным из трёх.
    df["EXT_SOURCE_MEAN"] = df[EXT_SOURCES].mean(axis=1)
    df["EXT_SOURCE_NAN_COUNT"] = df[EXT_SOURCES].isna().sum(axis=1).astype("int8")

    # Деление на 0 даёт inf, модели он не нужен. Превращаем в пропуск.
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def build_application_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = add_domain_features(clean_application(train))
    test = add_domain_features(clean_application(test))
    return train, test


def to_lgbm_categories(X: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str]):
    X, X_test = X.copy(), X_test.copy()
    for col in cat_cols:
        cats = pd.Index(X[col].dropna().unique()).sort_values()
        X[col] = pd.Categorical(X[col], categories=cats)
        X_test[col] = pd.Categorical(X_test[col], categories=cats)
    return X, X_test
