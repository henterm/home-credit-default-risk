"""Общие кирпичики для агрегатов «многие строки -> одна строка на клиента».

Используются в bureau.py, previous.py, чтобы все три модуля
давали признаки в одном формате: <PREFIX><КОЛОНКА>_<СТАТИСТИКА>.
"""
import numpy as np
import pandas as pd


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Деление, где x/0 и 0/0 дают NaN, а не inf. Для деревьев NaN = «не определено»."""
    return num / den.where(den != 0)


def present(aggs: dict, df: pd.DataFrame) -> dict:
    return {col: stats for col, stats in aggs.items() if col in df.columns}


def aggregate(df: pd.DataFrame, key: str, aggs: dict, prefix: str) -> pd.DataFrame:
    """groupby(key).agg(aggs) и склейка MultiIndex ("DAYS_CREDIT", "mean") -> PREFIX + "DAYS_CREDIT_MEAN".

    "sum" считается с min_count=1: у pandas сумма одних пропусков = 0, и клиент, у которого
    поле ни разу не заполнено, выглядел бы как «сумма ровно 0». Это ложный сигнал, нужен NaN.
    """
    aggs = present(aggs, df)
    grouped = df.groupby(key)
    if not aggs:
        return pd.DataFrame(index=pd.Index(sorted(df[key].unique()), name=key))
    parts = []
    other = {c: [s for s in stats if s != "sum"] for c, stats in aggs.items()}
    other = {c: s for c, s in other.items() if s}
    if other:
        o = grouped.agg(other)
        o.columns = [f"{prefix}{col}_{stat.upper()}" for col, stat in o.columns]
        parts.append(o)
    sum_cols = [c for c, stats in aggs.items() if "sum" in stats]
    if sum_cols:
        s = grouped[sum_cols].sum(min_count=1)
        s.columns = [f"{prefix}{c}_SUM" for c in sum_cols]
        parts.append(s)
    out = pd.concat(parts, axis=1)
    order = [f"{prefix}{c}_{s.upper()}" for c, stats in aggs.items() for s in stats]
    return out[order]


def add_flags(df: pd.DataFrame, column: str, values: dict) -> list[str]:
    """Создаёт бинарные флаги column == value. values: {"ИМЯ_ФЛАГА": "значение в данных"}.

    Возвращает имена созданных колонок. NaN в исходной колонке даёт флаг 0:
    «статус неизвестен» — это не «Active».
    """
    if column not in df.columns:
        return []
    names = []
    for name, value in values.items():
        df[name] = (df[column] == value).astype("float32")
        names.append(name)
    return names


def finalize(out: pd.DataFrame, key: str) -> pd.DataFrame:
    """inf -> NaN, float64 -> float32, ключ из индекса в колонку.
    """
    out = out.replace([np.inf, -np.inf], np.nan)
    float_cols = out.select_dtypes(include="float64").columns
    out[float_cols] = out[float_cols].astype("float32")
    int_cols = out.select_dtypes(include="int64").columns
    out[int_cols] = out[int_cols].astype("int32")
    out.index.name = key
    return out.reset_index()
