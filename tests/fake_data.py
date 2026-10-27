"""Маленькие синтетические таблицы со схемой Home Credit. Нужны тестам, чтобы не грузить реальные CSV."""
import numpy as np
import pandas as pd

from src.config import DAYS_EMPLOYED_ANOMALY


def fake_application(n: int = 2000, seed: int = 0, with_target: bool = True, id_start: int = 100000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ext = rng.uniform(0, 1, size=(n, 3))
    ext[rng.uniform(size=(n, 3)) < 0.3] = np.nan
    days_employed = -rng.integers(100, 15000, n).astype(float)
    days_employed[rng.uniform(size=n) < 0.15] = DAYS_EMPLOYED_ANOMALY
    credit = rng.uniform(5e4, 2e6, n)
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(id_start, id_start + n),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        "CODE_GENDER": rng.choice(["M", "F", "XNA"], n, p=[0.45, 0.549, 0.001]),
        "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
        "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "XNA", "Self-employed", "Medicine", None], n),
        "DAYS_BIRTH": -rng.integers(21 * 365, 69 * 365, n),
        "DAYS_EMPLOYED": days_employed,
        "AMT_INCOME_TOTAL": rng.uniform(3e4, 5e5, n),
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": credit / rng.uniform(10, 60, n),
        "AMT_GOODS_PRICE": credit * rng.uniform(0.8, 1.0, n),
        "EXT_SOURCE_1": ext[:, 0],
        "EXT_SOURCE_2": ext[:, 1],
        "EXT_SOURCE_3": ext[:, 2],
        "OWN_CAR_AGE": np.where(rng.uniform(size=n) < 0.6, np.nan, rng.integers(0, 30, n)),
    })
    if with_target:
        filled = np.where(np.isnan(ext), 0.5, ext)       # пропуск = "средний" скор
        score = -0.5 - 4.0 * filled.mean(axis=1) + rng.normal(0, 0.5, n)
        df.insert(1, "TARGET", (rng.uniform(size=n) < 1 / (1 + np.exp(-score))).astype(int))
    return df


def fake_bureau(client_ids, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # у части клиентов нет ни одной записи в бюро, как в реальных данных
    with_history = rng.choice(client_ids, size=int(len(client_ids) * 0.85), replace=False)
    counts = rng.integers(1, 8, len(with_history))
    ids = np.repeat(with_history, counts)
    m = len(ids)
    credit_sum = rng.uniform(1e4, 1e6, m)
    credit_sum[rng.uniform(size=m) < 0.02] = 0.0  # нулевые суммы: проверка деления на 0
    return pd.DataFrame({
        "SK_ID_CURR": ids,
        "SK_ID_BUREAU": np.arange(5_000_000, 5_000_000 + m),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed", "Sold", "Bad debt"], m, p=[0.37, 0.62, 0.008, 0.002]),
        "DAYS_CREDIT": -rng.integers(1, 2900, m),
        "AMT_CREDIT_SUM": credit_sum,
        "AMT_CREDIT_SUM_DEBT": np.where(rng.uniform(size=m) < 0.15, np.nan, credit_sum * rng.uniform(0, 1, m)),
        "CREDIT_DAY_OVERDUE": np.where(rng.uniform(size=m) < 0.97, 0, rng.integers(1, 2000, m)),
    })


def write_fake_dataset(folder, n_train: int = 3000, n_test: int = 800) -> None:
    """Пишет фейковые application_train/test и bureau в папку (для смоук-теста train.py)."""
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    train = fake_application(n_train, seed=1)
    test = fake_application(n_test, seed=2, with_target=False, id_start=900000)
    train.to_csv(folder / "application_train.csv", index=False)
    test.to_csv(folder / "application_test.csv", index=False)
    all_ids = np.concatenate([train.SK_ID_CURR.values, test.SK_ID_CURR.values])
    fake_bureau(all_ids, seed=3).to_csv(folder / "bureau.csv", index=False)
