import numpy as np

from src.config import DAYS_EMPLOYED_ANOMALY
from src.preprocessing import build_application_features, categorical_columns, to_lgbm_categories
from tests.fake_data import fake_application


def _build():
    train = fake_application(1000, seed=1)
    test = fake_application(300, seed=2, with_target=False, id_start=900000)
    return build_application_features(train, test)


def test_anomaly_flag_and_nan():
    train, test = _build()
    for df in (train, test):
        assert (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).sum() == 0
        assert df["DAYS_EMPLOYED_ANOM"].sum() > 0
        assert df.loc[df["DAYS_EMPLOYED_ANOM"] == 1, "DAYS_EMPLOYED"].isna().all()


def test_train_and_test_on_same_scale():
    # Тот самый баг из первого ноутбука: возраст в годах в train и в днях в test.
    train, test = _build()
    for df in (train, test):
        assert df["AGE_YEARS"].between(18, 75).all()
        assert (df["EMPLOYED_TO_AGE"].dropna() >= 0).all()


def test_no_infinities():
    train, test = _build()
    for df in (train, test):
        num = df.select_dtypes(include="number")
        assert not np.isinf(num.to_numpy(dtype=float)).any()


def test_same_columns_except_target():
    train, test = _build()
    assert set(train.columns) - set(test.columns) == {"TARGET"}


def test_lgbm_categories_shared():
    train, test = _build()
    X, X_test = train.drop(columns=["TARGET"]), test
    cats = categorical_columns(X)
    X, X_test = to_lgbm_categories(X, X_test[X.columns], cats)
    for col in cats:
        assert list(X[col].cat.categories) == list(X_test[col].cat.categories)
