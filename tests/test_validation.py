import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.models import make_logreg
from src.preprocessing import build_application_features, categorical_columns
from src.validation import bootstrap_auc_ci, classification_metrics, cross_validate, ks_statistic
from tests.fake_data import fake_application


def test_ks_extremes():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == 1.0  # идеальное разделение
    assert ks_statistic(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) == 0.0  # перевёрнутая модель
    assert ks_statistic(y, np.full(6, 0.5)) == 0.0                           # константа ничего не различает


def test_gini_is_2auc_minus_1():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(size=500)
    m = classification_metrics(y, p)
    assert abs(m["gini"] - (2 * m["roc_auc"] - 1)) < 1e-12


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = (y * 0.3 + rng.uniform(size=2000)) / 1.3   # вероятности в [0, 1]
    auc = classification_metrics(y, p)["roc_auc"]
    lo, hi = bootstrap_auc_ci(y, p, n_boot=100)
    assert lo < auc < hi


def test_cv_fills_every_oof_row():
    X = pd.DataFrame({"a": np.arange(300.0)})
    y = pd.Series(np.r_[np.zeros(270), np.ones(30)].astype(int))
    res = cross_validate(lambda: DummyClassifier(strategy="prior"), X, y, X.iloc[:10], n_splits=5, verbose=False)
    assert len(res.fold_auc) == 5
    assert np.allclose(res.oof, 0.1)          # каждую строку предсказала модель, не видевшая её
    assert len(res.test_pred) == 10


def test_logreg_beats_random_on_fake_data():
    train = fake_application(2000, seed=1)
    test = fake_application(300, seed=2, with_target=False, id_start=900000)
    train, test = build_application_features(train, test)
    y = train.pop("TARGET")
    X = train.drop(columns=["SK_ID_CURR"])
    cats = categorical_columns(X)
    nums = [c for c in X.columns if c not in cats]
    res = cross_validate(lambda: make_logreg(nums, cats, C=0.1), X, y, test[X.columns], verbose=False)
    assert classification_metrics(y, res.oof)["roc_auc"] > 0.6
