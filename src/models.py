"""Модели. Всё, что учится на данных (импутация, масштаб, кодировки), сидит внутри Pipeline,
поэтому на кросс-валидации fit делается только на обучающих фолдах."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import SEED


def make_logreg(num_cols: list[str], cat_cols: list[str], C: float = 0.01) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # NaN -> медиана столбца (из train-фолда)
        ("scale", StandardScaler()),                     # (x - mean) / std, тоже из train-фолда
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),  # новая категория в test -> все нули
    ])
    prep = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(C=C, max_iter=2000)),
    ])


LGBM_PARAMS = dict(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.5,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def make_lgbm(**overrides):
    import lightgbm as lgb  # импорт внутри, чтобы logreg работал и без lightgbm
    return lgb.LGBMClassifier(**{**LGBM_PARAMS, **overrides})


def fit_lgbm_early_stopping(model, X: pd.DataFrame, y: pd.Series, es_frac: float = 0.1, patience: int = 200):
    """Early stopping на отдельном кусочке ОБУЧАЮЩЕГО фолда.

    Если останавливаться по валидационному фолду, мы подбираем число деревьев
    под те же данные, на которых потом считаем AUC, и оценка становится чуть завышенной.
    """
    import lightgbm as lgb
    X_fit, X_es, y_fit, y_es = train_test_split(
        X, y, test_size=es_frac, stratify=y, random_state=SEED
    )
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_es, y_es)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def feature_importance(models: list, columns: list[str]) -> pd.DataFrame:
    """Средняя важность (gain) по фолдам. Для LightGBM."""
    imp = np.mean(
        [m.booster_.feature_importance(importance_type="gain") for m in models], axis=0
    )
    return (
        pd.DataFrame({"feature": columns, "gain": imp})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
