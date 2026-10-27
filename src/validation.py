"""Кросс-валидация и метрики.

Схема: стратифицированный K-fold. Для каждой строки train получаем предсказание модели,
которая эту строку не видела (out-of-fold, OOF). Метрики считаем по всем OOF сразу.
Предсказание для test: среднее по моделям всех фолдов.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold

from src.config import N_SPLITS, SEED


def ks_statistic(y_true, y_score) -> float:
    """Колмогоров-Смирнов: максимальный разрыв между долей "плохих" (TPR) и долей "хороших" (FPR),
    которые оказываются выше порога, по всем порогам. 0 = модель не различает классы, 1 = идеально.
    roc_curve перебирает только уникальные пороги, поэтому одинаковые скоры обрабатываются верно."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def classification_metrics(y_true, y_score) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_score)
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "logloss": log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7)),
    }


def bootstrap_auc_ci(y_true, y_score, n_boot: int = 200, alpha: float = 0.05, seed: int = SEED):
    """Доверительный интервал AUC: пересэмплируем строки с возвращением и смотрим разброс."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue  # в выборку попал один класс, AUC не определён
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@dataclass
class CVResult:
    oof: np.ndarray
    test_pred: np.ndarray
    fold_auc: list[float]
    models: list = field(default_factory=list)

    @property
    def fold_auc_std(self) -> float:
        return float(np.std(self.fold_auc))


def cross_validate(
    make_model: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame | None = None,
    fit_fn: Callable | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    keep_models: bool = False,
    verbose: bool = True,
) -> CVResult:
    """make_model: функция без аргументов, которая возвращает НОВУЮ необученную модель.
    fit_fn(model, X_tr, y_tr): как обучать (по умолчанию model.fit). Нужна для early stopping."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test)) if X_test is not None else np.array([])
    fold_auc, models = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        model = make_model()
        if fit_fn is None:
            model.fit(X_tr, y_tr)
        else:
            model = fit_fn(model, X_tr, y_tr)

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        fold_auc.append(roc_auc_score(y_va, oof[va_idx]))
        if X_test is not None:
            test_pred += model.predict_proba(X_test)[:, 1] / n_splits
        if keep_models:
            models.append(model)
        if verbose:
            print(f"fold {fold}: AUC {fold_auc[-1]:.4f}")

    return CVResult(oof=oof, test_pred=test_pred, fold_auc=fold_auc, models=models)
