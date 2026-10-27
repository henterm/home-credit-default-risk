"""Шаг 4: можно ли верить скору как вероятности, где ставить порог и почему модель так решила.

Все функции работают с OOF-предсказаниями (outputs/<прогон>/oof.csv): для каждой строки train
скор получен моделью, которая эту строку не видела, поэтому это честная оценка «как на новых клиентах».
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from src.config import SEED

# ---------------------------------------------------------------- калибровка

def calibration_table(y_true, y_prob, n_bins: int = 20) -> pd.DataFrame:
    """Корзины по квантилям скора: средний скор против фактической доли дефолтов.

    Квантильные корзины, а не равные по ширине: скоры сосредоточены у нуля, и в
    равных корзинах [0.5, 1] оказались бы единицы клиентов.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    return (df.groupby("bin", observed=True)
              .agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size"))
              .reset_index(drop=True))


def calibration_summary(y_true, y_prob, n_bins: int = 20) -> dict[str, float]:
    """Brier, Brier skill score и ECE.

    Brier = mean((p - y)^2). Сам по себе мало что говорит, поэтому сравниваем с наивным
    прогнозом «всем базовую ставку»: Brier_ref = p̄(1 - p̄). BSS = 1 - Brier / Brier_ref.
    ECE = средневзвешенный |доля дефолтов - средний скор| по корзинам: на сколько п.п.
    в среднем ошибается скор, если читать его как вероятность.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    base = y_true.mean()
    brier = brier_score_loss(y_true, y_prob)
    brier_ref = base * (1 - base)
    table = calibration_table(y_true, y_prob, n_bins)
    ece = float(np.average(np.abs(table["frac_pos"] - table["mean_pred"]), weights=table["n"]))
    return {
        "brier": float(brier),
        "brier_ref": float(brier_ref),
        "brier_skill": float(1 - brier / brier_ref),
        "ece": ece,
        "mean_pred": float(y_prob.mean()),
        "default_rate": float(base),
    }


def crossfit_isotonic(y_true, y_prob, n_splits: int = 5, seed: int = SEED) -> np.ndarray:
    """Изотоническая перекалибровка OOF-скора, тоже out-of-fold.

    Если обучить калибратор на всех OOF и на них же замерить Brier, получится заниженная
    ошибка (калибратор подгоняется под шум). Поэтому калибратор каждой части учится на остальных.
    Isotonic монотонна, поэтому AUC почти не меняется — меняется только шкала скора.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    out = np.zeros_like(y_prob, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(y_prob.reshape(-1, 1), y_true):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(y_prob[tr], y_true[tr])
        out[va] = iso.predict(y_prob[va])
    return out


# ---------------------------------------------------------------- порог по цене ошибки

def bayes_threshold(cost_fn: float, cost_fp: float) -> float:
    """Оптимальный порог для ИДЕАЛЬНО откалиброванного скора.

    Отказываем, если ожидаемая цена одобрения больше цены отказа:
    p·C_FN > (1 - p)·C_FP  =>  p > C_FP / (C_FP + C_FN). При C_FN = 5, C_FP = 1 это 1/6 ≈ 0.167.
    """
    return cost_fp / (cost_fp + cost_fn)


def cost_curve(y_true, y_prob, cost_fn: float = 5.0, cost_fp: float = 1.0,
               thresholds=None) -> pd.DataFrame:
    """Стоимость решений на сетке порогов. Отказ, если скор >= порога.

    FN — одобрили будущего дефолтника (стоит cost_fn), FP — отказали хорошему клиенту (cost_fp).
    Считается векторно через сортировку, поэтому 2000 порогов на 300k строк — доли секунды.
    """
    y_true, y_prob = np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float)
    if thresholds is None:
        thresholds = np.unique(np.quantile(y_prob, np.linspace(0, 1, 2001)))
    thresholds = np.asarray(thresholds, dtype=float)
    pos = np.sort(y_prob[y_true == 1])
    neg = np.sort(y_prob[y_true == 0])
    tp = len(pos) - np.searchsorted(pos, thresholds, side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="left")
    fn = len(pos) - tp
    tn = len(neg) - fp
    cost = cost_fn * fn + cost_fp * fp
    return pd.DataFrame({
        "threshold": thresholds, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cost": cost, "cost_per_client": cost / len(y_true),
        "approval_rate": (fn + tn) / len(y_true),
    })


def metrics_at_threshold(y_true, y_prob, threshold: float, cost_fn: float = 5.0,
                         cost_fp: float = 1.0) -> dict[str, float]:
    """Метрики класса «дефолт» при правиле «отказ, если скор >= threshold»."""
    y_true = np.asarray(y_true).astype(int)
    reject = np.asarray(y_prob) >= threshold
    tp = int(np.sum(reject & (y_true == 1)))
    fp = int(np.sum(reject & (y_true == 0)))
    fn = int(np.sum(~reject & (y_true == 1)))
    tn = int(np.sum(~reject & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = len(y_true)
    approve_all_cost = cost_fn * (tp + fn)          # без модели: одобряем всех
    cost = cost_fn * fn + cost_fp * fp
    return {
        "threshold": float(threshold),
        "recall": recall,                           # доля пойманных дефолтов
        "precision": precision,                     # доля дефолтов среди отказанных
        "f1": f1,
        "approval_rate": (fn + tn) / n,             # доля одобренных заявок
        "default_rate_approved": fn / (fn + tn) if fn + tn else 0.0,
        "cost": float(cost),
        "cost_per_client": cost / n,
        "saving_vs_approve_all": 1 - cost / approve_all_cost if approve_all_cost else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def best_cost_threshold(y_true, y_prob, cost_fn: float = 5.0, cost_fp: float = 1.0) -> dict[str, float]:
    curve = cost_curve(y_true, y_prob, cost_fn, cost_fp)
    t = float(curve.loc[curve["cost"].idxmin(), "threshold"])
    return metrics_at_threshold(y_true, y_prob, t, cost_fn, cost_fp)


def best_f1_threshold(y_true, y_prob) -> float:
    """Порог, максимизирующий F1, — для сравнения: F1 не знает про цену ошибок."""
    curve = cost_curve(y_true, y_prob)
    prec = curve["tp"] / (curve["tp"] + curve["fp"]).replace(0, np.nan)
    rec = curve["tp"] / (curve["tp"] + curve["fn"])
    f1 = 2 * prec * rec / (prec + rec)
    return float(curve.loc[f1.idxmax(), "threshold"])


# ---------------------------------------------------------------- SHAP

def lgbm_shap(model, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """SHAP-значения LightGBM через встроенный TreeSHAP (pred_contrib=True), без пакета shap.

    Возвращает (contrib, base): contrib[i, j] — вклад признака j в скор клиента i
    в шкале log-odds; base — ожидаемое значение (последний столбец pred_contrib).
    Для каждой строки base + сумма вкладов = логит предсказанной вероятности.
    """
    raw = model.predict(X, pred_contrib=True)
    raw = np.asarray(raw)
    contrib = pd.DataFrame(raw[:, :-1], columns=X.columns, index=X.index)
    return contrib, raw[:, -1]


def shap_importance(contrib: pd.DataFrame) -> pd.DataFrame:
    """Глобальная важность: средний |вклад|. В отличие от gain показывает величину влияния на скор."""
    return (contrib.abs().mean().sort_values(ascending=False)
            .rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}))


def explain_client(contrib: pd.DataFrame, X: pd.DataFrame, base: np.ndarray, row, top: int = 10) -> pd.DataFrame:
    """Топ признаков, которые сдвинули скор одного клиента. row — метка индекса."""
    c = contrib.loc[row]
    order = c.abs().sort_values(ascending=False).index[:top]
    i = contrib.index.get_loc(row)
    logit = base[i] + c.sum()
    table = pd.DataFrame({
        "feature": order,
        "value": [X.loc[row, f] for f in order],
        "shap_logodds": c[order].values,
    })
    table["direction"] = np.where(table["shap_logodds"] > 0, "↑ риск", "↓ риск")
    table.attrs["base_logodds"] = float(base[i])
    table.attrs["prob"] = float(1 / (1 + np.exp(-logit)))
    return table
