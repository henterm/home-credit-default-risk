"""Признаки из bureau.csv (кредиты клиента в других банках по данным бюро кредитных историй).

Результат: одна строка на SK_ID_CURR, столбец SK_ID_CURR + признаки с префиксом "BUREAU_".
TARGET не используется: агрегаты — это детерминированная свёртка истории клиента,
она не знает, кто дефолтнул. Поэтому её можно посчитать один раз на всей таблице
(train и test клиенты вперемешку) без утечки. Target encoding, наоборот, кладёт в признак
среднее TARGET по группе — без out-of-fold схемы строка «видит» собственный таргет.

Что считается:
  1. mean / max / min / sum по DAYS_CREDIT, AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT, CREDIT_DAY_OVERDUE
     и более узкие агрегаты по остальным числовым полям;
  2. MultiIndex ("DAYS_CREDIT", "mean") склеен в "BUREAU_DAYS_CREDIT_MEAN";
  3. BUREAU_COUNT — число кредитов в бюро;
  4. доли статусов CREDIT_ACTIVE (BUREAU_ACTIVE_SHARE и др.) и типов CREDIT_TYPE;
  5. долг / сумма кредита: на уровне строки (потом mean/max) и на уровне клиента
     (сумма долгов / сумма кредитов — взвешенная версия, крупные кредиты весят больше);
  6. bureau_balance (опционально): сначала свёртка по SK_ID_BUREAU, потом к клиенту.

Клиенты без записей в бюро в результате отсутствуют. После merge(how="left") в train.py
у них все BUREAU_* = NaN, а BUREAU_COUNT дозаполняется нулём (кредитов действительно 0).
"""
import pandas as pd

from src.config import ID_COL
from src.features.common import add_flags, aggregate, finalize, safe_div

PREFIX = "BUREAU_"

# Базовые четыре колонки - полный набор статистик
BASE_AGGS = {
    "DAYS_CREDIT": ["mean", "max", "min", "sum"],
    "AMT_CREDIT_SUM": ["mean", "max", "min", "sum"],
    "AMT_CREDIT_SUM_DEBT": ["mean", "max", "min", "sum"],
    "CREDIT_DAY_OVERDUE": ["mean", "max", "min", "sum"],
}

# Остальные числовые поля: только статистики, у которых есть смысл.
EXTRA_AGGS = {
    "DAYS_CREDIT_ENDDATE": ["mean", "max", "min"],   # когда закончатся текущие кредиты
    "DAYS_ENDDATE_FACT": ["mean", "max"],            # когда закрылись закрытые
    "DAYS_CREDIT_UPDATE": ["mean", "max"],           # свежесть информации в бюро
    "AMT_CREDIT_MAX_OVERDUE": ["mean", "max"],
    "AMT_CREDIT_SUM_OVERDUE": ["mean", "max", "sum"],
    "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
    "AMT_ANNUITY": ["mean", "max", "sum"],
    "CNT_CREDIT_PROLONG": ["sum"],
    # признаки уровня строки, созданные в _row_features
    "DEBT_CREDIT_RATIO": ["mean", "max"],
    "OVERDUE_DEBT_RATIO": ["mean", "max"],
    "PLANNED_DURATION": ["mean"],
    "ENDDATE_SHIFT": ["mean", "max"],
    "IS_OVERDUE_NOW": ["mean", "sum"],
}

STATUS_FLAGS = {"ACTIVE": "Active", "CLOSED": "Closed", "SOLD": "Sold", "BAD_DEBT": "Bad debt"}
TYPE_FLAGS = {
    "TYPE_CONSUMER": "Consumer credit",
    "TYPE_CARD": "Credit card",
    "TYPE_CAR": "Car loan",
    "TYPE_MORTGAGE": "Mortgage",
    "TYPE_MICROLOAN": "Microloan",
}

# Агрегаты только по активным кредитам: текущая долговая нагрузка клиента.
ACTIVE_AGGS = {
    "AMT_CREDIT_SUM": ["sum", "mean"],
    "AMT_CREDIT_SUM_DEBT": ["sum", "max"],
    "DAYS_CREDIT": ["max"],
    "DAYS_CREDIT_ENDDATE": ["max"],
    "DEBT_CREDIT_RATIO": ["mean", "max"],
}


def _row_features(b: pd.DataFrame) -> pd.DataFrame:
    """Признаки уровня одного кредита — до агрегации"""
    b = b.copy()
    if {"AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"} <= set(b.columns):
        # AMT_CREDIT_SUM бывает 0 -> NaN вместо inf
        b["DEBT_CREDIT_RATIO"] = safe_div(b["AMT_CREDIT_SUM_DEBT"], b["AMT_CREDIT_SUM"])
    if {"AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_SUM_DEBT"} <= set(b.columns):
        b["OVERDUE_DEBT_RATIO"] = safe_div(b["AMT_CREDIT_SUM_OVERDUE"], b["AMT_CREDIT_SUM_DEBT"])
    if {"DAYS_CREDIT_ENDDATE", "DAYS_CREDIT"} <= set(b.columns):
        b["PLANNED_DURATION"] = b["DAYS_CREDIT_ENDDATE"] - b["DAYS_CREDIT"]
    if {"DAYS_ENDDATE_FACT", "DAYS_CREDIT_ENDDATE"} <= set(b.columns):
        # > 0: закрыл позже плана (тянул), < 0: досрочно
        b["ENDDATE_SHIFT"] = b["DAYS_ENDDATE_FACT"] - b["DAYS_CREDIT_ENDDATE"]
    if "CREDIT_DAY_OVERDUE" in b.columns:
        b["IS_OVERDUE_NOW"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype("float32")
    return b


def build_bureau_balance_features(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """первый шаг: одна строка на SK_ID_BUREAU.

    STATUS: C — закрыт, X — нет данных, 0 — без просрочки, 1..5 — просрочка 1-30, 31-60, ... 120+ дней.
    """
    bb = bureau_balance.copy()
    status = bb["STATUS"].astype(str)
    dpd = status.map({str(k): float(k) for k in range(6)})
    bb["BB_DPD_FLAG"] = (dpd >= 1).astype("float32")       # месяц с любой просрочкой
    bb["BB_DPD60_FLAG"] = (dpd >= 2).astype("float32")     # просрочка больше 30 дней
    bb["BB_DPD_LEVEL"] = dpd                               # худшая стадия просрочки
    bb["BB_UNKNOWN_FLAG"] = (status == "X").astype("float32")
    recent = bb["MONTHS_BALANCE"] >= -12
    bb["BB_DPD_FLAG_12M"] = bb["BB_DPD_FLAG"].where(recent)

    per_credit = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS=("MONTHS_BALANCE", "size"),
        BB_DPD_SHARE=("BB_DPD_FLAG", "mean"),
        BB_DPD60_SHARE=("BB_DPD60_FLAG", "mean"),
        BB_DPD_WORST=("BB_DPD_LEVEL", "max"),
        BB_UNKNOWN_SHARE=("BB_UNKNOWN_FLAG", "mean"),
        BB_DPD_SHARE_12M=("BB_DPD_FLAG_12M", "mean"),
    )
    return per_credit.reset_index()


def build_bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame | None = None) -> pd.DataFrame:
    b = _row_features(bureau)
    status_cols = add_flags(b, "CREDIT_ACTIVE", STATUS_FLAGS)
    type_cols = add_flags(b, "CREDIT_TYPE", TYPE_FLAGS)

    aggs = {**BASE_AGGS, **EXTRA_AGGS}
    for col in status_cols + type_cols:
        aggs[col] = ["mean", "sum"]                         # доля и количество
    out = aggregate(b, ID_COL, aggs, PREFIX)

    # число кредитов и доли статусов под понятными именами
    out[f"{PREFIX}COUNT"] = b.groupby(ID_COL).size()
    renames = {}
    for col in status_cols + type_cols:
        renames[f"{PREFIX}{col}_MEAN"] = f"{PREFIX}{col}_SHARE"
        renames[f"{PREFIX}{col}_SUM"] = f"{PREFIX}{col}_COUNT"
    out = out.rename(columns=renames)

    # Взвешенная долговая нагрузка: сумма долгов / сумма кредитов
    debt_sum, credit_sum = f"{PREFIX}AMT_CREDIT_SUM_DEBT_SUM", f"{PREFIX}AMT_CREDIT_SUM_SUM"
    if {debt_sum, credit_sum} <= set(out.columns):
        out[f"{PREFIX}TOTAL_DEBT_CREDIT_RATIO"] = safe_div(out[debt_sum], out[credit_sum])

    # Только активные кредиты
    if "CREDIT_ACTIVE" in b.columns:
        active = b[b["CREDIT_ACTIVE"] == "Active"]
        act = aggregate(active, ID_COL, ACTIVE_AGGS, f"{PREFIX}ACTIVE_")
        out = out.join(act, how="left")
        # Клиент есть в бюро, но активных кредитов нет: текущий долг действительно 0
        for col in (f"{PREFIX}ACTIVE_AMT_CREDIT_SUM_SUM", f"{PREFIX}ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"):
            if col in out.columns:
                out[col] = out[col].fillna(0)

    # bureau_balance -> кредит -> клиент
    if bureau_balance is not None and "SK_ID_BUREAU" in b.columns:
        per_credit = build_bureau_balance_features(bureau_balance)
        bb = b[[ID_COL, "SK_ID_BUREAU"]].merge(per_credit, on="SK_ID_BUREAU", how="left")
        bb_aggs = {
            "BB_MONTHS": ["mean", "max", "sum"],
            "BB_DPD_SHARE": ["mean", "max"],
            "BB_DPD60_SHARE": ["mean", "max"],
            "BB_DPD_WORST": ["max"],
            "BB_UNKNOWN_SHARE": ["mean"],
            "BB_DPD_SHARE_12M": ["mean", "max"],
        }
        out = out.join(aggregate(bb, ID_COL, bb_aggs, PREFIX), how="left")

    out.index.name = ID_COL
    return finalize(out, ID_COL)