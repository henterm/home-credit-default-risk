"""Признаки из previous_application.csv (прошлые заявки клиента в Home Credit).

Результат: одна строка на SK_ID_CURR, признаки с префиксом "PREV_". TARGET не используется.

Что считается:
  1. PREV_COUNT и доли/количества по NAME_CONTRACT_STATUS (Approved, Refused, Canceled, Unused offer);
  2. агрегаты AMT_APPLICATION, AMT_CREDIT, AMT_ANNUITY, CNT_PAYMENT, DAYS_DECISION и др.;
  3. APP_CREDIT_RATIO = AMT_APPLICATION / AMT_CREDIT — сколько просили против сколько дали;
  4. 365243 в DAYS_FIRST_DRAWING / DAYS_FIRST_DUE / DAYS_LAST_DUE* / DAYS_TERMINATION -> NaN до агрегации
     (это та же заглушка, что в DAYS_EMPLOYED: «события не было / дата в далёком будущем»);
  5. отдельные агрегаты по одобренным (PREV_APPROVED_*) и отказанным (PREV_REFUSED_*) заявкам.
"""
import pandas as pd

from src.config import DAYS_EMPLOYED_ANOMALY, ID_COL
from src.features.common import add_flags, aggregate, finalize, safe_div

PREFIX = "PREV_"

DAY_COLS_WITH_ANOMALY = [
    "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE", "DAYS_TERMINATION",
]

STATUS_FLAGS = {
    "APPROVED": "Approved",
    "REFUSED": "Refused",
    "CANCELED": "Canceled",
    "UNUSED": "Unused offer",
}
CONTRACT_FLAGS = {
    "CONTRACT_CASH": "Cash loans",
    "CONTRACT_CONSUMER": "Consumer loans",
    "CONTRACT_REVOLVING": "Revolving loans",
}
EXTRA_FLAGS = {
    # высокая ставка — банк сам считал заявку рискованной
    "NAME_YIELD_GROUP": {"YIELD_HIGH": "high"},
    "NAME_CLIENT_TYPE": {"CLIENT_NEW": "New"},
}

ALL_AGGS = {
    "AMT_APPLICATION": ["mean", "max", "min", "sum"],
    "AMT_CREDIT": ["mean", "max", "min", "sum"],
    "AMT_ANNUITY": ["mean", "max", "min"],
    "AMT_DOWN_PAYMENT": ["mean", "max"],
    "AMT_GOODS_PRICE": ["mean", "max"],
    "CNT_PAYMENT": ["mean", "max", "sum"],
    "DAYS_DECISION": ["mean", "max", "min"],   # max = самая свежая заявка
    "DAYS_FIRST_DRAWING": ["mean", "max"],
    "DAYS_FIRST_DUE": ["mean", "max"],
    "DAYS_LAST_DUE_1ST_VERSION": ["mean", "max"],
    "DAYS_LAST_DUE": ["mean", "max"],
    "DAYS_TERMINATION": ["mean", "max"],
    "NFLAG_INSURED_ON_APPROVAL": ["mean"],
    # признаки уровня заявки из _row_features
    "APP_CREDIT_RATIO": ["mean", "max", "min"],
    "CREDIT_GOODS_RATIO": ["mean", "max"],
    "ANNUITY_CREDIT_RATIO": ["mean", "max"],
    "OVERPAY_RATIO": ["mean", "max"],
    "EARLY_REPAY_DAYS": ["mean", "max"],
}

APPROVED_AGGS = {
    "AMT_CREDIT": ["mean", "max", "sum"],
    "AMT_ANNUITY": ["mean", "max", "sum"],
    "CNT_PAYMENT": ["mean", "max"],
    "DAYS_DECISION": ["max", "min"],
    "DAYS_LAST_DUE_1ST_VERSION": ["max"],       # > 0: кредит ещё не погашен по графику
    "APP_CREDIT_RATIO": ["mean", "max"],
    "OVERPAY_RATIO": ["mean", "max"],
    "EARLY_REPAY_DAYS": ["mean"],
}
REFUSED_AGGS = {
    "AMT_APPLICATION": ["mean", "max", "sum"],
    "DAYS_DECISION": ["max", "mean"],           # max = как давно был последний отказ
    "CNT_PAYMENT": ["mean"],
}


def _row_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    # заглушка 365243 -> NaN (иначе mean/max по датам уезжают на 1000 лет)
    for col in DAY_COLS_WITH_ANOMALY:
        if col in p.columns:
            p[col] = p[col].where(p[col] != DAYS_EMPLOYED_ANOMALY)

    cols = set(p.columns)
    if {"AMT_APPLICATION", "AMT_CREDIT"} <= cols:
        # AMT_CREDIT = 0 у отменённых заявок -> NaN
        p["APP_CREDIT_RATIO"] = safe_div(p["AMT_APPLICATION"], p["AMT_CREDIT"])
    if {"AMT_CREDIT", "AMT_GOODS_PRICE"} <= cols:
        p["CREDIT_GOODS_RATIO"] = safe_div(p["AMT_CREDIT"], p["AMT_GOODS_PRICE"])
    if {"AMT_ANNUITY", "AMT_CREDIT"} <= cols:
        p["ANNUITY_CREDIT_RATIO"] = safe_div(p["AMT_ANNUITY"], p["AMT_CREDIT"])
    if {"AMT_ANNUITY", "CNT_PAYMENT", "AMT_CREDIT"} <= cols:
        # Сколько клиент суммарно отдаёт на рубль кредита: прокси процентной ставки
        p["OVERPAY_RATIO"] = safe_div(p["AMT_ANNUITY"] * p["CNT_PAYMENT"], p["AMT_CREDIT"])
    if {"DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE"} <= cols:
        # > 0: последний платёж случился раньше, чем планировалось изначально
        p["EARLY_REPAY_DAYS"] = p["DAYS_LAST_DUE_1ST_VERSION"] - p["DAYS_LAST_DUE"]
    return p


def build_previous_features(previous: pd.DataFrame) -> pd.DataFrame:
    p = _row_features(previous)
    flag_cols = add_flags(p, "NAME_CONTRACT_STATUS", STATUS_FLAGS)
    flag_cols += add_flags(p, "NAME_CONTRACT_TYPE", CONTRACT_FLAGS)
    for column, values in EXTRA_FLAGS.items():
        flag_cols += add_flags(p, column, values)

    aggs = dict(ALL_AGGS)
    for col in flag_cols:
        aggs[col] = ["mean", "sum"]
    out = aggregate(p, ID_COL, aggs, PREFIX)
    out[f"{PREFIX}COUNT"] = p.groupby(ID_COL).size()
    out = out.rename(columns={
        **{f"{PREFIX}{c}_MEAN": f"{PREFIX}{c}_SHARE" for c in flag_cols},
        **{f"{PREFIX}{c}_SUM": f"{PREFIX}{c}_COUNT" for c in flag_cols},
    })

    # отдельно одобренные и отказанные
    if "NAME_CONTRACT_STATUS" in p.columns:
        approved = p[p["NAME_CONTRACT_STATUS"] == "Approved"]
        refused = p[p["NAME_CONTRACT_STATUS"] == "Refused"]
        out = out.join(aggregate(approved, ID_COL, APPROVED_AGGS, f"{PREFIX}APPROVED_"), how="left")
        out = out.join(aggregate(refused, ID_COL, REFUSED_AGGS, f"{PREFIX}REFUSED_"), how="left")
        # Насколько крупнее были одобренные кредиты по сравнению с тем, о чём просили в отказанных
        a, r = f"{PREFIX}APPROVED_AMT_CREDIT_MEAN", f"{PREFIX}REFUSED_AMT_APPLICATION_MEAN"
        if {a, r} <= set(out.columns):
            out[f"{PREFIX}REFUSED_TO_APPROVED_AMT_RATIO"] = safe_div(out[r], out[a])

    # Отказы за последние 2 года — более свежий сигнал, чем доля отказов за всю жизнь
    if {"DAYS_DECISION", "REFUSED"} <= set(p.columns):
        recent = p[p["DAYS_DECISION"] >= -730]
        out[f"{PREFIX}LAST2Y_REFUSED_SHARE"] = recent.groupby(ID_COL)["REFUSED"].mean()
        out[f"{PREFIX}LAST2Y_COUNT"] = recent.groupby(ID_COL).size()
        out[f"{PREFIX}LAST2Y_COUNT"] = out[f"{PREFIX}LAST2Y_COUNT"].fillna(0)

    out.index.name = ID_COL
    return finalize(out, ID_COL)