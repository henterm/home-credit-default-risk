# Home Credit Default Risk: модель вероятности дефолта

Задача кредитного скоринга: по заявке клиента предсказать вероятность того, что у него
возникнут трудности с выплатами (TARGET = 1). Данные: соревнование
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) на Kaggle,
307 511 заявок в train, доля дефолтов 8.1%.

## Что сделано

- **EDA** (`notebooks/01_eda.ipynb`): пропуски, аномалия `DAYS_EMPLOYED = 365243`
  (у таких клиентов дефолтов 5.4% против 8.7%, сохранена как отдельный флаг),
  связь возраста и внешних скорингов с дефолтом. Структура EDA опирается на кернел
  Will Koehrsen «Start Here: A Gentle Introduction».
- **Признаки** (`src/preprocessing.py`): долговая нагрузка (кредит и аннуитет к доходу),
  срок кредита, стаж к возрасту, агрегаты внешних скорингов.
- **Признаки из бюро кредитных историй** (`src/features/bureau.py`): агрегаты по кредитам клиента
  в других банках (число кредитов, доли статусов, отношение долга к сумме кредита). Самые сильные:
  максимальная и средняя долговая нагрузка, в том числе только по активным кредитам.
- **Валидация** (`src/validation.py`): стратифицированный 5-fold, все метрики по
  out-of-fold предсказаниям, бутстрэп-интервал для ROC AUC, разброс по фолдам.
  Все преобразования (импутация, масштабирование, кодирование) внутри sklearn Pipeline,
  поэтому обучаются только на тренировочных фолдах. Early stopping у LightGBM
  на отдельном куске обучающего фолда, а не на валидационном.
- **Трекинг экспериментов**: каждый прогон пишется в MLflow (параметры, метрики, git-коммит,
  набор признаков, артефакты).

## Результаты

OOF-метрики на train (5-fold). Заполняется по мере прогонов.

| Модель | Признаки | ROC AUC [95% ДИ] | Gini | KS | PR AUC |
|---|---|---|---|---|---|
| Логистическая регрессия | application | 0.750 [0.748; 0.754] | 0.500 | 0.371 | 0.231 |
| LightGBM | application | 0.768 [0.765; 0.771] | 0.535 | 0.400 | 0.254 |
| LightGBM | application + bureau | 0.777 [0.774; 0.780] | 0.554 | 0.415 | 0.267 |

Признаки из бюро кредитных историй дали +0.009 ROC AUC; доверительные интервалы моделей
с бюро и без него не пересекаются.

## Как запустить

```bash
python -m venv venv
venv\Scripts\activate            # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
# положить CSV из Kaggle в data/raw/
python -m src.train --model logreg --sample-frac 0.2   # быстрый прогон
python -m src.train --model lgbm
mlflow ui --backend-store-uri sqlite:///mlflow.db       # http://127.0.0.1:5000
python -m pytest
```

## Структура

```
src/
  config.py          пути и константы
  data.py            загрузка таблиц
  preprocessing.py   очистка и признаки по заявке
  features/          агрегаты по bureau, previous_application
  models.py          logreg (Pipeline) и LightGBM
  validation.py      кросс-валидация, ROC AUC, Gini, KS, бутстрэп
  train.py           CLI: обучение, метрики, запись в MLflow
tests/               проверки признаков и валидации на синтетических данных
notebooks/           EDA и разбор результатов
```
