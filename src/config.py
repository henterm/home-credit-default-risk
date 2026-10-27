"""Пути и константы проекта. Все остальные модули берут настройки отсюда."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# CSV из Kaggle лежат в data/raw. Для тестов путь можно подменить переменной окружения.
DATA_DIR = Path(os.environ.get("HC_DATA_DIR", ROOT / "data" / "raw"))
OUTPUT_DIR = ROOT / "outputs"

# MLflow хранит прогоны в файле mlflow.db, а артефакты (csv, графики) в папке mlruns.
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = (ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT = "home-credit"

SEED = 42
N_SPLITS = 5

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

# В DAYS_EMPLOYED значение 365243 (~1000 лет) означает "не работает" (в основном пенсионеры).
DAYS_EMPLOYED_ANOMALY = 365243
