"""Обучение и оценка с записью в MLflow.
"""
import argparse
import json
import subprocess
import time

import pandas as pd

from src.config import (ID_COL, MLFLOW_ARTIFACTS, MLFLOW_EXPERIMENT, MLFLOW_URI,
                        N_SPLITS, OUTPUT_DIR, ROOT, TARGET)
from src.data import load_application, read_table
from src.models import (LGBM_PARAMS, feature_importance, fit_lgbm_early_stopping,
                        make_lgbm, make_logreg)
from src.preprocessing import build_application_features, categorical_columns, to_lgbm_categories
from src.validation import bootstrap_auc_ci, classification_metrics, cross_validate

FEATURE_SETS = ("app", "bureau", "previous")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Home Credit: CV + MLflow")
    p.add_argument("--model", choices=["logreg", "lgbm"], default="logreg")
    p.add_argument("--features", default="app", help="через запятую: app,bureau,previous")
    p.add_argument("--sample-frac", type=float, default=None, help="доля train для быстрого прогона")
    p.add_argument("--C", type=float, default=0.01, help="регуляризация logreg (меньше = сильнее)")
    p.add_argument("--n-splits", type=int, default=N_SPLITS)
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def build_dataset(feature_sets: list[str], sample_frac: float | None):
    train, test = load_application(sample_frac)
    train, test = build_application_features(train, test)

    extra = []
    if "bureau" in feature_sets:
        from src.features.bureau import build_bureau_features
        extra.append(build_bureau_features(read_table("bureau")))
    if "previous" in feature_sets:
        from src.features.previous import build_previous_features
        extra.append(build_previous_features(read_table("previous_application")))
    for feats in extra:
        train = train.merge(feats, on=ID_COL, how="left")
        test = test.merge(feats, on=ID_COL, how="left")

    y = train[TARGET].astype(int)
    X = train.drop(columns=[TARGET, ID_COL])
    X_test = test[X.columns]  # тот же набор и порядок столбцов, что в train
    return X, y, X_test, test[ID_COL]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "no-git"


def main(argv=None):
    args = parse_args(argv)
    feature_sets = [f.strip() for f in args.features.split(",") if f.strip()]
    unknown = set(feature_sets) - set(FEATURE_SETS)
    if unknown:
        raise SystemExit(f"Неизвестные наборы признаков: {unknown}. Доступны: {FEATURE_SETS}")

    t0 = time.time()
    X, y, X_test, test_ids = build_dataset(feature_sets, args.sample_frac)
    cat_cols = categorical_columns(X)
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"train {X.shape}, test {X_test.shape}, default rate {y.mean():.4f}, "
          f"categorical {len(cat_cols)}")

    if args.model == "logreg":
        params = {"C": args.C}
        result = cross_validate(lambda: make_logreg(num_cols, cat_cols, C=args.C),
                                X, y, X_test, n_splits=args.n_splits)
        importance = None
    else:
        params = dict(LGBM_PARAMS)
        X, X_test = to_lgbm_categories(X, X_test, cat_cols)
        result = cross_validate(make_lgbm, X, y, X_test, fit_fn=fit_lgbm_early_stopping,
                                n_splits=args.n_splits, keep_models=True)
        importance = feature_importance(result.models, list(X.columns))
        params["mean_best_iteration"] = round(
            sum(m.best_iteration_ for m in result.models) / len(result.models))

    metrics = classification_metrics(y, result.oof)
    metrics["auc_ci_low"], metrics["auc_ci_high"] = bootstrap_auc_ci(y, result.oof)
    metrics["auc_fold_std"] = result.fold_auc_std
    metrics["minutes"] = (time.time() - t0) / 60

    print("\nOOF-метрики:")
    for k, v in metrics.items():
        print(f"  {k:<13} {v:.4f}")

    # Сохраняем результаты в outputs/<имя прогона>/
    run_name = args.run_name or f"{args.model}_{'+'.join(feature_sets)}" + (
        f"_s{args.sample_frac}" if args.sample_frac else "")
    out = OUTPUT_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: test_ids.values, TARGET: result.test_pred}).to_csv(out / "submission.csv", index=False)
    pd.DataFrame({"y": y.values, "oof": result.oof}).to_csv(out / "oof.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if importance is not None:
        importance.to_csv(out / "feature_importance.csv", index=False)
        print("\nТоп-15 признаков (gain):")
        print(importance.head(15).to_string(index=False))
    print(f"\nФайлы: {out}")

    if not args.no_mlflow:
        log_to_mlflow(run_name, args, feature_sets, params, metrics, X.shape, out)


def log_to_mlflow(run_name, args, feature_sets, params, metrics, shape, out_dir):
    try:
        import mlflow
    except ImportError:
        print("mlflow не установлен, прогон не записан (pip install mlflow).")
        return
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"model": args.model, "features": "+".join(feature_sets),
                         "git_commit": git_commit()})
        mlflow.log_params({**params, "n_splits": args.n_splits, "sample_frac": args.sample_frac,
                           "n_rows": shape[0], "n_features": shape[1]})
        mlflow.log_metrics(metrics)
        for f in out_dir.iterdir():
            if f.name != "submission.csv":
                mlflow.log_artifact(str(f))
    print(f"Записано в MLflow, эксперимент '{MLFLOW_EXPERIMENT}'. Посмотреть: mlflow ui "
          f"--backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
