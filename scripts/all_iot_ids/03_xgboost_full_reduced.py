import os
import time
import joblib

from xgboost import XGBClassifier

import config
from utils import (
    ensure_dirs, set_all_seeds, load_prepared_npz, compute_metrics,
    benchmark_sklearn_model, model_file_size_mb, save_result
)

set_all_seeds(config.RANDOM_SEED)
ensure_dirs(config.RESULTS_DIR, config.MODELS_DIR)

def train_eval_xgb(feature_set_name, npz_name):
    data = load_prepared_npz(os.path.join(config.ARTIFACT_DIR, npz_name))
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    class_names = data["class_names"]

    params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    model = XGBClassifier(**params)

    start = time.perf_counter()
    try:
        if config.USE_GPU_FOR_XGBOOST:
            model.set_params(device="cuda")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    except Exception as e:
        print(f"GPU XGBoost failed for {feature_set_name}; falling back to CPU.")
        print("Reason:", str(e))
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    training_time = time.perf_counter() - start

    y_pred = model.predict(X_test)

    model_path = os.path.join(config.MODELS_DIR, f"xgboost_{feature_set_name.lower()}.joblib")
    joblib.dump(model, model_path)

    metrics = compute_metrics(
        y_test, y_pred, class_names, benign_label_name=config.BENIGN_LABEL_NAME
    )

    efficiency = benchmark_sklearn_model(
        model, X_test,
        sample_size=config.BENCHMARK_SAMPLE_SIZE,
        batch_size=config.BENCHMARK_BATCH_SIZE,
    )
    efficiency["training_time_sec"] = float(training_time)
    efficiency["model_size_mb"] = float(model_file_size_mb(model_path))

    result = {
        "model": "XGBoost",
        "feature_set": feature_set_name,
        "features_used": data["feature_names"],
        "metrics": metrics,
        "efficiency": efficiency,
    }

    print(f"\n=== XGBoost {feature_set_name} ===")
    print(metrics["classification_report"])
    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])

    save_result(result, config.RESULTS_DIR, "XGBoost", feature_set_name)

train_eval_xgb("Full", "full_tree.npz")
train_eval_xgb("Reduced", "reduced_tree.npz")
