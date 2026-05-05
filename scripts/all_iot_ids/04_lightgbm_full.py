import os
import time
import joblib

import config
from utils import (
    ensure_dirs, set_all_seeds, load_prepared_npz, compute_metrics,
    benchmark_sklearn_model, model_file_size_mb, save_result
)

set_all_seeds(config.RANDOM_SEED)
ensure_dirs(config.RESULTS_DIR, config.MODELS_DIR)

try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise ImportError(
        "LightGBM is not installed. In Kaggle, try: !pip install lightgbm"
    ) from e

data = load_prepared_npz(os.path.join(config.ARTIFACT_DIR, "full_tree.npz"))
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
class_names = data["class_names"]

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=config.RANDOM_SEED,
    n_jobs=-1,
    objective="multiclass",
    class_weight="balanced" if config.USE_CLASS_WEIGHTS else None,
)

start = time.perf_counter()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="multi_logloss",
)
training_time = time.perf_counter() - start

y_pred = model.predict(X_test)

model_path = os.path.join(config.MODELS_DIR, "lightgbm_full.joblib")
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
    "model": "LightGBM",
    "feature_set": "Full",
    "features_used": data["feature_names"],
    "metrics": metrics,
    "efficiency": efficiency,
}

print("\nClassification report:\n")
print(metrics["classification_report"])
print("\nConfusion matrix:")
print(metrics["confusion_matrix"])

save_result(result, config.RESULTS_DIR, "LightGBM", "Full")
