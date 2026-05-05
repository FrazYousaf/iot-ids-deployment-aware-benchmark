import os
import json
import time
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def save_json(obj, path):
    def converter(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        return str(x)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=converter)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def model_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def false_positives_per_10000_benign(y_true, y_pred, class_names, benign_label_name="Benign"):
    class_names = list(class_names)
    benign_idx = class_names.index(benign_label_name)

    benign_mask = y_true == benign_idx
    total_benign = int(np.sum(benign_mask))

    false_positives = int(np.sum((y_true == benign_idx) & (y_pred != benign_idx)))
    fp_per_10000 = (false_positives / total_benign) * 10000 if total_benign > 0 else 0.0

    return false_positives, total_benign, fp_per_10000

def compute_metrics(y_true, y_pred, class_names, benign_label_name="Benign"):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    fp, total_benign, fp10k = false_positives_per_10000_benign(
        y_true,
        y_pred,
        class_names,
        benign_label_name=benign_label_name,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "false_positives": fp,
        "total_benign_samples": total_benign,
        "false_positives_per_10000_benign": float(fp10k),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            digits=4,
        ),
    }

def benchmark_sklearn_model(model, X, sample_size=50000, batch_size=512):
    n = min(len(X), sample_size)
    X_sample = X[:n]

    # Warm-up
    _ = model.predict(X_sample[:min(batch_size, n)])

    batch_latencies = []
    all_preds = []

    start_total = time.perf_counter()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X_sample[start:end]

        t1 = time.perf_counter()
        y_batch = model.predict(X_batch)
        t2 = time.perf_counter()

        all_preds.append(y_batch)
        batch_latencies.append(((t2 - t1) / len(X_batch)) * 1000)

    end_total = time.perf_counter()

    total_time = end_total - start_total
    throughput = n / total_time
    latencies = np.array(batch_latencies)

    return {
        "benchmark_samples": int(n),
        "total_inference_time_sec": float(total_time),
        "mean_latency_ms_per_sample": float(np.mean(latencies)),
        "p95_latency_ms_per_sample": float(np.percentile(latencies, 95)),
        "p99_latency_ms_per_sample": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": float(throughput),
    }

def benchmark_keras_model(model, X, sample_size=50000, batch_size=512, reshape_for_sequence=False):
    n = min(len(X), sample_size)
    X_sample = X[:n]

    if reshape_for_sequence:
        X_sample = X_sample.reshape((X_sample.shape[0], X_sample.shape[1], 1))

    # Warm-up
    _ = model.predict(X_sample[:min(batch_size, n)], batch_size=batch_size, verbose=0)

    batch_latencies = []

    start_total = time.perf_counter()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X_sample[start:end]

        t1 = time.perf_counter()
        _ = model.predict(X_batch, batch_size=batch_size, verbose=0)
        t2 = time.perf_counter()

        batch_latencies.append(((t2 - t1) / len(X_batch)) * 1000)

    end_total = time.perf_counter()

    total_time = end_total - start_total
    throughput = n / total_time
    latencies = np.array(batch_latencies)

    return {
        "benchmark_samples": int(n),
        "total_inference_time_sec": float(total_time),
        "mean_latency_ms_per_sample": float(np.mean(latencies)),
        "p95_latency_ms_per_sample": float(np.percentile(latencies, 95)),
        "p99_latency_ms_per_sample": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": float(throughput),
    }

def save_result(result, results_dir, model_name, feature_set):
    os.makedirs(results_dir, exist_ok=True)

    result_path = os.path.join(results_dir, f"{model_name}_{feature_set}_result.json")
    save_json(result, result_path)

    flat = {
        "model": result["model"],
        "feature_set": result["feature_set"],
        "accuracy": result["metrics"]["accuracy"],
        "balanced_accuracy": result["metrics"]["balanced_accuracy"],
        "macro_f1": result["metrics"]["macro_f1"],
        "weighted_f1": result["metrics"]["weighted_f1"],
        "fp_per_10000_benign": result["metrics"]["false_positives_per_10000_benign"],
        "training_time_sec": result["efficiency"]["training_time_sec"],
        "total_inference_time_sec": result["efficiency"]["total_inference_time_sec"],
        "mean_latency_ms_per_sample": result["efficiency"]["mean_latency_ms_per_sample"],
        "p95_latency_ms_per_sample": result["efficiency"]["p95_latency_ms_per_sample"],
        "p99_latency_ms_per_sample": result["efficiency"]["p99_latency_ms_per_sample"],
        "throughput_samples_per_sec": result["efficiency"]["throughput_samples_per_sec"],
        "model_size_mb": result["efficiency"]["model_size_mb"],
    }

    csv_path = os.path.join(results_dir, f"{model_name}_{feature_set}_summary.csv")
    pd.DataFrame([flat]).to_csv(csv_path, index=False)

    print("\nSaved result JSON:", result_path)
    print("Saved summary CSV:", csv_path)
    print("\nSummary:")
    print(pd.DataFrame([flat]).T)

def load_prepared_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "X_test": data["X_test"],
        "y_train": data["y_train"],
        "y_val": data["y_val"],
        "y_test": data["y_test"],
        "class_names": data["class_names"].tolist(),
        "feature_names": data["feature_names"].tolist(),
    }
