import os
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import config
from utils import (
    ensure_dirs, set_all_seeds, load_prepared_npz, compute_metrics,
    benchmark_keras_model, model_file_size_mb, save_result
)

set_all_seeds(config.RANDOM_SEED)
ensure_dirs(config.RESULTS_DIR, config.MODELS_DIR)

data = load_prepared_npz(os.path.join(config.ARTIFACT_DIR, "full_scaled.npz"))
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
class_names = data["class_names"]
num_classes = len(class_names)

print("TensorFlow devices:", tf.config.list_physical_devices())

def build_mlp(input_dim, num_classes):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_mlp(X_train.shape[1], num_classes)

cb = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.DL_PATIENCE,
        restore_best_weights=True,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    ),
]

start = time.perf_counter()
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=config.DL_EPOCHS,
    batch_size=config.DL_BATCH_SIZE,
    callbacks=cb,
    verbose=1,
)
training_time = time.perf_counter() - start

y_prob = model.predict(X_test, batch_size=config.DL_BATCH_SIZE, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

model_path = os.path.join(config.MODELS_DIR, "mlp_dnn_full.keras")
model.save(model_path)

metrics = compute_metrics(
    y_test, y_pred, class_names, benign_label_name=config.BENIGN_LABEL_NAME
)

efficiency = benchmark_keras_model(
    model, X_test,
    sample_size=config.BENCHMARK_SAMPLE_SIZE,
    batch_size=config.BENCHMARK_BATCH_SIZE,
    reshape_for_sequence=False,
)
efficiency["training_time_sec"] = float(training_time)
efficiency["model_size_mb"] = float(model_file_size_mb(model_path))

result = {
    "model": "MLP_DNN",
    "feature_set": "Full",
    "features_used": data["feature_names"],
    "metrics": metrics,
    "efficiency": efficiency,
    "parameters": int(model.count_params()),
}

print("\nClassification report:\n")
print(metrics["classification_report"])
print("\nConfusion matrix:")
print(metrics["confusion_matrix"])
print("\nModel parameters:", model.count_params())

save_result(result, config.RESULTS_DIR, "MLP_DNN", "Full")
