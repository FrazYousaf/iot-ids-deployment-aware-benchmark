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

print("TensorFlow devices:", tf.config.list_physical_devices())

def build_cnn_lstm_attention(n_features, num_classes):
    inputs = layers.Input(shape=(n_features, 1))

    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(64, return_sequences=True)(x)

    attn = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_eval(feature_set_name, npz_name):
    data = load_prepared_npz(os.path.join(config.ARTIFACT_DIR, npz_name))
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    class_names = data["class_names"]
    num_classes = len(class_names)

    X_train_seq = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_seq = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test_seq = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_cnn_lstm_attention(X_train.shape[1], num_classes)

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
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=config.DL_EPOCHS,
        batch_size=config.DL_BATCH_SIZE,
        callbacks=cb,
        verbose=1,
    )
    training_time = time.perf_counter() - start

    y_prob = model.predict(X_test_seq, batch_size=config.DL_BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    model_path = os.path.join(config.MODELS_DIR, f"cnn_lstm_attention_{feature_set_name.lower()}.keras")
    model.save(model_path)

    metrics = compute_metrics(
        y_test, y_pred, class_names, benign_label_name=config.BENIGN_LABEL_NAME
    )

    efficiency = benchmark_keras_model(
        model, X_test,
        sample_size=config.BENCHMARK_SAMPLE_SIZE,
        batch_size=config.BENCHMARK_BATCH_SIZE,
        reshape_for_sequence=True,
    )
    efficiency["training_time_sec"] = float(training_time)
    efficiency["model_size_mb"] = float(model_file_size_mb(model_path))

    result = {
        "model": "CNN_LSTM_Attention",
        "feature_set": feature_set_name,
        "features_used": data["feature_names"],
        "metrics": metrics,
        "efficiency": efficiency,
        "parameters": int(model.count_params()),
    }

    print(f"\n=== CNN-LSTM-Attention {feature_set_name} ===")
    print(metrics["classification_report"])
    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])
    print("\nModel parameters:", model.count_params())

    save_result(result, config.RESULTS_DIR, "CNN_LSTM_Attention", feature_set_name)

train_eval("Full", "full_scaled.npz")
train_eval("Reduced", "reduced_scaled.npz")
