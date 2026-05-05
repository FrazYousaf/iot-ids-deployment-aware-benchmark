"""
Configuration for the IoT IDS benchmark.

Edit DATASET_PATH before running 00_prepare_features.py.
All other scripts load the prepared arrays from ARTIFACT_DIR.

Recommended execution order:
1) 00_prepare_features.py
2) 01_decision_tree_full.py
3) 02_random_forest_full.py
4) 03_xgboost_full_reduced.py
5) 04_lightgbm_full.py
6) 05_mlp_dnn_full.py
7) 06_cnn_lstm_attention_full_reduced.py
8) 07_make_final_table.py
"""

import os

# =========================
# User paths
# =========================

# Change this path to your Kaggle dataset path.
# Example:
# DATASET_PATH = "/kaggle/input/your-folder/final_iot_without_mitm.csv"
DATASET_PATH = "/kaggle/input/YOUR_FOLDER/final_iot_without_mitm.csv"

LABEL_COLUMN = "Label"

# Kaggle output directories
WORKING_DIR = "/kaggle/working"
ARTIFACT_DIR = os.path.join(WORKING_DIR, "artifacts")
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
MODELS_DIR = os.path.join(WORKING_DIR, "models")

# =========================
# Reproducibility
# =========================

RANDOM_SEED = 42

# Split: train 70%, validation 15%, test 15%
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# =========================
# Feature settings
# =========================

# Number of top features selected from XGBoost importance.
# You can set this to 10, 15, 20, etc.
TOP_K_FEATURES = 15

# If True, remove one feature from highly correlated pairs before feature selection.
# Leave False if you want "full features" to mean all available numeric features.
DROP_HIGH_CORR_FEATURES = False
CORR_THRESHOLD = 0.98

# =========================
# Training options
# =========================

USE_CLASS_WEIGHTS = False

# XGBoost GPU:
# If True, the script tries CUDA first and falls back to CPU if CUDA fails.
USE_GPU_FOR_XGBOOST = True

# Deep learning
DL_EPOCHS = 20
DL_BATCH_SIZE = 512
DL_PATIENCE = 5

# Inference benchmark settings
BENCHMARK_SAMPLE_SIZE = 50000
BENCHMARK_BATCH_SIZE = 512

# Benign class name for IDS false-positive metric
BENIGN_LABEL_NAME = "Benign"
