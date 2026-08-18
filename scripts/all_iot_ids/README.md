**Note: This directory contains the base benchmarking scripts used during development. Dataset-specific configurations and the exact implementations corresponding to the paper results are provided in the notebooks under notebooks/.**


# IoT IDS Benchmark: Accuracy + Deployment-Aware Evaluation

This package contains Kaggle-ready Python scripts for fairly comparing ML and DL models on the same IoT IDS dataset.

## Models

- Decision Tree: full features
- Random Forest: full features
- XGBoost: full and reduced features
- LightGBM: full features
- MLP/DNN: full features
- CNN-LSTM-Attention: full and reduced features

## Metrics

Each model logs:

- Accuracy
- Balanced accuracy
- Macro-F1
- Weighted-F1
- Per-class precision/recall/F1
- Confusion matrix
- False positives per 10,000 benign samples
- Training time
- Inference time
- Mean latency per sample
- p95 latency per sample
- p99 latency per sample
- Throughput in samples/second
- Model size in MB

## Execution order

1. Edit `config.py` and set `DATASET_PATH`.
2. Run:

```bash
python 00_prepare_features.py
python 01_decision_tree_full.py
python 02_random_forest_full.py
python 03_xgboost_full_reduced.py
python 04_lightgbm_full.py
python 05_mlp_dnn_full.py
python 06_cnn_lstm_attention_full_reduced.py
python 07_make_final_table.py
```

In Kaggle notebooks, you can also use:

```python
%run 00_prepare_features.py
```

and similarly for the other files.

## Fair comparison notes

- All models use the same train/validation/test split.
- Feature selection is performed only on the training set.
- The same reduced features are used for all reduced-feature experiments.
- Imputation and scaling are fitted only on training data to avoid leakage.
- Tree models use imputed non-scaled features.
- Deep learning models use imputed and standardized features.
