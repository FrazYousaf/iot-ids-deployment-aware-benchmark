# Deployment-Aware Benchmarking of ML and DL Models for IoT/IIoT Intrusion Detection

This repository contains the implementation notebooks for the paper:

**Beyond Accuracy: Deployment-Aware Benchmarking of Machine Learning and Deep Learning Models for IoT/IIoT Intrusion Detection**

## Overview

The study evaluates representative machine-learning, tree-based ensemble, and deep-learning models for IoT/IIoT intrusion detection under full-feature and reduced-feature settings.

The main objective is not only to report high accuracy, but to compare models using deployment-aware metrics such as false positives, training time, inference time, latency, throughput, model size, and reduced-feature behavior.

## Notebooks

| Notebook | Dataset / Experiment | Classes |
|---|---|---|
| `01_CIC_IoT_2023.ipynb` | CIC-IoT 2023 | BENIGN, DDOS, DOS, MIRAI |
| `02_CIC_IoT_IDAD_2024.ipynb` | CIC-IoT-IDAD 2024 | BENIGN, DDOS, DOS, MIRAI |
| `03_TabularIoTAttack_2024.ipynb` | TabularIoTAttack-2024 | BENIGN, DDOS, DOS, MQTT DOS |
| `04_DataSense_CIC_IIoT_2025.ipynb` | DataSense CIC-IIoT 2025 | BENIGN, DDOS, DOS, MIRAI |
| `05_Merged_CIC_IDAD_TabularIoTAttack_2024.ipynb` | Merged CIC-IoT-IDAD 2024 + TabularIoTAttack-2024 | BENIGN, DDOS, DOS, MIRAI, MQTT DOS |

## Models Evaluated

The following models are evaluated in the notebooks:

1. Decision Tree
2. Random Forest
3. XGBoost
4. LightGBM
5. MLP/DNN
6. CNN-LSTM-Attention

Each model is evaluated using both full-feature and reduced-feature settings.

## Evaluation Metrics

The notebooks report:

- Accuracy
- Balanced accuracy
- Macro-F1
- Weighted-F1
- False positives per 10,000 benign samples
- Training time
- Inference time
- Mean latency
- P95/P99 latency where available
- Throughput
- Model size
- Confusion matrices and classification reports where available

## Dataset Availability

The raw datasets are not included in this repository because of file size and dataset usage restrictions. Please download the datasets from their official sources and update the dataset path in each notebook before running.

## Output Files

The notebooks generate result tables and figures when executed. Generated outputs can be exported from the notebook output cells and later placed in:

```text
results/
figures/