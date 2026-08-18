# Beyond Accuracy: Deployment-Aware Benchmarking of ML and DL Models for IoT/IIoT Intrusion Detection

This repository contains the implementation and experiment notebooks for the paper:

**“Beyond Accuracy: A Deployment-Aware Multi-Dataset Evaluation of Machine Learning and Deep Learning Models for IoT/IIoT Intrusion Detection.”**

The study evaluates machine-learning and deep-learning intrusion detection models across multiple recent IoT/IIoT datasets, with particular emphasis on **deployment-aware model selection rather than accuracy alone**.

## Overview

Machine-learning-based intrusion detection systems are often evaluated primarily using overall accuracy. However, high accuracy alone may hide minority-class errors, benign false alarms, computational overhead, or model sizes that may be unsuitable for practical IoT/IIoT deployment.

This repository reproduces the experimental evaluation presented in the paper using:

* four individual IoT/IIoT datasets; and
* one schema-aligned pooled multi-source setting constructed from CIC-IoT-DIAD 2024 and TabularIoTAttack-2024.

Each experiment evaluates the same six model families under **full-feature** and **reduced-feature** settings.

The main evaluation criteria include:

* class-balanced detection performance;
* benign false-alarm burden;
* feature reduction;
* training cost;
* batch inference performance;
* throughput; and
* serialized model size.

The objective is to examine the trade-off between **detection quality and deployment-related cost**.

---

## Experimental Settings

The repository contains the following notebooks:

| Notebook                                                   | Dataset / Experiment                       | Classes                            |
| ---------------------------------------------------------- | ------------------------------------------ | ---------------------------------- |
| `notebooks/01-cic-iot-2023.ipynb`                          | CICIoT2023                                 | BENIGN, DDOS, DOS, MIRAI           |
| `notebooks/02-cic-iot-idad-2024.ipynb`                     | CIC-IoT-DIAD 2024                          | BENIGN, DDOS, DOS, MIRAI           |
| `notebooks/03-tabulariotattack-2024.ipynb`                 | TabularIoTAttack-2024                      | BENIGN, DDOS, DOS, MQTT DOS        |
| `notebooks/04-datasense-cic-iiot-2025.ipynb`               | DataSense CIC-IIoT 2025                    | BENIGN, DDOS, DOS, MIRAI           |
| `notebooks/05-merged-cic-idad-tabulariotattack-2024.ipynb` | Schema-aligned pooled DIAD–Tabular setting | BENIGN, DDOS, DOS, MIRAI, MQTT DOS |

### Pooled DIAD–Tabular Setting

The fifth experiment is a **schema-aligned pooled multi-source evaluation** constructed from:

* CIC-IoT-DIAD 2024; and
* TabularIoTAttack-2024.

Only shared features are retained before concatenation, and exact duplicate rows are removed.

The resulting pooled dataset contains five classes:

* BENIGN
* DDOS
* DOS
* MIRAI
* MQTT DOS

The pooled experiment should be interpreted as an **in-distribution multi-source evaluation**. Both source datasets are represented in the training and test partitions. It is therefore **not a source-held-out or cross-dataset generalization experiment**.

---

## Models Evaluated

The following six classifiers are evaluated:

1. Decision Tree
2. Random Forest
3. XGBoost
4. LightGBM
5. MLP/DNN
6. CNN-LSTM-Attention

Each model is evaluated using both:

* **Full-feature configuration**
* **Reduced-feature configuration**

The reduced-feature experiments are included to investigate whether comparable detection performance can be maintained while reducing the number of monitored features and their associated collection, preprocessing, storage, and maintenance requirements.

---

## Feature Reduction

Feature selection is performed using the **training partition only**.

The dataset-specific procedures used in the paper are:

| Dataset / Experiment    | Feature Selection Procedure                      | Reduced Features |
| ----------------------- | ------------------------------------------------ | ---------------: |
| CICIoT2023              | XGBoost feature importance                       |               15 |
| CIC-IoT-DIAD 2024       | XGBoost feature importance                       |               35 |
| TabularIoTAttack-2024   | Random Forest importance + correlation filtering |               35 |
| DataSense CIC-IIoT 2025 | Random Forest importance + correlation filtering |               35 |
| Pooled DIAD–Tabular     | Random Forest importance + correlation filtering |               35 |

For CICIoT2023 and CIC-IoT-DIAD 2024, the use of XGBoost importance to construct the reduced feature subsets should be considered when interpreting reduced-feature model rankings.

---

## Evaluation Metrics

The experiments report detection and deployment-related metrics including:

### Detection Performance

* Accuracy
* Balanced accuracy
* Macro-F1
* Weighted-F1
* Per-class precision and recall
* Confusion matrices
* Classification reports

### Benign False-Alarm Burden

The experiments report:

**False positives per 10,000 benign samples (FP/10k)**

This metric measures the number of benign samples incorrectly classified as attacks per 10,000 benign observations.

### Computational and Deployment Metrics

* Training time
* Batch inference time
* Mean per-sample inference time derived from batched prediction
* Throughput
* Serialized model size

Timing measurements should be interpreted as comparative measurements from the experimental environment. They are **not direct measurements of single-flow online latency or deployment latency on resource-constrained edge hardware**.

---

## Dataset Splits

The experiments use the following stratified data splits:

| Dataset / Experiment    | Split                                       |
| ----------------------- | ------------------------------------------- |
| CICIoT2023              | 70% training / 15% validation / 15% testing |
| CIC-IoT-DIAD 2024       | 70% training / 15% validation / 15% testing |
| TabularIoTAttack-2024   | 80% training / 20% testing                  |
| DataSense CIC-IIoT 2025 | 80% training / 20% testing                  |
| Pooled DIAD–Tabular     | 80% training / 20% testing after pooling    |

Feature selection and preprocessing operations that require fitted parameters are performed using the training partition only.

---

## Dataset Availability

The raw datasets are **not redistributed in this repository** because of their size and dataset distribution conditions.

Please obtain the datasets from their official sources:

* **CICIoT2023**
  Canadian Institute for Cybersecurity, University of New Brunswick
  https://www.unb.ca/cic/datasets/iotdataset-2023.html

* **CIC-IoT-DIAD 2024**
  Canadian Institute for Cybersecurity, University of New Brunswick
  https://www.unb.ca/cic/datasets/iot-diad-2024.html

* **TabularIoTAttack-2024**
  Canadian Institute for Cybersecurity, University of New Brunswick
  https://www.unb.ca/cic/datasets/tabular-iot-attack-2024.html

* **DataSense CIC-IIoT 2025**
  Canadian Institute for Cybersecurity, University of New Brunswick
  https://www.unb.ca/cic/datasets/iiot-dataset-2025.html

After downloading the datasets, update the corresponding dataset path in each notebook before execution.

---

## Repository Structure

```text
iot-ids-deployment-aware-benchmark/
│
├── notebooks/
│   ├── 01-cic-iot-2023.ipynb
│   ├── 02-cic-iot-idad-2024.ipynb
│   ├── 03-tabulariotattack-2024.ipynb
│   ├── 04-datasense-cic-iiot-2025.ipynb
│   └── 05-merged-cic-idad-tabulariotattack-2024.ipynb
│
├── results/
│
├── figures/
│
├── requirements.txt
│
└── README.md
```

The `results/` and `figures/` directories may be used to store generated experiment tables, metrics, confusion matrices, and figures.

---

## Reproducing the Experiments

### 1. Clone the repository

```bash
git clone https://github.com/FrazYousaf/iot-ids-deployment-aware-benchmark.git
cd iot-ids-deployment-aware-benchmark
```

### 2. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the datasets

Download the required datasets from the official sources listed above.

### 4. Configure dataset paths

Update the dataset path in the corresponding notebook according to your local or Kaggle environment.

### 5. Run the notebooks

The notebooks can be executed independently for the four individual datasets.

For the pooled DIAD–Tabular experiment, both CIC-IoT-DIAD 2024 and TabularIoTAttack-2024 must be available.

---

## Experimental Environment

The experiments reported in the paper were executed in a **Kaggle computational environment**.

Deep-learning models used GPU acceleration where available, while classical machine-learning models used their standard CPU implementations.

Consequently, training time, inference time, and throughput should be interpreted as comparative measurements from the stated experimental environment rather than direct measurements on constrained IoT/IIoT edge hardware.

---

## Relation to the Paper

The notebooks correspond to the experimental settings reported in the paper:

| Notebook                                         | Paper Experiment                           |
| ------------------------------------------------ | ------------------------------------------ |
| `01-cic-iot-2023.ipynb`                          | CICIoT2023 results                         |
| `02-cic-iot-idad-2024.ipynb`                     | CIC-IoT-DIAD 2024 results                  |
| `03-tabulariotattack-2024.ipynb`                 | TabularIoTAttack-2024 results              |
| `04-datasense-cic-iiot-2025.ipynb`               | DataSense CIC-IIoT 2025 results            |
| `05-merged-cic-idad-tabulariotattack-2024.ipynb` | Schema-aligned pooled DIAD–Tabular results |

The repository is intended to provide the implementation details, preprocessing procedures, feature-selection procedures, model configurations, and experimental code associated with the reported results.

---

## Reproducibility Notes

When reproducing the experiments, please note that:

* dataset paths may differ between local and Kaggle environments;
* raw datasets are not included in this repository;
* feature selection is performed using training data only;
* the pooled experiment contains samples from both source datasets in both training and testing partitions;
* the pooled experiment is therefore not intended as a source-transfer benchmark;
* computational measurements can vary depending on hardware, software versions, available CPU/GPU resources, and system load;
* the reported timing values should not be interpreted as direct edge-device latency measurements.

---

## Citation

If you use this repository or its experimental implementation in your research, please cite the associated paper:

> F. Yousaf, L. Veltri, T. Bakhshi, G. Penzotti, and F. Zanichelli,
> **“Beyond Accuracy: A Deployment-Aware Multi-Dataset Evaluation of Machine Learning and Deep Learning Models for IoT/IIoT Intrusion Detection,”**
> 2026.

Final publication information and DOI will be added after publication.

---

## Authors

* Fraz Yousaf — University of Parma
* Luca Veltri — University of Parma
* Taimur Bakhshi — Birkbeck, University of London
* Gabriele Penzotti — University of Parma
* Francesco Zanichelli — University of Parma

---

## Repository

https://github.com/FrazYousaf/iot-ids-deployment-aware-benchmark
