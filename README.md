# Explainable Anomaly Detection in Medical Time Series

This repository contains the code and thesis material for a bachelor thesis on
explainable anomaly detection in medical time series. The project focuses on
Transformer-based anomaly detection models and compares explanation methods such
as attention visualizations and SHAP-style feature attribution.

## Repository Structure

```text
bachelor-thesis/
│
├── explainability/
│   ├── attention.py
│   ├── shap_explainer.py
│   ├── anomaly_score_wrappers.py
│   └── evaluation.py
│
├── models/
│   ├── anoformer.py
│   └── tranad.py
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_train_anoformer.ipynb
│   ├── 03_train_tranad.ipynb
│   ├── 04_shap_anoformer.ipynb
│   ├── 05_shap_tranad.ipynb
│   └── 06_explainability_evaluation.ipynb
│
├── results/
│   ├── anoformer/
│   └── tranad/
│
├── utils/
│   ├── dataset.py
│   ├── anomaly_threshold.py
│   └── seed.py
│
├── requirements.txt
└── README.md
```
