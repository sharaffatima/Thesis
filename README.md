# Explainable Anomaly Detection in Medical Time Series

This repository contains the code and thesis material for a bachelor thesis on
explainable anomaly detection in medical time series. The project focuses on
Transformer-based anomaly detection models and compares explanation methods such
as attention visualizations and SHAP feature attribution.

## Repository Structure

```text
bachelor-thesis/
│
├── explainability/
│   ├── anoformer_attention.py
│   ├── attention_utils.py
│   ├── evaluation.py
│   ├── sample_selection.py
│   ├── shap_utils.py
│   ├── shap_wrappers.py
│   ├── tranad_attention.py
│   └── visualization.py
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
│   ├── 06_faithfulness_evaluation.ipynb
│   ├── 07_stability_evaluation.ipynb
│   └── 08_rec_error_agreement_evaluation.ipynb
│
│
├── utils/
│   ├── anomaly_threshold.py
│   ├── dataset.py
│   └── seed.py
│
├── README.md
└── requirements.txt
```

## How to run

## Important notes
