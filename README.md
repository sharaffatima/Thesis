# Explainable Anomaly Detection in Medical Time Series

This repository contains the code and thesis material for a bachelor thesis on
explainable anomaly detection in medical time series. The project focuses on
Transformer-based anomaly detection models and compares explanation methods such
as attention visualizations and SHAP-style feature attribution.

## Repository Structure

```text
bachelor-thesis/

+-- explainability/
|   +-- attention.py
|   `-- shap.py
+-- models/
|   +-- anoformer.py
|   +-- classic.py
|   `-- tranad.py
+-- notebooks/
|   +-- 01_preprocessing.ipynb
|   +-- 02_train_anoformer.ipynb
|   `-- 03_train_tranad.ipynb
+-- results/
|   +-- anoformer/
|   +-- classic/
|   `-- tranad/
+-- utils/
|   +-- dataset.py
|   +-- seed.py

+-- requirements.txt
`-- README.md
```
