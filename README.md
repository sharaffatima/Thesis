# Explainable Anomaly Detection in Medical Time Series

This repository contains the code and thesis material for a bachelor thesis on
explainable anomaly detection in medical time series. The project focuses on
Transformer-based anomaly detection models and compares explanation methods such
as attention visualizations and SHAP-style feature attribution.

## Repository Structure

```text
bachelor-thesis/
+-- dataset/
|   `-- ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
+-- explainability/
|   +-- attention.py
|   `-- shap.py
+-- models/
|   +-- anoformer.py
|   +-- classic.py
|   `-- tranad.py
+-- notebooks/
|   +-- preprocessing.ipynb
|   +-- train_anoformer.ipynb
|   +-- train_classic.ipynb
|   `-- train_tranad.ipynb
+-- results/
|   +-- anoformer/
|   +-- classic/
|   `-- tranad/
+-- utils/
|   +-- dataset.py
|   +-- preprocessing.py
|   +-- ptbxl_loader.py
|   +-- seed.py
|   `-- training.py
+-- requirements.txt
+-- tagebuch.md
`-- README.md
```
