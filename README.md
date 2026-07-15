# Explainable Anomaly Detection in Medical Time Series

This repository contains the code and thesis material for a bachelor thesis on
explainable anomaly detection in medical time series. The project focuses on
Transformer-based anomaly detection models and compares explanation methods,
specifically attention visualizations and SHAP feature attribution.

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

## Execution Environment

The notebooks were developed and executed in Google Colab. At the beginning of
each notebook, setup cells are included to mount Google Drive, configure dataset
and result paths, and make the local project modules available.

If the notebooks are executed in another environment, these paths may need to be
adapted manually.

## Execution Order

The notebooks should be executed in a specific order because the explanation
and evaluation samples are selected based on the anomaly detection results of
both models.

Recommended execution order:

1. Run `01_preprocessing.ipynb` if not using the already preprocessed data
2. Run `02_train_anoformer.ipynb` until the AnoFormer anomaly detection results
   are saved
3. Run `03_train_tranad.ipynb` until the TranAD anomaly detection results are
   saved
4. Continue `02_train_anoformer.ipynb` because it performs the shared sample selection. The selected samples are based on the saved outputs of both AnoFormer and TranAD
5. Run the SHAP notebooks:
   - `04_shap_anoformer.ipynb`
   - `05_shap_tranad.ipynb`
6. Run the explanation evaluation notebooks:
   - `06_faithfulness_evaluation.ipynb`
   - `07_stability_evaluation.ipynb`
   - `08_rec_error_agreement_evaluation.ipynb`

The shared visualization and evaluation samples require both models anomaly
detection outputs. Therefore, both AnoFormer and TranAD must be trained and their
test outputs must be saved before the explanation and evaluation notebooks are
executed.
