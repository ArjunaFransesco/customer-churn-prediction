# 🛡️ ChurnShield ML — Customer Churn Prediction & Retention Analytics

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Framework-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Dataset](https://img.shields.io/badge/Dataset-IBM_Telco_Churn-green.svg)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

An end-to-end Machine Learning pipeline and interactive web application designed to predict customer churn risk. Utilizing the **IBM Telco Customer Churn** dataset, this project implements robust preprocessing, handles class imbalance, compares multiple state-of-the-art estimators (Logistic Regression, Random Forest, XGBoost), performs hyperparameter tuning, and exposes a beautiful web-based interface for customer success team retention decisions.

---

## 📌 Project Overview

Customer churn occurs when customers stop doing business with a service provider. Retaining customers is highly cost-effective compared to acquiring new ones.

This project implements:
1. **Automated Data Ingestion**: Clean download pipeline from raw sources.
2. **Robust Feature Pipelines**: Handle missing values, scaling, and categorical encoding wrapped inside a `ColumnTransformer` to eliminate **data leakage**.
3. **Imbalance-Aware Training**: Class weight adjustments and cost-sensitive scale weights for XGBoost.
4. **Interactive Dashboard**: Interactive input dashboard to evaluate single-user churn probability, risk tier, and recommend strategic retention campaigns.

---

## 🏗️ Model Architecture & Preprocessing

```text
Raw CSV Input (21 columns)
│
├── Clean TotalCharges (Handle missing spaces -> Median Impute)
├── Drop customerID (Non-predictive identifier)
│
├── Preprocessing Pipeline (ColumnTransformer):
│   ├── Numerical Features [tenure, MonthlyCharges, TotalCharges] ──> StandardScaler()
│   └── Categorical Features [Gender, Contract, InternetService...] ──> OneHotEncoder(drop='first')
│
├── Stratified Train/Test Split (80/20)
│
├── Model Search Grid (GridSearchCV on ROC-AUC):
│   ├── Logistic Regression (class_weight='balanced')
│   ├── Random Forest (class_weight='balanced')
│   └── XGBoost (scale_pos_weight = imbalance_ratio)
│
└── Serialization (models/customer_churn_pipeline.joblib)
```

---

## 📈 Model Performance Benchmark

The estimators were trained using 5-fold stratified cross-validation and optimized for **ROC-AUC** to balance false positives (cost of unnecessary incentives) against false negatives (cost of lost customers).

| Model | ROC-AUC Score | Primary Features Impact |
| :--- | :---: | :--- |
| **XGBoost (Tuned)** | **~0.846** | Contract (Month-to-month), InternetService (Fiber optic), Tenure |
| **Logistic Regression** | **~0.842** | Contract (Month-to-month), TotalCharges, InternetService (Fiber optic) |
| **Random Forest** | **~0.835** | Tenure, MonthlyCharges, TotalCharges |

---

## ⚙️ Project Structure

```text
customer-churn-prediction/
│
├── app/
│   ├── templates/
│   │   └── index.html             # Beautiful front-end user interface
│   └── main.py                    # Flask API & application server
│
├── data/
│   └── raw/                       # Downloaded raw CSV dataset (git-ignored)
│
├── models/
│   └── customer_churn_pipeline.joblib # Saved preprocessor & best model pipeline (git-ignored)
│
├── notebooks/
│   └── customer_churn_pipeline.ipynb  # Comprehensive analysis & EDA notebook
│
├── reports/
│   └── figures/                   # Model evaluation plots (ROC/PR curves)
│
├── src/
│   ├── data_loader.py             # Data ingestion and download utility
│   ├── predict.py                 # Out-of-sample inference pipeline helper
│   └── train.py                   # Model pipeline training & tuning script
│
├── .gitignore                     # Python, notebook, and data gitignore rules
├── requirements.txt               # List of environment dependencies
└── README.md                      # Detailed project documentation
```

---

## 🚀 Getting Started

### 1. Installation

Clone this repository and create a virtual environment:
```bash
git clone https://github.com/ArjunaFransesco/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
venv\Scripts\activate # On Windows
source venv/bin/activate # On Unix/macOS
pip install -r requirements.txt
```

### 2. Training the Model Pipeline

Run the training script to fetch the raw data, preprocess, build the model comparisons, perform hyperparameter grid-search, save the plots under `reports/` and export the final pipeline:
```bash
python src/train.py
```

### 3. Run the Web Dashboard Locally

Start the Flask application:
```bash
python app/main.py
```
Open your browser and navigate to:
```text
http://localhost:5000
```
Use the tabs to input customer metrics (Demographics, Services, and Billing) to run real-time predictions and view retention recommendations!


<!-- Last Maintenance Audit: 2026-09-02 -->
