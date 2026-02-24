# 🏦 End-to-End Credit Risk Prediction System (MLOps Pipeline)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

## 📖 Project Overview
This project implements a complete **Machine Learning Pipeline** to predict loan defaults (Credit Risk). It covers the entire lifecycle from raw data processing to model deployment via a REST API.

The goal is to help financial institutions minimize losses by accurately identifying high-risk loan applicants (Default: 1) versus safe applicants (Paid: 0).

---

## Architecture & Workflow

This project follows a strict MLOps workflow to ensure reproducibility and prevent data leakage.

### 1. Data Engineering (DataOps)
* **Data Cleaning:** Handled missing values and removed outliers (e.g., age > 100).
* **Feature Engineering:**
    * **Ordinal Encoding:** Applied to `loan_grade` (A -> G mapping).
    * **One-Hot Encoding:** Applied to nominal categories (`home_ownership`, `loan_intent`).
* **Scaling:** Applied `StandardScaler` to normalize numerical features (Income, Loan Amount, etc.).

### 2. Model Engineering (ML)
* **Pipeline Architecture:** Used `sklearn.pipeline.Pipeline` to encapsulate preprocessing and modeling. This prevents **Data Leakage** during Cross-Validation.
* **Validation Strategy:** Implemented **Stratified K-Fold Cross-Validation (K=5)** to maintain class balance in training/validation splits.
* **Model Selection:**
    * Compared **Logistic Regression**, **Decision Tree**, and **Random Forest**.
    * **Winner:** `Random Forest Classifier` with `class_weight='balanced'`.

### 3. Model Serving (Ops)
* **Serialization:** The final model is serialized using `joblib` for persistence.
* **API:** Wrapped the model in a **FastAPI** application for real-time inference.

---

## Model Performance

After rigorous testing with Stratified K-Fold and an unseen test set (20%), the **Random Forest** model achieved the following results:

| Metric | Score | Business Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **~93.4%** | Excellent capability to distinguish between defaulters and payers. |
| **Precision (Class 1)** | **98%** | When the model flags a customer as "Risk", it is correct 98% of the time. (Low False Alarms). |
| **Recall (Class 1)** | **~71%** | The model successfully catches 71% of all actual defaulters. |
| **Accuracy** | **~93%** | Overall correct prediction rate. |

> **Note:** The high precision ensures we don't reject good customers unnecessarily, while the solid recall helps minimize financial risk.

---

## Project Structure

```text
01-future-mlops-credit-risk-prediction-project/
│
├── data/
│   ├── credit_risk_dataset.csv    # Raw data
│   └── credit_risk_ready.csv      # Processed data (Encoded)
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb  # Exploratory Data Analysis
│   └── 02_model_training.ipynb    # Pipeline, CV, and Final Training
│
├── models/
│   └── credit_risk_model.joblib   # The serialized Random Forest model
│
├── app.py                         # FastAPI application for serving
├── confusion_matrix.png
|── eda_my_analysis_dolu.png            
└── README.md                      # Project documentation


# Clone the repository (if you haven't)
git clone [https://github.com/YOUR_USERNAME/mlops-dataops-devops-portfolio.git](https://github.com/YOUR_USERNAME/mlops-dataops-devops-portfolio.git)

# Navigate to the project folder
cd mlops-dataops-devops-portfolio/01-future-mlops-credit-risk-prediction-project

# Install dependencies
pip install -r requirements.txt

# You can run the notebook or convert it to a script
jupyter notebook notebooks/02_model_training.ipynb

uvicorn app:app --reload

4. Make Predictions
Open your browser and go to the Swagger UI to test the API interactively:
https://www.google.com/search?q=http://127.0.0.1:8000/docs

Or send a POST request via terminal:
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'Content-Type: application/json' \
  -d '{
  "person_age": 25,
  "person_income": 60000,
  "person_emp_length": 2,
  "loan_amnt": 10000,
  "loan_int_rate": 12.5,
  "loan_percent_income": 0.16,
  "cb_person_default_on_file": 0,
  "cb_person_cred_hist_length": 4,
  "person_home_ownership_OTHER": 0,
  "person_home_ownership_OWN": 0,
  "person_home_ownership_RENT": 1,
  "loan_intent_EDUCATION": 1,
  "loan_intent_HOMEIMPROVEMENT": 0,
  "loan_intent_MEDICAL": 0,
  "loan_intent_PERSONAL": 0,
  "loan_intent_VENTURE": 0
}'

Future Improvements
Dockerization: Containerize the API for consistent deployment.

MLflow Integration: Add experiment tracking to log metrics and parameters automatically.

CI/CD Pipeline: Automate testing and deployment using GitHub Actions.

