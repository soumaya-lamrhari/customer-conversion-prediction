# Customer Conversion Prediction

An end-to-end machine learning project that predicts which bank customers are most likely to subscribe to a term deposit and helps prioritize customers for marketing outreach.

## Live API Demo

The final XGBoost model is publicly deployed as a REST API using FastAPI and Render.

**[Try the live API](https://customer-conversion-prediction-y44j.onrender.com/docs)**

The `/predict` endpoint generates a customer conversion probability and applies the selected targeting threshold to determine whether the customer should be prioritized for marketing outreach.

> **Note:** The API is hosted on Render's free tier and may take a short time to start after a period of inactivity.

## Business Problem

Marketing campaigns can be costly when outreach is directed to customers with a low likelihood of conversion.

The goal of this project is to use customer and campaign information to identify customers with higher conversion potential, allowing marketing teams to focus their efforts on a smaller and more relevant group.

**Business question:**  
Which customers are most likely to subscribe to a term deposit, and how can the bank use these predictions to prioritize customers for marketing outreach?

## Dataset

This project uses the **UCI Bank Marketing dataset**, which contains information about direct marketing campaigns conducted by a Portuguese banking institution.

- 45,211 customer records
- 16 original input features
- Target: whether the customer subscribed to a term deposit
- Conversion rate: 11.7%

The target is imbalanced, so model evaluation focuses on metrics such as precision, recall, F1-score, ROC-AUC, and PR-AUC rather than accuracy alone.

The `duration` feature was excluded from modeling because it represents the duration of the marketing contact and would not be available before contacting a customer. Including it would therefore introduce data leakage for a pre-contact targeting model.

## Project Workflow

### 1. Data Exploration

- Examined the target distribution, missing values, numerical features, and categorical conversion patterns.
- Investigated customer characteristics and previous campaign history.
- Identified the class imbalance in the target variable.
- Evaluated potential sources of data leakage.

### 2. Data Preprocessing

- Removed the `duration` feature to prevent data leakage.
- Preserved meaningful missing information as separate categories.
- Converted the target variable to binary format.
- One-hot encoded categorical variables.
- Used a stratified 80/20 train-test split.
- Fit preprocessing transformations using the training data.

### 3. Modeling

- Built a Logistic Regression baseline.
- Trained an XGBoost classifier.
- Compared the models using classification and ranking metrics including precision, recall, F1-score, ROC-AUC, and PR-AUC.

### 4. Threshold Selection

- Evaluated different probability thresholds on a validation set.
- Compared the trade-off between the number of customers targeted and the proportion of actual converters captured.
- Selected a probability threshold of **0.20** based on customer coverage, converter capture, precision, recall, and F1-score.

### 5. Model Explainability

- Used SHAP to analyze global model behavior.
- Examined how individual features influence predictions.
- Used individual SHAP explanations to understand specific customer predictions.

### 6. API and Deployment

- Serialized the final preprocessing and XGBoost pipeline using Joblib.
- Saved the selected targeting threshold with the trained model.
- Built a FastAPI REST API for scoring new customers.
- Tested the API locally using FastAPI's interactive documentation.
- Deployed the API publicly on Render for interactive model testing.

## Model Performance

The Logistic Regression baseline and XGBoost model were first compared using the default probability threshold of 0.50.

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Accuracy | 0.893 | 0.895 |
| Precision | 0.663 | 0.665 |
| Recall | 0.179 | 0.216 |
| F1-score | 0.281 | 0.325 |
| ROC-AUC | 0.772 | 0.804 |
| PR-AUC | 0.411 | 0.465 |

At the default threshold, XGBoost achieved better recall, F1-score, ROC-AUC, and PR-AUC than the Logistic Regression baseline.

XGBoost was therefore selected as the final model.

## Targeting Strategy

A default classification threshold of 0.50 is not necessarily the most useful threshold for a marketing prioritization problem.

Different thresholds were evaluated on a validation set to understand the trade-off between targeting more customers and capturing more potential converters.

A threshold of **0.20** was selected.

On the held-out test set, the final targeting strategy achieved:

- **12.6%** of customers selected for outreach
- **51.4%** of actual converters captured
- Recall: **0.514**
- Precision: **0.479**
- F1-score: **0.496**
- PR-AUC: **0.465**

This shows how the model can be used as a prioritization tool: marketing efforts can be focused on a smaller group of customers containing a substantial share of actual converters.

These results represent model performance on held-out test data and should not be interpreted as measured campaign cost savings or causal business impact.

## Model Explainability

SHAP was used to understand both overall model behavior and individual predictions.

The analysis showed that previous campaign outcomes, contact information, housing loan status, account balance, and campaign activity were among the influential features used by the model.

Previous campaign success strongly increased predicted conversion likelihood, while some other features showed nonlinear relationships with model predictions.

Individual SHAP explanations were also used to understand why specific customers received high or low conversion scores.

These patterns describe relationships learned by the model and should not be interpreted as causal effects.

## REST API

The trained preprocessing and XGBoost pipeline, together with the selected **0.20 targeting threshold**, is serialized with Joblib and served through FastAPI.

The API exposes the following endpoint:

`POST /predict`

### Example Request

```json
{
  "age": 35,
  "job": "management",
  "marital": "married",
  "education": "tertiary",
  "default": "no",
  "balance": 2500,
  "housing": "no",
  "loan": "no",
  "contact": "cellular",
  "day_of_month": 15,
  "month": "may",
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "no_previous_contact"
}
```

### Example Response

```json
{
  "conversion_probability": 0.632,
  "target_customer": true
}
```

In this example, the model estimates a **63.2% conversion probability**. Since this is above the selected **0.20 targeting threshold**, the customer is flagged for prioritized marketing outreach.

The deployed endpoint can be tested interactively through the FastAPI documentation:

**[Open the live API](https://customer-conversion-prediction-y44j.onrender.com/docs)**

## Repository Structure

```text
customer-conversion-prediction/
├── api/
│   └── main.py
├── model/
│   └── conversion_model.joblib
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_data_preprocessing.ipynb
│   ├── 3_modeling.ipynb
│   └── 4_model_explainability.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- Git & GitHub
- Render

## Run the API Locally

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI application:

```bash
uvicorn api.main:app --reload
```

Then open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Key Takeaway

This project demonstrates an end-to-end machine learning workflow, from exploratory data analysis and leakage prevention to model evaluation, threshold selection, explainability, API development, and public deployment.

The final model prioritizes **12.6% of customers while capturing 51.4% of actual converters** on held-out test data, demonstrating how predictive modeling can support more focused marketing outreach.
