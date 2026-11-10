# Customer Conversion Prediction

An end-to-end machine learning project that predicts which bank customers are most likely to subscribe to a term deposit and helps prioritize customers for marketing outreach.

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

The `duration` feature was excluded from modeling because it represents the duration of the marketing contact and would not be available before contacting a customer.

## Project Workflow
1. **Data Exploration**
   - Examined target distribution, missing values, numerical features, and categorical conversion patterns.
   - Investigated previous campaign history and customer characteristics.

2. **Data Preprocessing**
   - Removed the `duration` feature to prevent data leakage.
   - Preserved meaningful missing information as separate categories.
   - One-hot encoded categorical variables.
   - Used a stratified 80/20 train-test split.

3. **Modeling**
   - Built a Logistic Regression baseline.
   - Trained an XGBoost classifier.
   - Compared models using classification and ranking metrics.

4. **Threshold Selection**
   - Evaluated different probability thresholds on a validation set.
   - Selected a threshold of **0.20** based on the trade-off between customer coverage, converter capture, precision, recall, and F1-score.

5. **Model Explainability**
   - Used SHAP to analyze global feature importance and individual predictions.

6. **API**
   - Saved the final model pipeline and targeting threshold.
   - Built a FastAPI REST API to generate conversion probabilities for new customers.

## Model Performance

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Accuracy | 0.893 | 0.895 |
| Precision | 0.663 | 0.665 |
| Recall | 0.179 | 0.216 |
| F1-score | 0.281 | 0.325 |
| ROC-AUC | 0.772 | 0.804 |
| PR-AUC | 0.411 | 0.465 |

At the default 0.50 threshold, XGBoost provided better recall, F1-score, ROC-AUC, and PR-AUC than the Logistic Regression baseline.

## Targeting Strategy

The final XGBoost model uses a probability threshold of **0.20**.

On the held-out test set:

- **12.6%** of customers were selected for outreach.
- **51.4%** of actual converters were captured.
- Final recall: **0.514**
- Final precision: **0.479**
- Final F1-score: **0.496**
- PR-AUC: **0.465**

This demonstrates how the model can be used as a prioritization tool rather than simply making predictions at the default 0.50 threshold.

## Model Explainability

SHAP was used to understand both overall model behavior and individual predictions.

The analysis showed that previous campaign outcomes, contact information, housing loan status, account balance, and campaign activity were among the influential features used by the model.

Previous campaign success strongly increased predicted conversion likelihood, while some other features showed nonlinear relationships with the predictions.

These patterns describe relationships learned by the model and should not be interpreted as causal effects.

## REST API

The trained model is exposed through a FastAPI endpoint:

`POST /predict`

The API accepts customer information and returns:

- the predicted conversion probability
- whether the customer should be prioritized based on the selected targeting threshold

Example response:

```json
{
  "conversion_probability": 0.632,
  "target_customer": true
}
