import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(
    title="Customer Conversion Prediction API",
    description="API for predicting customer conversion probability",
    version="1.0"
)

# Load the trained model and selected targeting threshold
model_path = Path(__file__).resolve().parent.parent / "model" / "conversion_model.joblib"

model_artifact = joblib.load(model_path)

model = model_artifact["model"]
threshold = model_artifact["threshold"]

# Define the customer information expected by the API
class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day_of_month: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    
# Predict whether a customer should be prioritized for marketing outreach
@app.post("/predict")
def predict_conversion(customer: CustomerData):
    # Convert the customer input into the format expected by the model
    customer_df = pd.DataFrame([customer.model_dump()])

    # Generate the customer's conversion probability
    probability = model.predict_proba(customer_df)[:, 1][0]

    # Apply the targeting threshold selected during model evaluation
    target_customer = probability >= threshold

    return {
        "conversion_probability": round(float(probability), 3),
        "target_customer": bool(target_customer)
    }
