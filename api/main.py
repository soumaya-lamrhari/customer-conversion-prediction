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
