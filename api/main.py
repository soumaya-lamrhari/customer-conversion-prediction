from fastapi import FastAPI
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
