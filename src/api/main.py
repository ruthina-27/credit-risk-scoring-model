from fastapi import FastAPI
from src.api.pydantic_models import PredictRequest, PredictResponse
import joblib
import numpy as np
import os

app = FastAPI()

# Load the best model at startup
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.pkl')
model = joblib.load(MODEL_PATH)

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    # Convert request to model input
    input_data = np.array([
        [
            request.Recency,
            request.Frequency,
            request.Monetary,
            request.AvgTransactionAmount,
            request.StdTransactionAmount,
            request.MostCommonHour,
            request.MostCommonDay,
            request.MostCommonMonth,
            request.MostCommonYear
            # Add categorical features here if needed
        ]
    ])
    prob = model.predict_proba(input_data)[0, 1] if hasattr(model, 'predict_proba') else float(model.predict(input_data)[0])
    is_high_risk = int(prob > 0.5)
    return PredictResponse(risk_probability=prob, is_high_risk=is_high_risk)
