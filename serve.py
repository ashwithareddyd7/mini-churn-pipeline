from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import joblib
import pandas as pd
import sqlite3
from reverse_etl import log_prediction

app = FastAPI()

# Load model
model = joblib.load("model_store/churn_model.pkl")

# Load feature schema safely
with sqlite3.connect("model_store/feature_store.db") as conn:
    prototype = pd.read_sql("SELECT * FROM features LIMIT 1", conn)

feature_cols = list(prototype.drop(columns=["churn_flag"]).columns)

class CustomerFeatures(BaseModel):
    features: Dict[str, float]

@app.post("/predict")
def predict(data: CustomerFeatures):
    X = pd.DataFrame([data.features], columns=feature_cols)
    prob = float(model.predict_proba(X)[0][1])

    log_prediction(data.features, prob)

    return {"churn_probability": prob}

@app.get("/health")
def health():
    return {"status": "ok"}