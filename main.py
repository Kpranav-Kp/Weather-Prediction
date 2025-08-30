from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("weather_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI(title="Weather Prediction API", version="1.0")

@app.get("/")
def home():
    return {"message": "Weather Prediction API is running!"}
