from fastapi import FastAPI
import joblib
import numpy as np
from models import fetch_data

model = joblib.load("weather_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI(title="Weather Prediction API", version="1.0")

# Home route
@app.get("/")
def home():
    return {"message": "Weather Prediction API is running!"}

# Fetch data
@app.get("/data")
async def data():
    datas = fetch_data(52.52, 13.41, 8)
    return {"data":datas.to_dict(orient="records")}