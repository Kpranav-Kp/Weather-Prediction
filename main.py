from fastapi import FastAPI
import joblib
import numpy as np
from models import fetch_data

model = joblib.load("weather_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI(title="Weather Prediction API", version="1.0")
lag_features = [
        "temperature_2m_mean (°C)", 
        "relative_humidity_2m_mean (%)", 
        "pressure_msl_mean (hPa)", 
        "precipitation_sum (mm)", 
        "cloud_cover_mean (%)"
    ]

# Home route
@app.get("/")
def home():
    return {"message": "Weather Prediction API is running!"}

# Fetch data
@app.get("/data")
async def data():
    datas = fetch_data(52.52, 13.41, 8)
    return {"data":datas.to_dict(orient="records")}

@app.get("/predict")
async def predict(lat: float = 52.52, lon: float = 13.41, days: int = 8):
    datas = fetch_data(lat, lon, days + 7)  # fetch extra days for lag7

    for col in lag_features:
        datas[f"{col}_lag1"] = datas[col].shift(1)
        datas[f"{col}_lag7"] = datas[col].shift(7)
    for col in lag_features:
        datas[f"{col}_rollmean3"] = datas[col].shift(1).rolling(3).mean()

    datas = datas.dropna().reset_index(drop=True)
    latest_row = datas.iloc[[-1]]

    predictors = [col for col in latest_row.columns if col != "date"]
    X = latest_row[predictors].to_numpy()
    y_pred = model.predict(X)
    y_pred_label = label_encoder.inverse_transform(y_pred)[0]

    return {
        "date": str(latest_row["date"].values[0]),
        "prediction": str(y_pred_label)
    }
