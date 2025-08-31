import requests
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

lag_features = [
    "temperature_2m_mean (°C)", 
    "relative_humidity_2m_mean (%)", 
    "pressure_msl_mean (hPa)", 
    "precipitation_sum (mm)", 
    "cloud_cover_mean (%)"
]

# Fetch the past 8 days data
def fetch_data(lat: float, lon: float, days: int = 8) ->pd.DataFrame:
    url = url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude" : lat,
        "longitude" : lon,
        "daily" : [
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "pressure_msl_mean",
            "precipitation_sum",
            "cloud_cover_mean"
        ],
        "timezone" : "auto",
        "start_date" : (pd.Timestamp.today().normalize() - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
        "end_date" : (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    relative_humidity_2m_mean = daily.Variables(1).ValuesAsNumpy()
    pressure_msl_mean = daily.Variables(2).ValuesAsNumpy()
    precipitation_sum = daily.Variables(3).ValuesAsNumpy()
    cloud_cover_mean = daily.Variables(4).ValuesAsNumpy()

    dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        periods=daily.Variables(0).ValuesLength(),
        freq=pd.Timedelta(seconds=daily.Interval())
    )

    df = pd.DataFrame({
        "date": dates,
        "temperature_2m_mean (°C)": temperature_2m_mean,
        "relative_humidity_2m_mean (%)": relative_humidity_2m_mean,
        "pressure_msl_mean (hPa)": pressure_msl_mean,
        "precipitation_sum (mm)": precipitation_sum,
        "cloud_cover_mean (%)": cloud_cover_mean
    })

    return df.reset_index(drop=True)