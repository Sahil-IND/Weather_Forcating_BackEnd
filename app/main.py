from fastapi import FastAPI, Query
import joblib
import requests
import pandas as pd
from datetime import datetime
import os
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correct path to model
model_path = os.path.join(os.path.dirname(__file__), "..", "rain_model.pkl")
model = joblib.load(model_path)

def fetch_weather(lat: float, lon: float):
    url = "https://archive-api.open-meteo.com/v1/archive"
    today = datetime.now().strftime("%Y-%m-%d")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": today,
        "end_date": today,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "surface_pressure_mean",
            "cloud_cover_mean",
            "wind_speed_10m_max",
            "precipitation_sum"
        ],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "daily" not in data:
        raise ValueError("Weather data not available")

    df = pd.DataFrame(data["daily"])
    return df

@app.get("/predict")
def predict_rain(latitude: float = Query(...), longitude: float = Query(...)):
    try:
        df = fetch_weather(latitude, longitude)

        features = df[[
            "precipitation_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "surface_pressure_mean",
            "cloud_cover_mean",
            "wind_speed_10m_max"
        ]].values

        prediction = model.predict(features)
        return {
            "latitude": latitude,
            "longitude": longitude,
            "will_rain_tomorrow": bool(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}
