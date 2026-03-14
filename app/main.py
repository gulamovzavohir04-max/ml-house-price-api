from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="House Price Prediction API",
    description="Machine Learning API for predicting California house prices",
    version="1.0.0"
)

# путь к модели
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

# загружаем модель
model = joblib.load(MODEL_PATH)


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict_price(features: HouseFeatures):
    data = np.array([[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude
    ]])

    prediction = model.predict(data)[0]

    return {
        "predicted_price": float(prediction)
    }