from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("../models/bodyfat_model.pkl")


class BodyFatInput(BaseModel):
    Age: int
    Weight: float
    Height: float
    Neck: float
    Chest: float
    Abdomen: float
    Hip: float
    Thigh: float
    Knee: float
    Ankle: float
    Biceps: float
    Forearm: float
    Wrist: float


@app.get("/")
def root():
    return {"message": "Body fat estimator API is running"}


@app.post("/predict")
def predict(data: BodyFatInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_bodyfat": round(float(prediction), 2)}