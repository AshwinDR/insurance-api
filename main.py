from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import os


app = FastAPI(
    title="Insurance Prediction API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "insurance_model.pkl")


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


class InsuranceInput(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: int


@app.get("/")
def home():
    return {"message": "Insurance API is running"}


@app.post("/predict")
def predict(data: InsuranceInput):

    input_data = np.array([[
        data.age,
        data.sex,
        data.bmi,
        data.children,
        data.smoker,
        data.region
    ]])

    input_df = pd.DataFrame(
        input_data,
        columns=["age", "sex", "bmi", "children", "smoker", "region"]
    )

    prediction = model.predict(input_df)

    return {"predicted_charges": float(prediction[0])}
