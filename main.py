from fastapi import FastAPI
from pydantic import BaseModel

import pickle
import numpy as np
import pandas as pd


# Create app
app = FastAPI(
    title="Insurance Prediction API",
    description="Predict insurance charges using ML",
    version="1.0"
)


# Load trained model
with open("insurance_model.pkl", "rb") as file:
    model = pickle.load(file)


# Input schema
class InsuranceInput(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: int


# Home route
@app.get("/")
def home():
    return {"message": "Insurance API is running"}


# Prediction route
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

    return {
        "predicted_charges": float(prediction[0])
    }
