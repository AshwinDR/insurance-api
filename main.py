import pickle
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="Insurance Prediction API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)



# Load trained model
with open("insurance_model.pkl", "rb") as f:
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

    return {
        "predicted_charges": float(prediction[0])
    }
