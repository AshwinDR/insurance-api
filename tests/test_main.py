from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


# Test home route
def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Insurance API is running"
    }


# Test prediction (valid input)
def test_predict_valid():

    payload = {
        "age": 30,
        "sex": 1,
        "bmi": 25.5,
        "children": 1,
        "smoker": 0,
        "region": 2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_charges" in data

    assert isinstance(data["predicted_charges"], float)


# Test prediction (invalid input)
def test_predict_invalid():

    payload = {
        "age": "wrong",
        "bmi": 25.5
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
