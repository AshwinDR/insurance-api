import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load dataset
data = pd.read_csv("insurance.csv")


# Encode categorical columns
data.replace({
    "sex": {"male": 0, "female": 1},
    "smoker": {"yes": 0, "no": 1},
    "region": {
        "southeast": 0,
        "southwest": 1,
        "northeast": 2,
        "northwest": 3
    }
}, inplace=True)


# Split features and target
X = data.drop("charges", axis=1)
y = data["charges"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# Save model
with open("insurance_model.pkl", "wb") as file:
    pickle.dump(model, file)


print("✅ Model trained and saved successfully!")
