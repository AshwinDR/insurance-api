import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load dataset
df = pd.read_csv("insurance.csv")


# Encode categorical values
df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
df.replace({'region': {
    'southeast': 0,
    'southwest': 1,
    'northeast': 2,
    'northwest': 3
}}, inplace=True)


# Split X and Y
X = df.drop("charges", axis=1)
Y = df["charges"]


# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)


# Train model
model = LinearRegression()
model.fit(X_train, Y_train)


# Save model
with open("insurance_model.pkl", "wb") as f:
    pickle.dump(model, f)


print("✅ Model trained and saved!")