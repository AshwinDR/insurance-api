# Insurance Prediction API - Complete Setup & Execution Guide

This document contains all step-by-step instructions to:

1. Set up the project
2. Train the Machine Learning model
3. Run the FastAPI application
4. Test the API manually
5. Run unit tests

# Project Overview

This project builds a Machine Learning model using Linear Regression to predict insurance charges.

The trained model is exposed through a FastAPI REST API.

The project also includes automated unit tests using pytest.

# Project Structure

insurance/
│
├── main.py
├── train_model.py
├── insurance.csv
├── insurance_model.pkl (generated after training)
├── requirements.txt
│
└── tests/
├── init.py
└── test_main.py

# System Requirements

Before running this project, make sure you have:

- Python 3.10 or above
- pip package manager
- VS Code / Any Python IDE
- Internet connection (for installing packages)

# Step 1: Open Project in VS Code

1. Open VS Code
2. Click **File → Open Folder**
3. Select the "insurance" project folder
4. Open terminal using "Ctrl + `"

# Step 2: Create Virtual Environment
Run the following command in terminal:

python -m venv venv

To activate the environment:
For Windows
venv\Scripts\activate

For Linux / Mac
source venv/bin/activate

After activation, (venv) will appear in terminal.

# Step 3: Install Required Libraries

Install dependencies using:

pip install -r requirements.txt

# Step 4: Train Machine Learning Model

Train the model by running:

python train_model.py

After successful execution, this file will be created:

insurance_model.pkl

This file contains the trained ML model.

# Step 5: Run FastAPI Application

Start the API server using:

uvicorn main:app --reload

After running this command, open browser and go to:

http://127.0.0.1:8000/docs

This opens Swagger UI for API testing.

# Step 6: Test API Manually (Using Swagger)

Open /docs in browser

Click on POST "/predict"    

Click Try it out

Enter sample input:

{
  "age": 30,
  "sex": 1,
  "bmi": 25.5,
  "children": 1,
  "smoker": 0,
  "region": 2
}

Click Execute

View predicted insurance charges

# Step 7: Run Unit Tests
Run tests using:

python -m pytest

Expected output:

collected 3 items
test_main.py ... [100%]

This means all tests passed.

# Step 8: Development Workflow

Follow this order whenever starting the project:

1.Activate virtual environment

2.Install dependencies

3.Train model

4.Start API server

5.Test using Swagger

6.Run unit tests

Common Issues and Solutions
1.Module Not Found Error

Run:
pip install -r requirements.txt

2.Model File Not Found

Make sure you ran:
python train_model.py

3.API Not Starting

Check:
uvicorn main:app --reload
and verify file name is main.py.