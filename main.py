import numpy as np 
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

## Load the model and scaler
classifier = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

## Define the shape of the incoming data
class InputData(BaseModel):
    age: int
    estimated_salary: int

@app.get('/')
def read_root():
    return {"message": "Welcome to the Logistic Regression Prediction API"}

@app.post('/predict')
def predict(data: InputData):
    input_data = np.array([[data.age, data.estimated_salary]])
    input_data_scaled = scaler.transform(input_data)
    prediction = classifier.predict(input_data_scaled)
    return {'prediction': int(prediction[0])}