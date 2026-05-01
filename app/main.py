from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("model/model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "ML Pipeline API Running"}

@app.post("/predict")
def predict(feature1: float, feature2: float):
    data = np.array([[feature1, feature2]])
    prediction = model.predict(data)[0]
    return {"prediction": float(prediction)}
