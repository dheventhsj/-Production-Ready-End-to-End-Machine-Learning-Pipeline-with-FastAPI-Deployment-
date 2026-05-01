from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("model/model.pkl", "rb"))

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(age: int, income: int, experience: int):
    data = np.array([[age, income, experience]])
    pred = model.predict(data)[0]
    return {"prediction": int(pred)}
