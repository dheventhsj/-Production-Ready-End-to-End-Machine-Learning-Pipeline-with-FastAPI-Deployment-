import pickle
from src.data_ingestion import load_data
from src.preprocessing import preprocess
from src.train import train_model
from src.evaluate import evaluate

df = load_data("data/data.csv")

X_train, X_test, y_train, y_test = preprocess(df)

model = train_model(X_train, y_train)

mse = evaluate(model, X_test, y_test)
print("MSE:", mse)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
