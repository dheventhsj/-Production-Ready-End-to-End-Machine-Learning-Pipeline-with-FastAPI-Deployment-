import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.model import get_models

def run_pipeline():
    df = pd.read_csv("data/data.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models()
    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} accuracy:", acc)

        if acc > best_score:
            best_score = acc
            best_model = model

    with open("model/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("Best model saved with accuracy:", best_score)
