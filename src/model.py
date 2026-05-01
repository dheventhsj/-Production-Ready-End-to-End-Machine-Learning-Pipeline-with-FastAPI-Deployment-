from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_models():
    return {
        "logistic": LogisticRegression(),
        "rf": RandomForestClassifier(n_estimators=50)
    }
