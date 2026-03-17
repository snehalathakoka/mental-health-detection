
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

# Initialize
tfidf = TfidfVectorizer(max_features=5000)
model = LogisticRegression(max_iter=1000)

def train_model(df):
    """Train the ML model"""
    X = tfidf.fit_transform(df["clean_text"])
    y = df["is_depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred, average="weighted"))
    print(classification_report(y_test, pred))

    return model, tfidf

def predict_risk(text, model, tfidf):
    """Predict depression risk for a given text"""
    vector = tfidf.transform([text])
    probability = model.predict_proba(vector)[0][1]

    if probability > 0.7:
        risk = "High Risk"
    elif probability > 0.4:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return probability, risk

def save_model(model, tfidf):
    """Save model and vectorizer"""
    pickle.dump(model, open("/content/mental-health-detection/src/model.pkl", "wb"))
    pickle.dump(tfidf, open("/content/mental-health-detection/src/tfidf.pkl", "wb"))
    print("✅ Model saved!")

def load_model():
    """Load saved model and vectorizer"""
    model = pickle.load(open("/content/mental-health-detection/src/model.pkl", "rb"))
    tfidf = pickle.load(open("/content/mental-health-detection/src/tfidf.pkl", "rb"))
    return model, tfidf
