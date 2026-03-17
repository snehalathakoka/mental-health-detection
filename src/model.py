
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

tfidf = TfidfVectorizer(max_features=5000)
model = LogisticRegression(max_iter=1000)

def train_model(df):
    X = tfidf.fit_transform(df["clean_text"])
    y = df["is_depression"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred, average="weighted"))
    print(classification_report(y_test, pred))
    return model, tfidf

def predict_risk(text, model, tfidf):
    vector = tfidf.transform([text])
    probability = model.predict_proba(vector)[0][1]
    if probability >= 0.85:
        risk, emoji = "Critical", "🔴"
        message = "Immediate support needed!"
        suggestion = "Please contact a mental health professional immediately"
    elif probability >= 0.70:
        risk, emoji = "High Risk", "🟠"
        message = "Strong depression indicators detected"
        suggestion = "Strongly recommend speaking to a counselor"
    elif probability >= 0.40:
        risk, emoji = "Moderate Risk", "🟡"
        message = "Signs of emotional distress detected"
        suggestion = "Consider talking to someone you trust"
    elif probability >= 0.20:
        risk, emoji = "Mild Concern", "🔵"
        message = "Minor stress indicators present"
        suggestion = "Practice self care and monitor your mood"
    else:
        risk, emoji = "Healthy", "🟢"
        message = "No significant signs of distress"
        suggestion = "Keep maintaining positive habits!"
    return {"probability": probability, "risk": risk, "emoji": emoji,
            "message": message, "suggestion": suggestion}

def save_model(model, tfidf):
    pickle.dump(model, open("/content/mental-health-detection/src/model.pkl", "wb"))
    pickle.dump(tfidf, open("/content/mental-health-detection/src/tfidf.pkl", "wb"))
    print("✅ Model saved!")

def load_model():
    model = pickle.load(open("/content/mental-health-detection/src/model.pkl", "rb"))
    tfidf = pickle.load(open("/content/mental-health-detection/src/tfidf.pkl", "rb"))
    return model, tfidf
