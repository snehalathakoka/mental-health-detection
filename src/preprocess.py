
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def load_data(filepath):
    """Load dataset from CSV file"""
    df = pd.read_csv(filepath)
    return df

def clean_text(text):
    """Clean and preprocess text"""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra spaces
    text = text.strip()
    # Remove stopwords and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def preprocess_dataframe(df):
    """Apply cleaning to entire dataframe"""
    df = df.dropna()
    df["clean_text"] = df["clean_text"].apply(clean_text)
    return df
