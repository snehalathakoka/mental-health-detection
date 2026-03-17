
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon", quiet=True)

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER"""
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    
    if compound <= -0.5:
        sentiment = "Very Negative"
        emoji = "😢"
    elif compound <= -0.2:
        sentiment = "Negative"
        emoji = "😔"
    elif compound <= 0.2:
        sentiment = "Neutral"
        emoji = "😐"
    elif compound <= 0.5:
        sentiment = "Positive"
        emoji = "🙂"
    else:
        sentiment = "Very Positive"
        emoji = "😊"
    
    return {
        "compound": compound,
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "sentiment": sentiment,
        "emoji": emoji
    }

def get_emotional_pattern(df):
    """Analyze emotional patterns across dataset"""
    df["sentiment_score"] = df["clean_text"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda x: "Negative" if x < -0.2 else ("Positive" if x > 0.2 else "Neutral")
    )
    return df
