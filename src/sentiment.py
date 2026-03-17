
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon", quiet=True)

analyzer = SentimentIntensityAnalyzer()

# Emotion keyword patterns
EMOTION_PATTERNS = {
    "Sadness": ["sad", "cry", "tears", "grief", "loss", "empty", "hopeless", "alone"],
    "Anger": ["angry", "hate", "frustrated", "rage", "furious", "annoyed"],
    "Fear": ["scared", "afraid", "anxious", "worry", "panic", "terror", "dread"],
    "Joy": ["happy", "excited", "grateful", "blessed", "wonderful", "amazing"],
    "Hopelessness": ["pointless", "worthless", "meaningless", "give up", "no point"],
    "Isolation": ["alone", "lonely", "nobody", "no one", "isolated", "invisible"]
}

def detect_dominant_emotion(text):
    """Detect dominant emotion from text"""
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in EMOTION_PATTERNS.items():
        score = sum(1 for word in keywords if word in text_lower)
        emotion_scores[emotion] = score
    
    dominant = max(emotion_scores, key=emotion_scores.get)
    if emotion_scores[dominant] == 0:
        dominant = "Neutral"
    
    return dominant, emotion_scores

def detect_behavioral_patterns(text):
    """Detect behavioral patterns in text"""
    patterns = []
    text_lower = text.lower()
    
    if any(w in text_lower for w in ["alone", "lonely", "isolated", "nobody"]):
        patterns.append("🚩 Social Isolation")
    if any(w in text_lower for w in ["worthless", "useless", "failure", "burden"]):
        patterns.append("🚩 Low Self Worth")
    if any(w in text_lower for w in ["hopeless", "pointless", "no future", "give up"]):
        patterns.append("🚩 Hopelessness")
    if any(w in text_lower for w in ["cant sleep", "insomnia", "awake", "no sleep"]):
        patterns.append("🚩 Sleep Disturbance")
    if any(w in text_lower for w in ["die", "death", "suicide", "end it"]):
        patterns.append("🚨 Critical Warning - Self Harm Indicators")
    
    if not patterns:
        patterns.append("✅ No concerning behavioral patterns detected")
    
    return patterns

def analyze_sentiment(text):
    """Analyze sentiment with full emotional insight"""
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    
    if compound <= -0.5:
        sentiment, emoji = "Very Negative", "😢"
    elif compound <= -0.2:
        sentiment, emoji = "Negative", "😔"
    elif compound <= 0.2:
        sentiment, emoji = "Neutral", "😐"
    elif compound <= 0.5:
        sentiment, emoji = "Positive", "🙂"
    else:
        sentiment, emoji = "Very Positive", "😊"
    
    dominant_emotion, emotion_scores = detect_dominant_emotion(text)
    behavioral_patterns = detect_behavioral_patterns(text)
    intensity = abs(compound)
    
    if intensity >= 0.7:
        emotional_intensity = "High"
    elif intensity >= 0.3:
        emotional_intensity = "Moderate"
    else:
        emotional_intensity = "Low"
    
    return {
        "compound": compound,
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "sentiment": sentiment,
        "emoji": emoji,
        "dominant_emotion": dominant_emotion,
        "emotion_scores": emotion_scores,
        "behavioral_patterns": behavioral_patterns,
        "emotional_intensity": emotional_intensity
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
