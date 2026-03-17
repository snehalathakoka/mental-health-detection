
import streamlit as st
import sys
sys.path.append("/content/mental-health-detection")

from src.preprocess import load_data, preprocess_dataframe
from src.model import train_model, predict_risk, save_model, load_model
from src.sentiment import analyze_sentiment, get_emotional_pattern
from src.visualize import plot_risk_meter, plot_keyword_triggers, plot_wordcloud
import pandas as pd

# Load and train
df = load_data("/content/mental-health-detection/data/depression_dataset_reddit_cleaned.csv")
df = preprocess_dataframe(df)
model, tfidf = train_model(df)

# App
st.title("🧠 Mental Health Detection System")
st.write("Analyze social media posts for mental health indicators")

user_input = st.text_area("Enter a social media post:")

if st.button("Analyze"):
    if user_input:
        prob, risk = predict_risk(user_input, model, tfidf)
        sentiment = analyze_sentiment(user_input)
        
        st.subheader("📊 Results")
        
        if risk == "High Risk":
            st.error(f"🔴 {risk} | Probability: {prob:.1%}")
            st.warning("⚠️ Please reach out to a mental health professional!")
        elif risk == "Medium Risk":
            st.warning(f"🟡 {risk} | Probability: {prob:.1%}")
            st.info("💡 Consider talking to someone you trust")
        else:
            st.success(f"🟢 {risk} | Probability: {prob:.1%}")
        
        st.subheader("💭 Sentiment Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment", sentiment["sentiment"])
        col2.metric("Score", f"{sentiment['compound']:.2f}")
        col3.metric("Emoji", sentiment["emoji"])
        
        st.subheader("📈 Risk Meter")
        plot_risk_meter(prob)
        st.pyplot()
