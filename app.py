
import streamlit as st
import sys
sys.path.append("/content/mental-health-detection")
from src.preprocess import load_data, preprocess_dataframe
from src.model import train_model, predict_risk, save_model
from src.sentiment import analyze_sentiment, get_emotional_pattern
from src.visualize import plot_risk_meter, plot_emotion_radar
import matplotlib.pyplot as plt

# Load and train
@st.cache_resource
def load_and_train():
    df = load_data("/content/mental-health-detection/data/depression_dataset_reddit_cleaned.csv")
    df = preprocess_dataframe(df)
    model, tfidf = train_model(df)
    return model, tfidf

st.set_page_config(page_title="Mental Health Detection", page_icon="🧠", layout="wide")

st.title("🧠 Mental Health Detection System")
st.write("Analyzing social media posts for mental health indicators using ML")

with st.spinner("Loading model..."):
    model, tfidf = load_and_train()

st.success("✅ Model Ready!")

# Input
st.subheader("📝 Enter a Social Media Post")
user_input = st.text_area("Paste any social media post here:", height=150)

if st.button("🔍 Analyze", use_container_width=True):
    if user_input:
        result = predict_risk(user_input, model, tfidf)
        sentiment = analyze_sentiment(user_input)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Mental Health Classification")
            if result["risk"] == "Critical":
                st.error(f"{result['emoji']} {result['risk']} — {result['probability']:.1%}")
                st.error(f"⚠️ {result['message']}")
                st.warning(f"💡 {result['suggestion']}")
            elif result["risk"] == "High Risk":
                st.error(f"{result['emoji']} {result['risk']} — {result['probability']:.1%}")
                st.warning(f"💡 {result['suggestion']}")
            elif result["risk"] == "Moderate Risk":
                st.warning(f"{result['emoji']} {result['risk']} — {result['probability']:.1%}")
                st.info(f"💡 {result['suggestion']}")
            elif result["risk"] == "Mild Concern":
                st.info(f"{result['emoji']} {result['risk']} — {result['probability']:.1%}")
                st.info(f"💡 {result['suggestion']}")
            else:
                st.success(f"{result['emoji']} {result['risk']} — {result['probability']:.1%}")
                st.success(f"💡 {result['suggestion']}")

            st.subheader("💭 Emotional Pattern Insights")
            st.metric("Sentiment", f"{sentiment['emoji']} {sentiment['sentiment']}")
            st.metric("Sentiment Score", f"{sentiment['compound']:.2f}")
            st.metric("Emotional Intensity", sentiment["emotional_intensity"])
            st.metric("Dominant Emotion", sentiment["dominant_emotion"])

            st.subheader("🚩 Behavioral Patterns")
            for pattern in sentiment["behavioral_patterns"]:
                st.write(pattern)

        with col2:
            st.subheader("📊 Risk Meter")
            plot_risk_meter(result["probability"])
            st.pyplot(plt.gcf())
            plt.clf()

            st.subheader("🎭 Emotion Radar")
            plot_emotion_radar(sentiment["emotion_scores"])
            st.pyplot(plt.gcf())
            plt.clf()
    else:
        st.warning("Please enter a post to analyze!")

st.markdown("---")
st.caption("🧠 Mental Health Detection System | Built with ML + NLP")
