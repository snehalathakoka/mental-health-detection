
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

def plot_class_distribution(df):
    colors = ["#2ecc71", "#e74c3c"]
    df["is_depression"].value_counts().plot(kind="bar", color=colors)
    plt.title("Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("0 = Not Depressed | 1 = Depressed")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("/content/mental-health-detection/data/class_distribution.png")
    plt.show()

def plot_wordcloud(df, label, title, color):
    text = " ".join(df[df["is_depression"] == label]["clean_text"])
    wordcloud = WordCloud(width=800, height=400, background_color="white",
                         colormap=color, max_words=100).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"/content/mental-health-detection/data/{title}.png")
    plt.show()

def plot_keyword_triggers(model, tfidf):
    feature_names = np.array(tfidf.get_feature_names_out())
    coefficients = model.coef_[0]
    top_depression_idx = coefficients.argsort()[-20:][::-1]
    top_normal_idx = coefficients.argsort()[:20]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.barh(feature_names[top_depression_idx],
             coefficients[top_depression_idx], color="#e74c3c")
    ax1.set_title("Depression Trigger Words", fontweight="bold")
    ax1.set_xlabel("Importance Score")
    ax2.barh(feature_names[top_normal_idx],
             abs(coefficients[top_normal_idx]), color="#2ecc71")
    ax2.set_title("Non-Depression Words", fontweight="bold")
    ax2.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("/content/mental-health-detection/data/keyword_triggers.png")
    plt.show()

def plot_sentiment_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="sentiment_score",
                 hue="is_depression",
                 palette=["#2ecc71", "#e74c3c"], bins=30)
    plt.title("Sentiment Score Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Sentiment Score (-1 Negative to +1 Positive)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("/content/mental-health-detection/data/sentiment_distribution.png")
    plt.show()

def plot_risk_meter(probability):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]
    labels = ["Healthy", "Mild Concern", "Moderate Risk", "High Risk", "Critical"]
    boundaries = [0, 0.20, 0.40, 0.70, 0.85, 1.0]
    for i in range(5):
        ax.barh(0, boundaries[i+1] - boundaries[i],
                left=boundaries[i], color=colors[i],
                height=0.3, label=labels[i])
    ax.axvline(x=probability, color="black",
               linewidth=3, linestyle="--",
               label=f"Score: {probability:.1%}")
    ax.set_xlim(0, 1)
    ax.set_title("Risk Probability Meter", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("/content/mental-health-detection/data/risk_meter.png")
    plt.show()

def plot_emotion_radar(emotion_scores, title="Emotion Radar Chart"):
    emotions = list(emotion_scores.keys())
    values = list(emotion_scores.values())
    values += values[:1]
    angles = [n / float(len(emotions)) * 2 * np.pi for n in range(len(emotions))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, color="#e74c3c")
    ax.fill(angles, values, alpha=0.25, color="#e74c3c")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotions, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("/content/mental-health-detection/data/emotion_radar.png")
    plt.show()

def plot_emotion_timeline(posts, results, sentiments):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    x = range(1, len(posts) + 1)
    probabilities = [r["probability"] for r in results]
    sentiment_scores = [s["compound"] for s in sentiments]
    colors = []
    for r in results:
        if r["risk"] == "Critical": colors.append("#e74c3c")
        elif r["risk"] == "High Risk": colors.append("#e67e22")
        elif r["risk"] == "Moderate Risk": colors.append("#f39c12")
        elif r["risk"] == "Mild Concern": colors.append("#3498db")
        else: colors.append("#2ecc71")
    ax1.plot(x, probabilities, "o-", linewidth=2, color="#e74c3c", markersize=8)
    ax1.fill_between(x, probabilities, alpha=0.2, color="#e74c3c")
    for i, (xi, yi, r) in enumerate(zip(x, probabilities, results)):
        ax1.annotate(r["risk"], (xi, yi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax1.set_title("Mental Health Risk Timeline", fontweight="bold", fontsize=13)
    ax1.set_ylabel("Risk Probability")
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="High Risk threshold")
    ax1.axhline(y=0.4, color="orange", linestyle="--", alpha=0.5, label="Moderate threshold")
    ax1.legend(fontsize=8)
    ax1.set_xticks(x)
    ax2.bar(x, sentiment_scores,
            color=["#e74c3c" if s < -0.2 else "#2ecc71" if s > 0.2 else "#95a5a6"
                   for s in sentiment_scores])
    ax2.set_title("Emotion Timeline - Sentiment Scores", fontweight="bold", fontsize=13)
    ax2.set_ylabel("Sentiment Score")
    ax2.set_xlabel("Post Number")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xticks(x)
    plt.tight_layout()
    plt.savefig("/content/mental-health-detection/data/emotion_timeline.png")
    plt.show()
