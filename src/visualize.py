
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import pandas as pd

def plot_class_distribution(df):
    """Plot distribution of depression vs non-depression"""
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
    """Generate word cloud for depression or non-depression posts"""
    text = " ".join(df[df["is_depression"] == label]["clean_text"])
    wordcloud = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap=color,
        max_words=100
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"/content/mental-health-detection/data/{title}.png")
    plt.show()

def plot_keyword_triggers(model, tfidf):
    """Plot top depression and non-depression keywords"""
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
    """Plot sentiment distribution across dataset"""
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="sentiment_score", 
                 hue="is_depression",
                 palette=["#2ecc71", "#e74c3c"],
                 bins=30)
    plt.title("Sentiment Score Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Sentiment Score (-1 Negative to +1 Positive)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("/content/mental-health-detection/data/sentiment_distribution.png")
    plt.show()

def plot_risk_meter(probability):
    """Plot a risk meter for individual prediction"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    boundaries = [0, 0.4, 0.7, 1.0]
    
    for i in range(3):
        ax.barh(0, boundaries[i+1] - boundaries[i],
                left=boundaries[i], color=colors[i],
                height=0.3, label=labels[i])
    
    ax.axvline(x=probability, color="black", 
               linewidth=3, linestyle="--", label=f"Score: {probability:.1%}")
    
    ax.set_xlim(0, 1)
    ax.set_title("Risk Probability Meter", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("/content/mental-health-detection/data/risk_meter.png")
    plt.show()
