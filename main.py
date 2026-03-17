
import sys
sys.path.append("/content/mental-health-detection")

from src.preprocess import load_data, preprocess_dataframe
from src.model import train_model, predict_risk, save_model
from src.sentiment import analyze_sentiment, get_emotional_pattern
from src.visualize import (plot_class_distribution, plot_wordcloud,
                           plot_keyword_triggers, plot_sentiment_distribution,
                           plot_risk_meter)

def main():
    print("=" * 50)
    print("🧠 Mental Health Detection System")
    print("=" * 50)

    # Step 1 - Load & Preprocess
    print("\n📂 Loading dataset...")
    df = load_data("/content/mental-health-detection/data/depression_dataset_reddit_cleaned.csv")
    df = preprocess_dataframe(df)
    print(f"✅ Dataset loaded: {df.shape[0]} posts")

    # Step 2 - Visualize Class Distribution
    print("\n📊 Plotting class distribution...")
    plot_class_distribution(df)

    # Step 3 - Train Model
    print("\n🤖 Training model...")
    model, tfidf = train_model(df)
    save_model(model, tfidf)

    # Step 4 - Keyword Triggers
    print("\n🔑 Plotting keyword triggers...")
    plot_keyword_triggers(model, tfidf)

    # Step 5 - Sentiment Analysis
    print("\n💭 Analyzing emotional patterns...")
    df = get_emotional_pattern(df)
    plot_sentiment_distribution(df)

    # Step 6 - Word Clouds
    print("\n☁️ Generating word clouds...")
    plot_wordcloud(df, 1, "Depression WordCloud", "Reds")
    plot_wordcloud(df, 0, "Non-Depression WordCloud", "Greens")

    # Step 7 - Test Prediction
    print("\n🔍 Testing predictions...")
    test_posts = [
        "I feel hopeless and empty, nothing makes me happy anymore",
        "Had a great day today, feeling blessed and happy!",
        "I cant sleep, everything feels pointless and dark"
    ]

    for post in test_posts:
        prob, risk = predict_risk(post, model, tfidf)
        sentiment = analyze_sentiment(post)
        
        print("\n" + "="*50)
        print(f"📝 Post: {post[:60]}...")
        print(f"🎯 Mental Health Classification: {risk}")
        print(f"📊 Risk Probability: {prob:.1%}")
        print(f"💭 Sentiment: {sentiment['emoji']} {sentiment['sentiment']}")
        print(f"📈 Sentiment Score: {sentiment['compound']:.2f}")
        
        # Risk suggestions
        if risk == "High Risk":
            print("⚠️  Early Warning: This person may need immediate support!")
            print("💡 Suggestion: Please reach out to a mental health professional")
        elif risk == "Medium Risk":
            print("⚠️  Moderate concern detected")
            print("💡 Suggestion: Consider talking to someone you trust")
        else:
            print("✅ No immediate concern detected")
        
        plot_risk_meter(prob)

    print("\n✅ Analysis Complete!")

if __name__ == "__main__":
    main()
