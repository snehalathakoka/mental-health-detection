
import sys
sys.path.append("/content/mental-health-detection")
from src.preprocess import load_data, preprocess_dataframe
from src.model import train_model, predict_risk, save_model
from src.sentiment import analyze_sentiment, get_emotional_pattern
from src.visualize import (plot_class_distribution, plot_wordcloud,
                           plot_keyword_triggers, plot_sentiment_distribution,
                           plot_risk_meter, plot_emotion_radar, 
                           plot_emotion_timeline)

def main():
    print("=" * 60)
    print("🧠 Mental Health Detection System")
    print("=" * 60)

    # Step 1 - Load & Preprocess
    print("\n📂 Loading dataset...")
    df = load_data("/content/mental-health-detection/data/depression_dataset_reddit_cleaned.csv")
    df = preprocess_dataframe(df)
    print(f"✅ Dataset loaded: {df.shape[0]} posts")

    # Step 2 - Class Distribution
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

    # Step 7 - Real Life Examples
    print("\n🔍 Analyzing real life examples...")
    test_posts = [
        "I feel hopeless and empty, nothing makes me happy anymore. I cry every night and feel completely alone.",
        "Had an amazing day today! Got promoted at work, feeling so blessed and grateful!",
        "I cant sleep again. Everything feels pointless and dark. Whats the point of anything.",
        "Feeling a bit stressed about exams but im managing. Hope things get better soon.",
        "I hate myself. Im worthless and a burden to everyone around me. I want it all to end."
    ]

    all_results = []
    all_sentiments = []

    for i, post in enumerate(test_posts):
        result = predict_risk(post, model, tfidf)
        sentiment = analyze_sentiment(post)
        all_results.append(result)
        all_sentiments.append(sentiment)

        print("\n" + "="*60)
        print(f"📝 Post {i+1}: {post[:70]}...")
        print(f"\n🎯 MENTAL HEALTH CLASSIFICATION:")
        print(f"   {result['emoji']} {result['risk']} | Probability: {result['probability']:.1%}")
        print(f"   ⚠️  {result['message']}")
        print(f"   💡 {result['suggestion']}")
        print(f"\n💭 EMOTIONAL PATTERN INSIGHTS:")
        print(f"   Sentiment: {sentiment['emoji']} {sentiment['sentiment']}")
        print(f"   Sentiment Score: {sentiment['compound']:.2f}")
        print(f"   Emotional Intensity: {sentiment['emotional_intensity']}")
        print(f"   Dominant Emotion: {sentiment['dominant_emotion']}")
        print(f"\n🚩 BEHAVIORAL PATTERNS:")
        for pattern in sentiment["behavioral_patterns"]:
            print(f"   {pattern}")

        # Radar chart for each post
        plot_emotion_radar(sentiment["emotion_scores"], 
                          f"Emotion Radar - Post {i+1}")
        plot_risk_meter(result["probability"])

    # Step 8 - Emotion Timeline
    print("\n📈 Generating Emotion Timeline...")
    plot_emotion_timeline(test_posts, all_results, all_sentiments)

    print("\n" + "="*60)
    print("✅ Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
