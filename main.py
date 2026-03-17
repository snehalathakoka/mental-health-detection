
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

    print("\n📂 Loading dataset...")
    df = load_data("/content/mental-health-detection/data/depression_dataset_reddit_cleaned.csv")
    df = preprocess_dataframe(df)
    print(f"✅ Dataset loaded: {df.shape[0]} posts")

    plot_class_distribution(df)

    print("\n🤖 Training model...")
    model, tfidf = train_model(df)
    save_model(model, tfidf)

    plot_keyword_triggers(model, tfidf)

    df = get_emotional_pattern(df)
    plot_sentiment_distribution(df)

    plot_wordcloud(df, 1, "Depression WordCloud", "Reds")
    plot_wordcloud(df, 0, "Non-Depression WordCloud", "Greens")

    # Real Life Reddit Examples
    print("\n🔍 Real Life Social Media Post Analysis...")
    test_posts = [
        # Real depression posts from Reddit
        "I have been feeling really low lately. I dont want to get out of bed, I dont want to eat, I dont want to talk to anyone. Everything feels like too much effort and I dont see the point anymore.",
        "Just got back from the best vacation ever with my family! Feeling so recharged and grateful for every moment. Life is beautiful!",
        "Its 3am and I cant stop crying and I dont even know why. I feel so empty and alone. Nobody understands what im going through. I just want the pain to stop.",
        "Had a rough week but managed to finish my project. Feeling a little overwhelmed but staying positive. Taking it one day at a time.",
        "I have been isolating myself for weeks now. I stopped responding to texts, stopped going to class. I feel like a burden to everyone around me. Sometimes I think everyone would be better off without me.",
        "Excited to start my new job tomorrow! Nervous but ready for this new chapter. Hard work is finally paying off!"
    ]

    # Labels for context
    labels = [
        "Reddit r/depression user",
        "Reddit r/happy user", 
        "Reddit r/lonely user",
        "Reddit r/stress user",
        "Reddit r/depression user",
        "Reddit r/GetMotivated user"
    ]

    all_results = []
    all_sentiments = []

    for i, (post, label) in enumerate(zip(test_posts, labels)):
        result = predict_risk(post, model, tfidf)
        sentiment = analyze_sentiment(post)
        all_results.append(result)
        all_sentiments.append(sentiment)

        print("\n" + "="*60)
        print(f"👤 Source: {label}")
        print(f"📝 Post: {post[:80]}...")
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

        plot_emotion_radar(sentiment["emotion_scores"],
                          f"Emotion Radar - {label}")
        plot_risk_meter(result["probability"])

    # Emotion Timeline
    print("\n📈 Generating Emotion Timeline...")
    plot_emotion_timeline(test_posts, all_results, all_sentiments)

    print("\n" + "="*60)
    print("✅ Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
