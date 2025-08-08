import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import pipeline

# Load sentiment analysis pipeline from Hugging Face
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"  # Force using PyTorch
)

# Function to classify text into sentiment categories
def classify_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    if label == "POSITIVE":
        return "Positive"
    elif label == "NEGATIVE":
        return "Negative"
    else:
        return "Neutral"

# Function to create word cloud
def generate_wordcloud(df, sentiment_label):
    text = " ".join(df[df['sentiment'] == sentiment_label]['text'])
    wc = WordCloud(width=600, height=400, background_color="white").generate(text)
    return wc

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("🤖 Sentiment Analysis Dashboard (Hugging Face)")

# Upload CSV or use default
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default sample data...")
    df = pd.DataFrame({
        "text": [
            "I love this phone!",
            "This product is terrible.",
            "It was okay, not the best.",
            "Absolutely fantastic experience!",
            "Would not recommend at all.",
            "Neutral about the service.",
            "Exceeded my expectations!",
            "Poor quality and bad design."
        ]
    })

# Run sentiment analysis
st.subheader("📊 Analyzing Sentiment...")
df['sentiment'] = df['text'].apply(classify_sentiment)

# Show sample data
st.dataframe(df.head())

# Sentiment count bar chart
st.subheader("📈 Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x='sentiment', palette='Set2', ax=ax)
st.pyplot(fig)

# Word clouds
st.subheader("☁️ Word Clouds by Sentiment")
col1, col2, col3 = st.columns(3)
for sentiment, col in zip(['Positive', 'Neutral', 'Negative'], [col1, col2, col3]):
    with col:
        st.markdown(f"**{sentiment}**")
        try:
            wc_img = generate_wordcloud(df, sentiment)
            st.image(wc_img.to_array())
        except:
            st.write("Not enough data.")

# Real-time user input
st.subheader("🗣️ Real-Time Sentiment Detector")
user_input = st.text_input("Type a sentence:")
if user_input:
    result = classify_sentiment(user_input)
    st.success(f"Sentiment: **{result}**")