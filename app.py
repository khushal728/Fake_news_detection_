import streamlit as st
import requests
import pickle
import cohere
from bs4 import BeautifulSoup
import google.generativeai as genai

COHERE_API_KEY = "7gASb38xz7rJYESOJs0JunSLzfTgqHvbbav8tFiQ"

# 🔑 Hardcoded API keys (Replace with your own!)
TAVILY_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyBtsj5HlYcv_c-TSa82jOMwXq5N2Xc1PNo"
# Setup Cohere client
co = cohere.Client(COHERE_API_KEY)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Fake News Detector + News Summarizer",
    layout="wide",
    page_icon="📰"
)

# ----------------------------
# Custom CSS for better UI
# ----------------------------
st.markdown("""
    <style>
    .headline-card {
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        background-color: #040404;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .summary-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #040404;
        border: 1px solid #d6e4ff;
    }
    </style>
""", unsafe_allow_html=True)

# Load ML pipeline
MODEL_FILE = "text_pipeline.pkl"
with open(MODEL_FILE, "rb") as f:
    pipeline = pickle.load(f)

vectorizer = pipeline["vectorizer"]
model = pipeline["model"]
label_encoder = pipeline["label_encoder"]

# ----------------------------
# App Title
# ----------------------------
st.title("📰 Fake News Detector + Smart Summarizer")
st.write("Check if a news headline might be fake and summarize full news articles instantly.")

# ----------------------------
# Headline Classification
# ----------------------------
st.header("🔎 Headline Classification")

title = st.text_input("Enter a news headline")

if st.button("Predict", key="predict_headline"):
    if not title:
        st.error("Enter a headline to classify.")
    else:
        X_t = vectorizer.transform([title])
        pred = model.predict(X_t)
        label = label_encoder.inverse_transform(pred)[0]
        label_text = "✅ TRUE" if label.lower() in ["true", "1", "real"] else "❌ FALSE"
        st.markdown(f"<div class='headline-card'><b>Prediction:</b> {label_text}</div>", unsafe_allow_html=True)

# ----------------------------
# Fetch Live Headlines (Tavily)
# ----------------------------
st.header("🌍 Fetch Live Headlines")

if "fetched_headlines" not in st.session_state:
    st.session_state.fetched_headlines = []

if st.button("Fetch Latest Headlines", key="fetch_headlines"):
    st.session_state.fetched_headlines.clear()
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {TAVILY_API_KEY}"}
    payload = {"query": "latest world news", "search_depth": "advanced", "include_answer": True, "max_results": 5}

    try:
        r = requests.post(url, json=payload, headers=headers)
        data = r.json()
        if "results" in data:
            for res in data["results"]:
                headline = res.get("title", "No title")
                if headline:
                    st.session_state.fetched_headlines.append(headline)
                    X_t = vectorizer.transform([headline])
                    pred = model.predict(X_t)
                    label = label_encoder.inverse_transform(pred)[0]
                    label_text = "✅ TRUE" if label.lower() in ["true", "1", "real"] else "❌ FALSE"
                    st.markdown(f"<div class='headline-card'><b>{headline}</b><br>{label_text}<br><a href='{res.get('url','')}' target='_blank'>{res.get('url','')}</a></div>", unsafe_allow_html=True)
        else:
            st.warning("No results found from Tavily.")
    except Exception as e:
        st.error("Failed to fetch news: " + str(e))

# Summarize Any News Article using Cohere Chat
st.header("📝 Summarize Any News Article (using Cohere)")

article_url = st.text_input("Paste a news article link here")

if st.button("Summarize Article with Cohere"):
    if not article_url:
        st.error("Please paste a valid news article link.")
    else:
        try:
            # Step A: Extract article content via Tavily or via scraping
            res = requests.post(
                "https://api.tavily.com/extract",
                json={"url": article_url},
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {TAVILY_API_KEY}"}
            )
            data = res.json()
            article_text = data.get("content", "")

            if not article_text:
                st.error("Could not extract article content. Try another link.")
            else:
                # Step B: Use Cohere Chat API
                co = cohere.ClientV2(api_key=COHERE_API_KEY)

                prompt = f"Summarize the following news article in simple, clear bullet points:\n\n{article_text}"

                resp = co.chat(
                    model="command-a-03-2025",
                    messages=[
                        {"role": "system", "content": "You are a summarizer that makes bullet-point summaries."},
                        {"role": "user", "content": prompt}
                    ]
                )

                summ = resp.message.content[0].text

                st.subheader("📌 Article Summary")
                st.markdown(f"• " + summ.replace("\n", "\n• "))

        except Exception as e:
            st.error(f"Error summarizing article with Cohere: {e}")
