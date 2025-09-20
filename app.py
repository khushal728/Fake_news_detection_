import streamlit as st
import base64
import google.generativeai as genai

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 AI/ML Solution for Detecting Fake News and Misinformation")
st.write("Enter a news headline/article or upload an image. The AI will:\n"
         "1️⃣ Tell if it is TRUE or FAKE\n"
         "2️⃣ Summarize the news\n"
         "3️⃣ Explain the reasoning (truth check).")

# 🔑 Gemini API key
API_KEY = "AIzaSyByTqiqKsFxH0zWMX3Q6gdYR1CE5pgIi0w"   # replace with your key
genai.configure(api_key=API_KEY)

user_input = st.text_area("Enter News Headline or Article", "", height=150, key="news_input")
uploaded_image = st.file_uploader("Upload an Image (optional)", type=["jpg", "jpeg", "png"], key="news_image")

def analyze_news(text, image_file):
    try:
        analysis_prompt = (
            f"Analyze this news text (and image if provided) carefully. "
            f"1. Classify it strictly as TRUE or FAKE.\n"
            f"2. Give a short 1-2 line summary.\n"
            f"3. Explain why it is true or fake."
            f"\n\nNews:\n{text}"
        )

        # Build parts for Gemini input
        parts = [{"text": analysis_prompt}]
        if image_file:
            image_bytes = image_file.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_b64
                }
            })

        # Use Gemini SDK
        model = genai.GenerativeModel("gemini-1.5-flash")    # ✅ correct model name
        response = model.generate_content(parts)

        full_text = response.text

        classification, summary, reasoning = "", "", ""

        for line in full_text.splitlines():
            if "fake" in line.lower() or "true" in line.lower():
                classification = line
            elif "summary" in line.lower():
                summary = line
            elif "because" in line.lower() or "reason" in line.lower():
                reasoning = line

        if "fake" in classification.lower():
            st.error(f"🚨 Fake News Detected!\n\n{classification}")
        elif "true" in classification.lower():
            st.success(f"✅ Real News Detected!\n\n{classification}")

        if summary:
            st.info(f"📝 **Summary:** {summary}")
        if reasoning:
            st.write(f"🔍 **Truth Check:** {reasoning}")

    except Exception as e:
        st.error(f"Error: {e}")

# AUTO RUN whenever input changes
if user_input.strip() or uploaded_image:
    with st.spinner("Analyzing..."):
        analyze_news(user_input, uploaded_image)

st.caption("Powered by Google Gemini API (Real-time Text + Image Analysis) & Streamlit")
