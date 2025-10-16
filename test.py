import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from transformers import pipeline
import plotly.express as px

# ------------------------
# Streamlit setup
# ------------------------
st.set_page_config(page_title="Emotion Detector", page_icon="💭", layout="centered")

st.title("💭 Emotion Detection App")
st.caption("Detect the emotion behind your message — *Anger*, *Happy*, or *Romantic* 💫")

# ------------------------
# Google Sheets setup
# ------------------------
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

try:
    creds_dict = dict(st.secrets["gcp"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("UserInput").sheet1  # replace with your sheet name
except Exception as e:
    st.error(f"❌ Error connecting to Google Sheets: {e}")
    st.stop()

# ------------------------
# Load Hugging Face model
# ------------------------
@st.cache_resource
def load_model():
    return pipeline("text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True)

emotion_model = load_model()

# ------------------------
# Custom emotion mapper
# ------------------------
def map_to_custom_emotion(label):
    label = label.lower()
    if label in ["anger", "disgust"]:
        return "anger"
    elif label in ["joy"]:
        return "happy"
    elif label in ["love"]:
        return "romantic"
    else:
        return "neutral"

# ------------------------
# Emotion color mapping
# ------------------------
def get_emotion_color(emotion):
    colors = {
        "anger": "#FF4B4B",      # red
        "happy": "#FFD93D",      # yellow
        "romantic": "#FFB6C1",   # pink
        "neutral": "#A9A9A9"     # grey
    }
    return colors.get(emotion, "#A9A9A9")

# ------------------------
# User input
# ------------------------
user_input = st.text_input("💬 Enter your message:")

if st.button("Submit"):
    if user_input.strip():
        try:
            # Predict emotion
            results = emotion_model(user_input)[0]
            top_result = max(results, key=lambda x: x["score"])
            raw_emotion = top_result["label"]
            score = round(top_result["score"], 3)

            # Map to your custom emotions
            final_emotion = map_to_custom_emotion(raw_emotion)
            color = get_emotion_color(final_emotion)

            # 🎨 Styled Emotion Box
            st.markdown("---")
            st.markdown(
                f"""
                <div style='background-color:{color};padding:20px;border-radius:15px;text-align:center;'>
                    <h2 style='color:black;'>Emotion: {final_emotion.capitalize()}</h2>
                    <p style='font-size:18px;'>Confidence: {score*100:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")

            # Save to Google Sheets
            sheet.append_row([user_input, final_emotion, score])
            st.success("✅ Saved to Google Sheets!")

        except Exception as e:
            st.error(f"Error processing message: {e}")
    else:
        st.warning("Please enter a message before submitting.")

# ------------------------
# Display all stored data
# ------------------------
st.subheader("📋 All Messages with Detected Emotion")

try:
    all_messages = sheet.get_all_records()
    if all_messages:
        df = pd.DataFrame(all_messages)
        st.dataframe(df)

        # Emotion distribution chart
        if "emotion" in df.columns:
            emotion_counts = df["emotion"].value_counts().reset_index()
            emotion_counts.columns = ["emotion", "count"]
            fig = px.pie(emotion_counts, values="count", names="emotion",
                         title="Emotion Distribution", hole=0.4,
                         color_discrete_sequence=["#FF4B4B", "#FFD93D", "#FFB6C1", "#A9A9A9"])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No messages yet.")
except Exception as e:
    st.error(f"Error reading messages: {e}")
