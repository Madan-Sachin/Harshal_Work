import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from transformers import pipeline

# ------------------------
# Streamlit Setup
# ------------------------
st.set_page_config(page_title="Emotion Detector", page_icon="💭", layout="centered")
st.title("💭 Emotion Detector")
st.caption("Detect emotions — Happy, Love, Sad, or Anger — from your text")

# ------------------------
# Google Sheets Setup
# ------------------------
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

try:
    creds_dict = dict(st.secrets["gcp"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("UserInput").sheet1  # Replace with your sheet name
except Exception as e:
    st.error(f"❌ Error connecting to Google Sheets: {e}")
    st.stop()

# ------------------------
# Load emotion classifier (small, cloud-friendly)
# ------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

classifier = load_emotion_model()

# ------------------------
# Map Hugging Face labels to your 4 emotions
# ------------------------
label_mapping = {
    "joy": "happy",
    "love": "love",
    "sadness": "sad",
    "anger": "anger",
    "fear": "sad",   # Map fear to sad for simplicity
    "surprise": "happy"  # Map surprise to happy
}

def predict_emotion(text):
    try:
        result = classifier(text)
        hf_label = result[0]["label"].lower()
        emotion = label_mapping.get(hf_label, "neutral")
        return emotion
    except Exception as e:
        print("Prediction error:", e)
        return "neutral"

# ------------------------
# Emotion colors
# ------------------------
def get_color(emotion):
    return {
        "happy": "#FFD93D",
        "love": "#FFB6C1",
        "sad": "#89CFF0",
        "anger": "#FF4B4B",
        "neutral": "#A9A9A9"
    }.get(emotion, "#A9A9A9")

# ------------------------
# User Input
# ------------------------
user_input = st.text_input("💬 Enter your message:")

if st.button("Submit"):
    if user_input.strip():
        try:
            emotion = predict_emotion(user_input)
            color = get_color(emotion)

            # Display emotion box
            st.markdown(f"""
                <div style='background-color:{color};padding:20px;border-radius:15px;text-align:center;'>
                    <h2>Emotion: {emotion.capitalize()}</h2>
                </div>
            """, unsafe_allow_html=True)

            # Save to Google Sheets
            sheet.append_row([user_input, emotion])
            st.success("✅ Saved to Google Sheets!")

        except Exception as e:
            st.error(f"Error processing message: {e}")
    else:
        st.warning("Please enter a message before submitting.")

# ------------------------
# Show all messages
# ------------------------
st.subheader("📋 All Messages with Detected Emotion")
try:
    records = sheet.get_all_records()
    if records:
        df = pd.DataFrame(records)
        st.dataframe(df)
    else:
        st.write("No messages yet.")
except Exception as e:
    st.error(f"Error reading messages: {e}")
