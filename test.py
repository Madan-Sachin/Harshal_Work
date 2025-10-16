import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json

# ------------------------
# Streamlit setup
# ------------------------
st.set_page_config(page_title="Emotion Detector", page_icon="💭", layout="centered")
st.title("💭 Emotion Detector (LLM Version)")
st.caption("Detect emotions — Happy, Love, Sad, or Anger — from your text 💫")

# ------------------------
# Google Sheets setup
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
# Load Hugging Face LLM
# ------------------------
@st.cache_resource
def load_model():
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_model()

# ------------------------
# Map output to 4 emotions
# ------------------------
def get_emotion(text):
    prompt = f"""
You are an assistant that detects emotions. Only use one of these emotions: happy, love, sad, anger.
Analyze the text and return the result as a JSON object with the key 'emotion'.

Text: \"{text}\"
JSON output:
"""
    try:
        output = generator(prompt, max_new_tokens=50, do_sample=False)[0]["generated_text"]
        # Extract JSON part
        json_part = output.split("JSON output:")[-1].strip()
        emotion_dict = json.loads(json_part)
        emotion = emotion_dict.get("emotion", "neutral").lower()
        return emotion
    except Exception as e:
        print("Error parsing JSON:", e)
        return "neutral"

# ------------------------
# Emotion color mapping
# ------------------------
def get_emotion_color(emotion):
    colors = {
        "happy": "#FFD93D",
        "love": "#FFB6C1",
        "sad": "#89CFF0",
        "anger": "#FF4B4B",
        "neutral": "#A9A9A9"
    }
    return colors.get(emotion, "#A9A9A9")

# ------------------------
# User input
# ------------------------
user_input = st.text_input("💬 Enter your message:")

if st.button("Submit"):
    if user_input.strip():
        try:
            # Detect emotion
            emotion = get_emotion(user_input)
            color = get_emotion_color(emotion)

            # Display emotion box
            st.markdown("---")
            st.markdown(
                f"""
                <div style='background-color:{color};padding:20px;border-radius:15px;text-align:center;'>
                    <h2 style='color:black;'>Emotion: {emotion.capitalize()}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")

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
    all_messages = sheet.get_all_records()
    if all_messages:
        df = pd.DataFrame(all_messages)
        st.dataframe(df)

        # Emotion distribution chart
        if "emotion" in df.columns:
            emotion_counts = df["emotion"].value_counts().reset_index()
            emotion_counts.columns = ["emotion", "count"]
            import plotly.express as px
            fig = px.pie(
                emotion_counts, values="count", names="emotion",
                title="Emotion Distribution", hole=0.4,
                color_discrete_sequence=["#FFD93D", "#FFB6C1", "#89CFF0", "#FF4B4B", "#A9A9A9"]
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No messages yet.")
except Exception as e:
    st.error(f"Error reading messages: {e}")
