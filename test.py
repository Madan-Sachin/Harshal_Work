import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import plotly.express as px

# ------------------------
# Streamlit Setup
# ------------------------
st.set_page_config(page_title="Emotion Detector", page_icon="💭", layout="centered")
st.title("💭 Emotion Detector (Cloud-ready)")
st.caption("Detect emotions — Happy, Love, Sad, or Anger — from your text 💫")

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
# Load LLM
# ------------------------
@st.cache_resource
def load_llm():
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_llm()

# ------------------------
# Emotion detection
# ------------------------
def get_emotion(text):
    prompt = f"""
You are an assistant that detects emotions. Only use one of these emotions: happy, love, sad, anger.
Analyze the text and return the result as a JSON object with the key 'emotion'.

Text: "{text}"
JSON output:
"""
    try:
        output = generator(prompt, max_new_tokens=50, do_sample=False)[0]["generated_text"]
        json_part = output.split("JSON output:")[-1].strip()
        emotion_dict = json.loads(json_part)
        return emotion_dict.get("emotion", "neutral").lower()
    except Exception as e:
        print("JSON parse error:", e)
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
            emotion = get_emotion(user_input)
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

        # Pie chart
        if "emotion" in df.columns:
            counts = df["emotion"].value_counts().reset_index()
            counts.columns = ["emotion", "count"]
            fig = px.pie(counts, values="count", names="emotion",
                         color="emotion",
                         color_discrete_map={
                             "happy": "#FFD93D",
                             "love": "#FFB6C1",
                             "sad": "#89CFF0",
                             "anger": "#FF4B4B",
                             "neutral": "#A9A9A9"
                         },
                         title="Emotion Distribution", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No messages yet.")
except Exception as e:
    st.error(f"Error reading messages: {e}")
