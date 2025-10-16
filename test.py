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
st.title("💭 Emotion Detector App")
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
# Load Hugging Face model
# ------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True  # important for consistent output
    )

emotion_model = load_model()

# ------------------------
# Map model outputs to 4 custom emotions
# ------------------------
def map_to_custom_emotion(model_outputs, text=""):
    """
    model_outputs: list of dicts [{'label': 'joy', 'score': 0.9}, ...]
    text: original user input for optional heuristic
    """
    # Optional heuristic for love
    text_lower = text.lower()
    if any(word in text_lower for word in ["love", "darling", "sweetheart"]):
        return "love", 1.0

    # Find top label by score
    top_result = max(model_outputs, key=lambda x: x["score"])
    label = top_result["label"].lower()
    score = top_result["score"]

    # Map to 4 emotions
    if label in ["joy", "happiness"]:
        return "happy", score
    elif label == "love":
        return "love", score
    elif label in ["sadness", "sad"]:
        return "sad", score
    elif label == "anger":
        return "anger", score
    else:
        return "neutral", score

# ------------------------
# Emotion color mapping
# ------------------------
def get_emotion_color(emotion):
    colors = {
        "happy": "#FFD93D",     # yellow
        "love": "#FFB6C1",      # pink
        "sad": "#89CFF0",       # blue
        "anger": "#FF4B4B",     # red
        "neutral": "#A9A9A9"    # grey
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
            model_outputs = emotion_model(user_input)[0]  # returns a list of dicts
            final_emotion, score = map_to_custom_emotion(model_outputs, user_input)
            color = get_emotion_color(final_emotion)

            # Display styled emotion box
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
