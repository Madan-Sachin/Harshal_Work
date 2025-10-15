import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import json

# ------------------------
# Streamlit page config
# ------------------------
st.set_page_config(page_title="User Input Collector")

st.title("User Input Collector")

# ------------------------
# Google Sheets setup
# ------------------------
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

# Load credentials from Streamlit Secrets
try:
    creds_dict = dict(st.secrets["gcp"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    
    # Open your sheet by name
    sheet = client.open("UserInput").sheet1  # Replace with your Google Sheet name

except Exception as e:
    st.error(f"Error connecting to Google Sheets: {e}")
    st.stop()

# ------------------------
# User input form
# ------------------------
user_input = st.text_input("Enter your message:")

if st.button("Submit"):
    if user_input.strip() != "":
        try:
            sheet.append_row([user_input])
            st.success("✅ Your message has been saved!")
        except Exception as e:
            st.error(f"Error saving message: {e}")
    else:
        st.warning("Please enter a message.")

# ------------------------
# Display all messages
# ------------------------
st.subheader("All messages so far:")

try:
    all_messages = sheet.get_all_records()
    if all_messages:
        df = pd.DataFrame(all_messages)
        st.dataframe(df)
    else:
        st.write("No messages yet.")
except Exception as e:
    st.error(f"Error reading messages: {e}")

