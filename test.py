import streamlit as st
import csv
import os

st.title("Small Project Input Collector")

user_input = st.text_input("Enter your message:")

if st.button("Submit"):
    if user_input.strip() != "":
        file_exists = os.path.isfile("user_inputs.csv")
        with open("user_inputs.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Message"])  # header
            writer.writerow([user_input])
        st.success("✅ Message saved!")
    else:
        st.warning("Please enter a message.")

# Show all messages while the app is running
if os.path.exists("user_inputs.csv"):
    with open("user_inputs.csv", "r") as f:
        st.write(f.read())
