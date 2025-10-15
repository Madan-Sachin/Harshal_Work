import streamlit as st

st.title("Simple Text Input App")

# Take text input from user
user_input = st.text_input("Enter something:")

# When user clicks submit button
if st.button("Submit"):
    st.write(f"You entered: {user_input}")
    
    # You can use 'user_input' variable in backend logic here
    # Example:
    # process_input(user_input)
