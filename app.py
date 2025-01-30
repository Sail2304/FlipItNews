
import streamlit as st
from utils.preprocessing import preprocess
from utils.predict import load_model, predict
from pathlib import Path
# Load the trained model and TF-IDF vectorizer


# Streamlit UI
st.title("News Classification App")

# Get user input text
user_input = st.text_area("Enter text for classification:")

if st.button("Predict Category"):
    if user_input:
        model = load_model(model_path=Path("Artifacts/Model/model.pkl"))
        pred = predict(model, user_input, Path("Artifacts/Encoders/category.pkl"))
        st.write(pred)
    else:
        st.write("Please enter some text.")
