import streamlit as st
import pickle

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# App Title
st.title("Fake News Detection App")

# Text input
news_text = st.text_area("Enter the News Text:")

# Predict button
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize the input
        transformed_text = vectorizer.transform([news_text])

        # Predict
        prediction = model.predict(transformed_text)

        # Show result
        st.success(f"This news is: {prediction[0]}")
