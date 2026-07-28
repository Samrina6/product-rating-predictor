import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob

model = joblib.load("rating_model.pkl")

st.title("🛒 Product Rating Predictor")
st.write("Enter a product review below to predict its rating")

review_text = st.text_area("Write your review here", "This product is great and works well!")
num_helpful = st.number_input("Number of Helpful Votes", min_value=0, value=0)

if st.button("Predict Rating"):
    review_length = len(review_text)
    sentiment = TextBlob(review_text).sentiment.polarity
    
    input_data = pd.DataFrame([[review_length, num_helpful, sentiment]],
                               columns=['review_length', 'reviews.numHelpful', 'sentiment'])
    prediction = model.predict(input_data)
    rounded = round(prediction[0])
    rounded = max(1, min(5, rounded))
    
    st.success(f"⭐ Predicted Rating: {prediction[0]:.2f} (≈ {rounded} stars)")
    st.write(f"Detected sentiment score: {sentiment:.2f}")