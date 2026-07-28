##Product Rating Predictor

A machine learning app that predicts a star rating (1–5) from the text of a product
review — trained on Amazon consumer product review data. Built with scikit-learn and
deployed as a Streamlit app.

## Features

Predicts a numeric rating (rounded to 1–5 stars) from raw review text
Uses sentiment polarity (via TextBlob) and review length as model features
Also factors in the number of "helpful votes" a review received
Simple Streamlit interface — paste a review, get an instant prediction

## How it works

1. The review text is scored for sentiment polarity using TextBlob (-1 = negative, +1 = positive)
2. Review length (character count) is calculated
3. These features, plus the helpful-vote count, are fed into a trained regression model
4. The model outputs a continuous rating estimate, which is rounded and clamped to 1–5 stars

## Tech stack

- Python
- Streamlit (UI)
- scikit-learn (model training)
- TextBlob (sentiment analysis)
- pandas / joblib

## Setup

\`\`\`bash
git clone https://github.com/<your-username>/product-rating-predictor.git
cd product-rating-predictor
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Dataset

Amazon product reviews dataset — includes review text, rating, helpful-vote counts,
product metadata (brand, category), and review timestamps.

## Project structure

\`\`\`
product-rating-predictor/
├── app.py                  # Streamlit app
├── productrating.ipynb      # Model training / EDA notebook
├── rating_model.pkl         # Trained model
├── reviewss.csv             # Review dataset
├── requirements.txt
└── README.md
\`\`\`

## Future improvements

- Replace hand-crafted features (length, sentiment) with TF-IDF or embeddings over the
  full review text for stronger predictive power
- Add confidence intervals around the predicted rating
- Support batch predictions from an uploaded CSV of reviews
