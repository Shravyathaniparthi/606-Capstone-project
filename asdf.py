import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import joblib

# Load the trained Logistic Regression model using joblib
logreg = joblib.load('logreg_model.joblib')

# Load the fitted TF-IDF vectorizer using joblib
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

def run_sentiment_analysis(review_input):
    # Transform the review_input using the loaded TF-IDF vectorizer
    review_vector = tfidf_vectorizer.transform([review_input])

    # Predict using the Logistic Regression model
    logreg_pred = logreg.predict(review_vector)

    return logreg_pred

def main():
    # Main function to define the Streamlit app
    st.title("Sentiment Analysis Web App")

    # Streamlit user interface for prediction
    user_input = st.text_input("Enter a review:")
    if st.button("Predict"):
        # Call the prediction function
        logreg_pred = run_sentiment_analysis(user_input)

        # Display the prediction
        st.write("Logistic Regression prediction:", logreg_pred[0])

if __name__ == '__main__':
    main()
