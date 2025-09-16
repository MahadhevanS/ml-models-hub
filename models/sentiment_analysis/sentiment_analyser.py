import streamlit as st
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def load_resources():
    """Loads the model, stemmer, and stopwords and caches them."""
    try:
        model = joblib.load('models/sentiment_analysis/sentiment_analyzer.pkl')
        port_stem = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        return model, port_stem, stop_words
    except FileNotFoundError:
        st.error("Error: 'sentiment_analyzer.pkl' not found. Please ensure the model file is in the same directory.")
        return None, None, None

model, port_stem, stop_words = load_resources()

def stemming(content):
    """Preprocesses a single string for sentiment analysis."""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        port_stem.stem(word) for word in stemmed_content 
        if word not in stop_words
    ]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


def sentiment_analyser():
    st.write("Enter a tweet below to analyze its sentiment (positive or negative).")

    user_input = st.text_area("Enter your tweet here:", "")

    if st.button("Analyze Sentiment"):
        if user_input and model:
            with st.spinner('Analyzing...'):
                processed_input = stemming(user_input)
                
                prediction = model.predict([processed_input])
                
                sentiment_map = {0: "Negative", 1: "Positive"}
                result = sentiment_map.get(prediction[0], "Unknown")

                if result == "Positive":
                    st.success(f"The sentiment is: **{result}**")
                elif result == "Negative":
                    st.error(f"The sentiment is: **{result}**")
                else:
                    st.warning(f"Could not determine sentiment.")
        elif not user_input:
            st.warning("Please enter some text to analyze.")
        else:
            st.warning("Model could not be loaded. Check your file path.")
