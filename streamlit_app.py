import streamlit as st
import base64
import nltk
import os

nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download the 'stopwords' corpus if it's not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)
    
from models.cardio.Cardio_vascular import cardio_vascular
from models.iris_classifier.iris_classifier import Iris_Classifier
from models.Story_telling_and_answering_chatbot.Comprehenshion_chatbot import Chatbot
from models.digit_recognition.Digit_Recognizer import digitRecognizer
from models.Object_Classifier.object_classifier import objectClassifier
from models.house_price_predictor.House_price_predictor import house_price_predictor
from models.Fraudlent_prediction.Fraudlent_predictor import fraud_predictor
from models.sentiment_analysis.sentiment_analyser import sentiment_analyser
from models.chat_summarizer.Chat_Summarizer import text_summarizer as chat_summarizer

st.set_page_config(layout='wide')

query_params = st.query_params
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = query_params.get('page', 'home')
if query_params.get('page') and st.session_state['current_page'] != query_params['page']:
    st.session_state['current_page'] = query_params['page']

st.markdown("""
<style>
/* Global Styles */
body {
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    background: linear-gradient(135deg, #001f3f 0%, #004d99 100%);
    color: #f0f0f0; 
}

/* Main container for the app's main content */
.main {
    background: transparent;
}

/* Header and Text Styles */
h1, h2, h3, h4, h5, h6 {
    color: #FFD700;
    text-align: center;
}
h1 { font-size: 48px; font-weight: 700; }
h4 { font-size: 20px; color: #b0b0b0; }

/* Card Styling */
.card {
    border: 1px solid #3d4a5c;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease;
    text-align: center;
    margin-bottom: 20px;
    height: 350px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start; /* Aligns content to the top */
    background: linear-gradient(135deg, rgba(13, 17, 31, 0.9), rgba(25, 31, 47, 0.9));
}
.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.5);
    background: linear-gradient(135deg, rgba(25, 31, 47, 0.95), rgba(30, 40, 78, 0.95));
}
.card-title {
    font-weight: 600;
    color: #f0f0f0;
    font-size: 24px;
    margin-top: 18px;
    height: 60px; 
    display: flex;
    align-items: center;
    justify-content: center;
    
}
.card-description {
    font-size: 15px;
    color: #b0b0b0;
    flex-grow: 1; /* Allows the description to take up available space */
    margin-top: 10px;
}
.card-image {
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-radius: 8px;
    border: 1px solid #3d4a5c;
}

/* Primary Button Styling - Blue Theme like Card */
.stButton > button {
    background: linear-gradient(135deg, rgba(13, 17, 31, 0.9), rgba(25, 31, 47, 0.9));
    color: #f0f0f0;
    padding: 12px 28px;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    font-size: 16px;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.3s ease, background-color 0.3s ease;
    box-shadow: 0 4px 8px rgba(0,0,0,0.5); /* Matching card shadow */
}

.stButton > button:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.6);
    background: linear-gradient(135deg, rgba(25, 31, 47, 0.95), rgba(30, 40, 78, 0.95));
}

/* Secondary Button Styling - Lighter Blue Variant */
.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, rgba(25, 31, 47, 0.9), rgba(35, 45, 75, 0.9));
    color: #f0f0f0;
    border: none;
    box-shadow: 0 4px 8px rgba(0,0,0,0.4);
}

.stButton > button[kind="secondary"]:hover {
    background: linear-gradient(135deg, rgba(30, 40, 78, 0.95), rgba(40, 50, 90, 0.95));
    color: #fff;
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.5);
}
</style>""", unsafe_allow_html=True)

def home_page():
    st.title("⚙️ Welcome to my ML Model Hub!")
    st.markdown("<h4>Explore my models by clicking on a card below.</h4>", unsafe_allow_html=True)
    
    models = [
        {"title": "chat summarizer", "image": "images/summarizer.jpeg", "description": "Summarizes chat conversations using advanced NLP techniques.", "page": "Chat_Summarizer"},
        {"title": "Sentiment analysis", "image": "images/sentiment.jpeg", "description": "Sentiment analysis using Sentiment140 dataset", "page": "Tweet_sentiment_predict"},
        {"title": "Comprehensive Chatbot", "image": "images/chatbot.jpeg", "description": "Answers to the Yes/No questions based on user story inputs.", "page": "yes/no"},
        {"title": "Cardiovascular Predictor", "image": "images/cardio_vascular.jpeg", "description": "Predicts the risk of heart disease based on various health metrics.", "page": "cardio"},
        {"title": "Iris Species Classifier", "image": "images/iris.jpeg", "description": "Classifies iris flowers into species based on their measurements.", "page": "iris"},
        {"title": "Digit Recognizer", "image": "images/digit.png", "description": "Predicts handwritten digits", "page": "digit_recognition"},
        {"title": "Object Classifier", "image": "images/object.jpeg", "description": "Predicts the object in a given image", "page": "object_classify"},
        {"title": "House Price Predictor", "image": "images/house.jpeg", "description": "Predicts the price of house based on key features", "page": "house_price_predict"},
        {"title": "Fraud Transaction Predictor", "image": "images/transaction.jpeg", "description": "Predicts whether a transaction has chances to be a fraud transaction using some key features", "page": "transaction_fraud_predict"}
        
        
    ]

    for i in range(0, len(models), 4):
        cols = st.columns(4)
        for j, model in enumerate(models[i:i + 4]):
            with cols[j]:
                try:
                    with open(model["image"], "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode()
                except FileNotFoundError:
                    st.error(f"Image not found: {model['image']}")
                    encoded_image = ""
                
                # Card HTML with image and link
                st.markdown(
                    f"""
                    <a href="?page={model['page']}" target="_self" style="text-decoration: none; color: inherit;">
                        <div class="card">
                            <img src="data:image/png;base64,{encoded_image}" class="card-image">
                            <h4 class="card-title">{model['title']}</h4>
                            <p class="card-description">{model['description']}</p>
                        </div>
                    </a>
                    """, unsafe_allow_html=True
                )

def model_page(title, github, func):
    button_col, title_col = st.columns([0.03, 0.99])
    
    with title_col:
        st.title(title)
        st.markdown(f"""
            <div style="text-align: center;">
                <a href="{github}" class="github-link" target="_blank">
                    Click to view code
                </a>
            </div>
            """, unsafe_allow_html=True)
    
    with button_col:
        if st.button("↩", key="back_button"):
            st.query_params["page"] = "home"
            st.rerun()

    func()

# --- Main app logic to display the correct page ---
page_funcs = {
    'home': home_page,
    'cardio': lambda: model_page("🫀 Heart Failure Prediction App", "https://github.com/MahadhevanS/Heart-Failure-predictor", cardio_vascular),
    'iris': lambda: model_page("🌸 Iris Flower Classifier", "https://github.com/MahadhevanS/Heart-Failure-predictor", Iris_Classifier),
    'yes/no': lambda: model_page("🤖 Simple Q&A YES/NO Chatbot", "https://github.com/MahadhevanS/Heart-Failure-predictor", Chatbot),
    'digit_recognition': lambda: model_page("1️⃣ Digit Recognizer", "https://github.com/MahadhevanS/Heart-Failure-predictor", digitRecognizer),
    'object_classify': lambda: model_page("🖼️ CIFAR10 Image Classifier", "https://github.com/MahadhevanS/Heart-Failure-predictor", objectClassifier),
    'house_price_predict': lambda: model_page("🏠 House Price Predictor", "https://github.com/MahadhevanS/Heart-Failure-predictor", house_price_predictor),
    'transaction_fraud_predict': lambda: model_page("🏠 Transaction Fraudlent Predictor", "https://github.com/MahadhevanS/Heart-Failure-predictor", fraud_predictor),
    'Tweet_sentiment_predict': lambda: model_page("🐦 Tweet Sentiment Analyzer", "https://github.com/MahadhevanS/Heart-Failure-predictor", sentiment_analyser),
    'Chat_Summarizer': lambda: model_page("📝 Chat Summarizer", "https://github.com/MahadhevanS/Chat_Summarizer", chat_summarizer)
}

if st.session_state['current_page'] in page_funcs:
    page_funcs[st.session_state['current_page']]()
else:
    home_page()
