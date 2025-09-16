import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models

#Vocabulary - The words that the chatbot can understand.
vocab = ['Mary', 'grabbed', 'apple', 'left', 'took', 'in', 'bathroom', 'put', 'no', 'there', 'yes', 'Daniel', 'travelled', 'dropped', 'got', '?', 'the', 'Sandra', 'hallway', 'picked', 'discarded', 'back', 'kitchen', 'milk', 'office', 'down', 'Is', 'moved', '.', 'garden', 'went', 'to', 'journeyed', 'John', 'up', 'bedroom', 'football']
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)
model = models.load_model("models/Story_telling_and_answering_chatbot/Project_Chatbot_model.keras")

def check_vocab(data,word_index=tokenizer.word_index):
    unknown_words = set()
    for story,ques,ans in data:
        for word in story:
            if word.lower() not in word_index:
                unknown_words.add(word)
        for word in ques:
            if word.lower() not in word_index:
                unknown_words.add(word)
    return unknown_words
# Mock vectorize_stories function. Replace this with your actual function.
def vectorize_stories(data,max_story_len=156,max_ques_len=6,word_index=tokenizer.word_index):
    X=[]
    Xq=[]
    Y=[]
    for story,question,answer in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in question]
        y = np.zeros(len(word_index)+1)
        y[word_index[answer]]=1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X,maxlen=max_story_len),pad_sequences(Xq,maxlen=max_ques_len),np.array(Y))

def Chatbot():
    # --- STREAMLIT APP LAYOUT ---

    st.set_page_config(layout="wide")
    st.markdown("---")

    with st.expander("📖 Supported Vocabulary"):
        vocab_html = "<div style='height:200px; overflow-y:auto; white-space:pre-line; font-family:monospace;'>"
        vocab_html += "<br>".join(sorted(vocab))  # one word per line
        vocab_html += "</div>"
        st.markdown(vocab_html, unsafe_allow_html=True)


    # Use a container to group the input fields and center them
    story , ques = st.columns([2,1])
    with story:
        st.subheader("Your Story")
        story = st.text_area(
            "Paste your story here:",
            height=120,
            placeholder="e.g., Mary went to the hallway . Sandra in bedroom moved there ."
        )
    with ques:
        st.subheader("Your YES/NO type Question")
        my_ques = st.text_input(
            "Enter your question:",
            placeholder="e.g., Is Sandra in bedroom ?"
        )

    #Mock answer
    ans = "yes"

    # Button to trigger the prediction
    if st.button("Predict Answer", use_container_width=True):
        if not story or not my_ques:
            st.warning("Please enter a story and a question to get a prediction.")
        else:
            # --- Prediction Logic from your code, adapted for Streamlit ---
            
            with st.spinner("Processing..."):
                
                data = [(story.split(), my_ques.split(), ans.lower())]
                unknown_words = check_vocab(data)
                if unknown_words:
                    st.error(f"⚠️ The following words are not in the model's vocabulary: {', '.join(unknown_words)}")
                    st.info("Please rephrase your input using only the supported vocabulary (see the list above).")

                # vectorize_story
                else:
                    story_vec, ques_vec, answer_vec = vectorize_stories(data)
                    story_vec, ques_vec, answer_vec = vectorize_stories(data)
            
                    pred_result = model.predict([story_vec, ques_vec])
                    val_max = np.argmax(pred_result)
                    predicted_word = tokenizer.index_word.get(val_max, "Unknown word").upper()
                    st.markdown(f"<h1 style='text-align:center;color:green;'>{predicted_word}</h1>", unsafe_allow_html=True)
                    