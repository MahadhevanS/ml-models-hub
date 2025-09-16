# Import necessary libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO

#Image size on which the model is trained
IMG_SIZE = 32

#Different objects on which the model is trained
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the model
model = load_model('models/Object_Classifier/CIFAR_object_classifier.keras')

# Image preprocessing
def preprocess_image(image_input):
    try:
        img = Image.open(image_input).convert('RGB')

        # Resize the image to the required input size (32x32)
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # Convert the image to a NumPy array
        img_array = np.array(img)

        # Normalize the pixel values to the range [0, 1]
        img_array = img_array / 255.0

        # Add an extra dimension to represent the batch size (which is 1 for a single image)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def objectClassifier():

    with st.expander("Available objects"):
        object_html = "<div style='height:200px; overflow-y:auto; white-space:pre-line; font-family:monospace;'>"
        object_html += "<br>".join(sorted(class_names))  # one word per line
        object_html += "</div>"
        st.markdown(object_html, unsafe_allow_html=True)
    st.write("---")

    #File Upload
    image_to_predict = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])


    # Display the image and make a prediction
    if image_to_predict is not None:
        # Display the uploaded image
            # st.image(image_to_predict, caption='Image to be classified', use_container_width=True)
        st.write("")
        
        if model:
            
            # Preprocess the image for the model
            processed_image = preprocess_image(image_to_predict)

            if processed_image is not None:
                # Make a prediction
                with st.spinner('Classifying...'):
                    predictions = model.predict(processed_image)
                
                # Get the predicted class and confidence
                predicted_class_index = np.argmax(predictions[0])
                predicted_class_name = class_names[predicted_class_index]
                confidence = predictions[0][predicted_class_index] * 100

                st.metric(label="Predicted Class", value=predicted_class_name)
                st.metric(label="Confidence", value=f"{confidence:.2f}%")

