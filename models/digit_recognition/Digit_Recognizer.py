import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import keras
from streamlit_drawable_canvas import st_canvas

def digitRecognizer():
   
    #Load the model
    model = keras.saving.load_model("models/digit_recognition/best_model.keras")
    st.markdown("<h3>Draw a Digit Here</h3>",unsafe_allow_html=True)
    # --- User Interface for Drawing (Centered) ---
    col_empty1, col_canvas, col_empty2 = st.columns([1.3, 1, 1])
    with col_canvas:
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict", use_container_width=True):
        if canvas_result.image_data is not None:
            # Get the image data from the canvas as a NumPy array
            image_data = canvas_result.image_data

            # Convert the RGBA image data to a grayscale image using OpenCV
            gray_img = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

            # Resize the image to 28x28 pixels
            resized_img = cv2.resize(gray_img, (28, 28), interpolation=cv2.INTER_AREA)

            # Invert the colors
            inverted_img = cv2.bitwise_not(resized_img)
            
            # Normalize the pixel values to be between 0 and 1
            normalized_img = inverted_img / 255.0

            # Reshape the image to a 4D tensor: (1, 28, 28, 1)
            img_tensor = normalized_img.reshape(1, 28, 28, 1)

            # Make the prediction
            prediction = model.predict(img_tensor)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)

            # Display the result
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Prediction Result")
            st.markdown(
                f"<div class='prediction-box'><h3>Predicted Digit: <strong>{predicted_digit}</strong></h3><p>Confidence: {confidence:.2f}</p></div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("Please draw a digit on the canvas first!")

if __name__ == "__main__":
    digitRecognizer()
