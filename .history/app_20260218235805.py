import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("digit_classifier.keras")

st.title("Handwritten Digit Classifier")
#Create a file uploader in the web app to upload images (only png, jpg, jpeg)
uploaded_file = st.file_uploader("Upload a 28x28 digit image", type=["png","jpg","jpeg"])
# Check if the user has uploaded a file
if uploaded_file is not None:
    # Open the uploaded image and convert it to grayscale ('L' mode
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28,28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,28,28)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.image(img, caption="Uploaded Image", width=150)
    st.write("Prediction:", predicted_class)
