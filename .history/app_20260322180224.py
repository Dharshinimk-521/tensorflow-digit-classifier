import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image,ImageOps

# Load model
model = load_model("digit_classifier.keras")

st.title("Handwritten Digit Classifier")
#Create a file uploader in the web app to upload images (only png, jpg, jpeg)
uploaded_file = st.file_uploader("Upload a 28x28 digit image(any background or color)", type=["png","jpg","jpeg"])
# Check if the user has uploaded a file
if uploaded_file is not None:
    # Open the uploaded image and convert it to grayscale ('L' mode)
    img = Image.open(uploaded_file)
    img = img.convert('L')
    if np.mean(np.array(img)) > 127:  # simple heuristic: if background is bright
        img = ImageOps.invert(img)
    img = img.resize((28,28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,28,28)
    #Reshape the array to match model input shape: (batch_size, height, width)
    # Here batch_size is 1 since we are predicting a single image
    prediction = model.predict(img_array)
    # Get the index of the highest probability from the prediction
    # This index corresponds to the predicted digit (0-9)
    predicted_class = np.argmax(prediction)

    st.image(img, caption="Uploaded Image", width=300)
    #Display the predicted digit
    st.write("")
    st.write("Prediction: ", predicted_class)
