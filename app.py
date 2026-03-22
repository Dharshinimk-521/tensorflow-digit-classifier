import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_from_directory
import io
import os

app = Flask(__name__, static_folder='static', template_folder='.')

# Load model once at startup
model = tf.keras.models.load_model("digit_classifier.h5")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read and preprocess image — same logic as original app.py
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('L')

        # Invert if background is bright (same heuristic as before)
        if np.mean(np.array(img)) > 127:
            img = ImageOps.invert(img)

        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        probabilities = prediction[0].tolist()
        confidence = float(probabilities[predicted_class])

        return jsonify({
            'predicted_digit': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)