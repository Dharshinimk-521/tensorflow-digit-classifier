
# Handwritten Digit Classifier

A simple Convolutional Neural Network (CNN) built with TensorFlow to classify handwritten digits (MNIST dataset).

---

## Project Structure

- `app.py` → Streamlit app to run predictions using the pre-trained model.
- `train_model.py` → Script to train the CNN model from scratch and save it to `saved_model/`.
- `saved_model/digit_classifier.keras` → Pre-trained model included for convenience.
- `requirements.txt` → Python dependencies.
- `.gitignore` → Ignore unnecessary files like `venv/`.

---

## How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
````

2. **Run the Streamlit app**:

```bash
streamlit run app.py
```

3. **To retrain the model from scratch**:

```bash
python train_model.py
```

---

## Model Details

* Architecture:

  * Conv2D + MaxPooling
  * Flatten + Dense layers with ReLU
  * Softmax output for probabilities
* Loss: `sparse_categorical_crossentropy`
* Optimizer: `Adam`
* Normalized input (0-1) for better training
* Pre-trained model included to use without retraining

---

## Notes

* The `saved_model/` folder contains the pre-trained model for easy testing.
* Feel free to retrain the model or modify the architecture for experimentation.

````

---

## ✅ **5️⃣ Steps to Upload to GitHub**

```bash
git init
git add .
git commit -m "Initial commit: app, training script, and pre-trained model"
git branch -M main
git remote add origin https://github.com/yourusername/handwritten-digit-classifier.git
git push -u origin main
````

---
