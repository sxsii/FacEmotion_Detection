from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("face_emotionModel.h5")

# Emotion labels (make sure this matches your model’s output order)
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template("index.html", display_message="No image uploaded!")

    file = request.files['image']
    if file.filename == '':
        return render_template("index.html", display_message="No file selected!")

    # Save temporarily
    img_path = os.path.join("static", file.filename)
    file.save(img_path)

    # Load and preprocess image (grayscale 48x48)
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)
    emotion = LABELS[np.argmax(preds)]
    confidence = round(np.max(preds) * 100, 2)

    message = f"Detected emotion: {emotion} ({confidence}%)"
    return render_template("index.html", emotion=emotion, display_message=message, uploaded_image=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
