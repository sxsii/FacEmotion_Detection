# app.py (cleaned, full file)
import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, flash
import numpy as np
import cv2
import json

from tensorflow.keras.models import load_model

# Config
UPLOAD_FOLDER = "uploads"
DB_PATH = "database.db"
MODEL_PATH = "face_emotionModel.h5"
CLASS_NAMES_PATH = "class_names.json"
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "replace-with-a-secure-random-string"

# Load model
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)
else:
    print("Warning: model file not found:", MODEL_PATH)

# Load class names and build messages
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    print("Loaded class names:", class_names)
else:
    class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    print("class_names.json not found — using default canonical list:", class_names)

CANONICAL_MESSAGES = {
    'angry': ("Angry", "You look angry. Take a breath — what's bothering you?"),
    'disgust': ("Disgust", "You look displeased. Everything okay?"),
    'fear': ("Fear", "You seem afraid. Want to share what's worrying you?"),
    'happy': ("Happy", "You're smiling! That's great — keep it up!"),
    'sad': ("Sad", "You are frowning. Why are you sad?"),
    'surprise': ("Surprise", "You look surprised! What happened?"),
    'neutral': ("Neutral", "You look neutral. How are you feeling today?")
}

def _message_for_classname(name):
    key = name.lower()
    if key in CANONICAL_MESSAGES:
        return CANONICAL_MESSAGES[key]
    label = name.replace("_", " ").title()
    return (label, f"You look {label.lower()}. How are you feeling?")

EMOTIONS = {idx: _message_for_classname(cname) for idx, cname in enumerate(class_names)}

# DB init
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        extra TEXT,
        emotion_label INTEGER,
        emotion_text TEXT,
        image_filename TEXT,
        image_blob BLOB,
        created_at TEXT
    )
    ''')
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def detect_and_prepare_face(image_path, target_size=(48,48)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read uploaded image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
        face = gray[y:y+h, x:x+w]
    else:
        face = gray
    face = cv2.resize(face, target_size)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form.get('name','').strip()
        email = request.form.get('email','').strip()
        extra = request.form.get('extra','').strip()
        file = request.files.get('image')

        if not name or not email:
            flash("Please provide your name and email.")
            return redirect(request.url)
        if not file or file.filename == '':
            flash("Please upload a picture.")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Unsupported file type. Use png/jpg/jpeg/bmp.")
            return redirect(request.url)

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            face_arr = detect_and_prepare_face(filepath)
        except Exception as e:
            flash("Error processing image: " + str(e))
            return redirect(request.url)

        if model is None:
            flash("Model not available. Place face_emotionModel.h5 in project root.")
            return redirect(request.url)

        preds = model.predict(face_arr)
        # debug block - paste right after preds = model.predict(face_arr)
        import numpy as _np
        _probs = _np.squeeze(preds)  # shape (num_classes,)
        top3_idx = _np.argsort(_probs)[-3:][::-1]
        print("DEBUG: raw probs:", _probs.tolist())
        print("DEBUG: top3 indices:", top3_idx.tolist())
        print("DEBUG: top3 (class_name, prob):", [(class_names[i], float(_probs[i])) for i in top3_idx])

        label_idx = int(np.argmax(preds))
        emotion_name, emotion_message = EMOTIONS.get(label_idx, ("Unknown", "Couldn't detect emotion."))

        with open(filepath, 'rb') as f:
            blob = f.read()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO users (name, email, extra, emotion_label, emotion_text, image_filename, image_blob, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, email, extra, label_idx, emotion_name, filename, blob, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

        flash(f"Detected emotion: {emotion_name}")
        return render_template("index.html", result_message=emotion_message, detected=emotion_name, name=name)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
