
import os
import uuid
import joblib
import numpy as np
import mysql.connector
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

from src.preprocess import load_and_preprocess_image
from src.features import extract_combined_features

app = Flask(__name__)
app.secret_key = "sih25004_livestock_secret_key"

# Configuration
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Ridhinya@2008',  # Adjust to match your local MySQL password
    'database': 'livestock_db'
}

# Load ML Artifacts
MODEL_DIR = "models"
model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        flash("No file selected!")
        return redirect(url_for("index"))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        flash("Invalid file format. Please upload JPG, PNG, or BMP.")
        return redirect(url_for("index"))

    # Save uploaded image
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(file_path)

    # 1. Preprocess Image
    gray, rgb = load_and_preprocess_image(file_path)
    if gray is None or rgb is None:
        flash("Corrupt or invalid image file!")
        return redirect(url_for("index"))

    # 2. Extract Combined Features (2,286-D)
    features = extract_combined_features(gray, rgb).reshape(1, -1)

    # 3. Scale Features & Predict
    features_scaled = scaler.transform(features)
    pred_encoded = model.predict(features_scaled)[0]
    predicted_breed = label_encoder.inverse_transform([pred_encoded])[0]

    # Calculate Confidence Probability
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = float(np.max(probabilities) * 100)

    # 4. Fetch Breed Info & Vaccination Details from MySQL
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM breed_information WHERE breed_name = %s", (predicted_breed,))
    breed_info = cursor.fetchone()

    cursor.execute("SELECT * FROM vaccination_information WHERE animal_type = 'Buffalo'")
    vaccinations = cursor.fetchall()

    cursor.close()
    conn.close()

    image_url = url_for('static', filename=f"uploads/{unique_name}")

    return render_template(
        "result.html",
        predicted_breed=predicted_breed,
        confidence=f"{confidence:.2f}",
        breed_info=breed_info,
        vaccinations=vaccinations,
        image_path=image_url
    )

@app.route("/register", methods=["POST"])
def register_animal():
    owner_name = request.form.get("owner_name")
    mobile_number = request.form.get("mobile_number")
    village = request.form.get("village")
    predicted_breed = request.form.get("predicted_breed")
    confidence = request.form.get("confidence")
    image_path = request.form.get("image_path")

    # Generate Unique Registration Tag
    tag_id = f"REG-2026-{uuid.uuid4().hex[:6].upper()}"

    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO animal_registration 
    (registration_tag, owner_name, mobile_number, village, animal_type, predicted_breed, prediction_confidence, image_path)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (tag_id, owner_name, mobile_number, village, "Buffalo", predicted_breed, float(confidence), image_path))
    conn.commit()

    cursor.close()
    conn.close()

    return render_template("register.html", tag_id=tag_id, owner_name=owner_name, breed=predicted_breed)

if __name__ == "__main__":
    app.run(debug=True, port=5000)