import os
import uuid
import base64
import joblib
import numpy as np
import mysql.connector
import re
from flask import Flask, request, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

from src.preprocess import load_and_preprocess_image
from src.features import extract_combined_features
from translations import LANGUAGES   # <-- Translation dictionary import

app = Flask(__name__)               
app.secret_key = 'your_secret_key'  # Session work aaguraathukku

# Language set route
@app.route("/set_lang/<lang_code>")
def set_lang(lang_code):
    if lang_code in LANGUAGES:
        session['lang'] = lang_code
    return redirect(url_for('index'))

# Configuration
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'jfif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'poornima@2008', 
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
    lang = session.get('lang', 'en') # Default English
    t = LANGUAGES.get(lang, LANGUAGES['en'])
    return render_template("index.html", t=t, current_lang=lang)

# 1. Prediction Route via File Upload
@app.route("/predict", methods=["POST"])
def predict():
    file = None
    for key in ['file', 'image', 'animal_image', 'upload']:
        if key in request.files and request.files[key].filename != '':
            file = request.files[key]
            break
            
    if not file:
        flash("No file selected!")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file format. Please upload JPG, PNG, WEBP, or JFIF.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(file_path)

    return process_prediction(file_path, unique_name)

# 2. Prediction Route via Live Camera Capture
@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    image_data = request.form.get("image_data")
    if not image_data:
        flash("No image captured from camera!")
        return redirect(url_for("index"))
    
    try:
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        
        unique_name = f"{uuid.uuid4().hex[:8]}_cam.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        
        with open(file_path, "wb") as f:
            f.write(binary_data)
            
        return process_prediction(file_path, unique_name)
    except Exception as e:
        print(f"Camera predict error: {e}")
        flash(f"Camera error: {e}")
        return redirect(url_for("index"))

def process_prediction(file_path, unique_name):
    try:
        gray, rgb = load_and_preprocess_image(file_path)
        if gray is None or rgb is None:
            flash("Corrupt or invalid image file!")
            return redirect(url_for("index"))

        features = extract_combined_features(gray, rgb).reshape(1, -1)
        features_scaled = scaler.transform(features)
        pred_encoded = model.predict(features_scaled)[0]
        predicted_breed = label_encoder.inverse_transform([pred_encoded])[0]

        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities) * 100)

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM breed_information WHERE breed_name = %s", (predicted_breed,))
        breed_info = cursor.fetchone()

        cursor.execute("SELECT * FROM vaccination_information WHERE animal_type = 'Buffalo'")
        vaccinations = cursor.fetchall()

        cursor.execute("SELECT * FROM medical_history WHERE breed_name = %s ORDER BY updated_at DESC", (predicted_breed,))
        medical_history = cursor.fetchall()

        cursor.close()
        conn.close()

        image_url = url_for('static', filename=f"uploads/{unique_name}")

        # Language selection
        lang = session.get('lang', 'en')
        t = LANGUAGES.get(lang, LANGUAGES['en'])

        # Instant local dictionary mapping for fast loading without lag!
        if breed_info:
            region = breed_info.get('native_region', '')
            color = breed_info.get('coat_colour', '')
            horn = breed_info.get('horn_type', '')

            breed_info['native_region'] = t.get('regions', {}).get(region, region)
            breed_info['coat_colour'] = t.get('colors', {}).get(color, color)
            breed_info['horn_type'] = t.get('horns', {}).get(horn, horn)

        return render_template(
            "result.html",
            predicted_breed=predicted_breed,
            confidence=f"{confidence:.2f}",
            breed_info=breed_info,
            vaccinations=vaccinations,
            medical_history=medical_history,
            image_path=image_url,
            t=t
        )

    except Exception as e:
        print(f"--> FULL ERROR: {e}")
        raise e

# 3. Animal Registration Route
@app.route("/register", methods=["POST"])
def register_animal():
    owner_name = request.form.get("owner_name")
    mobile_number = request.form.get("mobile_number")
    village = request.form.get("village")
    predicted_breed = request.form.get("predicted_breed")
    confidence = request.form.get("confidence")
    image_path = request.form.get("image_path")
    
    custom_tag_id = request.form.get("custom_tag_id", "").strip().upper()
    if custom_tag_id:
        custom_tag_id = re.sub(r'[^A-Z0-9-_]', '', custom_tag_id)
        tag_id = custom_tag_id
    else:
        tag_id = f"REG-2026-{uuid.uuid4().hex[:6].upper()}"

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        query = """
        INSERT INTO animal_registration 
        (registration_tag, owner_name, mobile_number, village, animal_type, predicted_breed, prediction_confidence, image_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (tag_id, owner_name, mobile_number, village, "Buffalo", predicted_breed, float(confidence), image_path))
        conn.commit()
    except Exception as e:
        print(f"--> Registration Error (Duplicate ID): {e}")
        flash("Error: This ID is already taken! Please choose a different ID.")
        return redirect(url_for("index"))
    finally:
        cursor.close()
        conn.close()

    return render_template("register.html", tag_id=tag_id, owner_name=owner_name, breed=predicted_breed)

# 4. Doctor Login Route
@app.route("/doctor_login", methods=["GET", "POST"])
def doctor_login():
    if request.method == "POST":
        doctor_name = request.form.get("doctor_name", "").strip()
        doctor_phone = request.form.get("doctor_phone", "").strip()
        password = request.form.get("password", "").strip()
        
        if doctor_name and doctor_phone and password:
            session['doctor_name'] = doctor_name
            session['doctor_phone'] = doctor_phone
            return redirect(url_for("doctor_dashboard"))
        else:
            flash("Please fill all fields to login!")
            return redirect(url_for("doctor_login"))
            
    return render_template("doctor_login.html")

# 6. Farmer History / Dashboard Route
@app.route("/farmer_history", methods=["GET", "POST"])
def farmer_history():
    registered_animals = []
    mobile_number = None
    
    if request.method == "POST":
        mobile_number = request.form.get("mobile_number", "").strip()
        session['farmer_mobile'] = mobile_number
    else:
        mobile_number = session.get('farmer_mobile')
        
    if mobile_number:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM animal_registration WHERE mobile_number = %s", (mobile_number,))
        registered_animals = cursor.fetchall()
        cursor.close()
        conn.close()
        
    return render_template("farmer_history.html", registered_animals=registered_animals, mobile_number=mobile_number)

# 7. Specific Breed Medical History View for Farmer
@app.route("/breed_medical_history/<breed_name>", methods=["GET"])
def breed_medical_history(breed_name):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT * FROM breed_information WHERE breed_name = %s", (breed_name,))
    breed_info = cursor.fetchone()
    
    cursor.execute("SELECT * FROM medical_history WHERE breed_name = %s ORDER BY updated_at DESC", (breed_name,))
    medical_history = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return render_template("farmer_breed_history.html", breed_name=breed_name, breed_info=breed_info, medical_history=medical_history)

# 5. Doctor Dashboard Route
@app.route("/doctor_dashboard", methods=["GET", "POST"])
def doctor_dashboard():
    if 'doctor_name' not in session:
        return redirect(url_for("doctor_login"))
        
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    animal_details = None
    search_tag = request.args.get("search_tag")

    if search_tag:
        cursor.execute("SELECT * FROM animal_registration WHERE registration_tag = %s", (search_tag.strip(),))
        animal_details = cursor.fetchone()
        if not animal_details:
            flash("No animal found with this Registration Tag ID!")

    if request.method == "POST":
        breed_name = request.form.get("breed_name")
        medical_notes = request.form.get("medical_notes")
        current_doctor = session.get('doctor_name')
        current_phone = session.get('doctor_phone')
        
        if breed_name and medical_notes:
            history_query = """
                INSERT INTO medical_history (breed_name, doctor_name, doctor_phone, medical_notes)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(history_query, (breed_name, current_doctor, current_phone, medical_notes))
            conn.commit()
            flash(f"Medical notes successfully added by Dr. {current_doctor}!")

    cursor.execute("SELECT breed_name, native_region, medical_notes, doctor_name, doctor_phone FROM breed_information")
    breeds = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template("doctor_dashboard.html", breeds=breeds, animal_details=animal_details, search_tag=search_tag, current_doctor=session.get('doctor_name'))

if __name__ == "__main__":
    app.run(debug=True, port=5000)