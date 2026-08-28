
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.preprocess import load_and_preprocess_image
from src.features import extract_combined_features

DATASET_DIR = os.path.join("dataset", "buffalo")
MODEL_DIR = "models"

def main():
    print("=" * 60)
    print("STEP 1: EXTRACTING FEATURES FROM ALL IMAGES")
    print("=" * 60)

    if not os.path.exists(DATASET_DIR):
        print(f"❌ ERROR: Cannot find directory '{DATASET_DIR}'")
        print(f"Current Working Directory: {os.getcwd()}")
        return

    X = []
    y = []

    breed_folders = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print(f"Found {len(breed_folders)} breed classes to extract features from.\n")

    start_time = time.time()
    total_processed = 0

    for breed in sorted(breed_folders):
        breed_path = os.path.join(DATASET_DIR, breed)
        images = [f for f in os.listdir(breed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        valid_count = 0
        for img_name in images:
            img_path = os.path.join(breed_path, img_name)
            gray, rgb = load_and_preprocess_image(img_path)
            
            if gray is not None and rgb is not None:
                features = extract_combined_features(gray, rgb)
                X.append(features)
                y.append(breed)
                valid_count += 1
                total_processed += 1

        print(f"Extracted features for: {breed:<20} | Processed: {valid_count} images")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"✅ Extraction Complete! {total_processed} total feature vectors in {elapsed:.2f} seconds.")

    X = np.array(X)
    y = np.array(y)

    # 2. Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 3. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    print(f"\nDataset Split: {len(X_train)} Train Samples | {len(X_test)} Test Samples")

    # 4. Feature Scaling
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Define Models
    models = {
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=10.0, probability=True, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, weights='distance')
    }

    print("\n" + "=" * 60)
    print("STEP 2: TRAINING AND EVALUATING ML MODELS")
    print("=" * 60)

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test_scaled)
        pred_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{precision * 100:.2f}%",
            "Recall": f"{recall * 100:.2f}%",
            "F1-Score": f"{f1 * 100:.2f}%",
            "Train Time (s)": round(train_time, 2),
            "Pred Time (s)": round(pred_time, 4)
        })

        trained_models[name] = (model, acc)

    # Display Matrix
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Select Best Model
    best_name = max(trained_models, key=lambda k: trained_models[k][1])
    best_model = trained_models[best_name][0]

    print("\n" + "=" * 60)
    print(f"🏆 BEST MODEL SELECTED: {best_name}")
    print("=" * 60)

    # 6. Save Artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    print(f"✅ Saved trained artifacts into '{MODEL_DIR}/' folder.")

if __name__ == "__main__":
    main()
    