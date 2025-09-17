# multi_task_atc_pipeline.py

import os
import cv2
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# CONFIGURATION
DATA_PATH                = "data/cow_data.csv"
LON_MODEL_PATH           = "models/longevity_model.pkl"
MILK_MODEL_PATH          = "models/milk_productivity_model.pkl"
REPRO_MODEL_PATH         = "models/reproductive_efficiency_model.pkl"
ELITE_MODEL_PATH         = "models/elite_dam_model.pkl"
CALIBRATION_CM_PER_PIXEL = 0.1

# 1. Load & Clean Dataset
def load_and_clean_dataset(path):
    df = pd.read_csv(path)
    # Impute numeric features with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    return df

# 2. Train Multiple Models
def train_models(df):
    features = [
        "age",
        "body_weight",
        "height_at_withers",
        "body_length",
        "chest_width",
        "parity",
        "historical_milk_yield"
    ]
    X = df[features]

    # Longevity (regression)
    y_lon = df["longevity"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_lon, test_size=0.2, random_state=42
    )
    lon_model = RandomForestRegressor(n_estimators=100, random_state=42)
    lon_model.fit(X_train, y_train)
    print("Longevity R²:", lon_model.score(X_val, y_val))
    joblib.dump(lon_model, LON_MODEL_PATH)

    # Milk Productivity (regression)
    y_milk = df["milk_productivity"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_milk, test_size=0.2, random_state=42
    )
    milk_model = RandomForestRegressor(n_estimators=100, random_state=42)
    milk_model.fit(X_train, y_train)
    print("Milk Productivity R²:", milk_model.score(X_val, y_val))
    joblib.dump(milk_model, MILK_MODEL_PATH)

    # Reproductive Efficiency (regression)
    y_repro = df["reproductive_efficiency"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_repro, test_size=0.2, random_state=42
    )
    repro_model = RandomForestRegressor(n_estimators=100, random_state=42)
    repro_model.fit(X_train, y_train)
    print("Reproductive Efficiency R²:", repro_model.score(X_val, y_val))
    joblib.dump(repro_model, REPRO_MODEL_PATH)

    # Elite Dam Label (classification)
    y_elite = df["elite_dam"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_elite, test_size=0.2, random_state=42
    )
    elite_model = RandomForestClassifier(n_estimators=100, random_state=42)
    elite_model.fit(X_train, y_train)
    print("Elite Dam Accuracy:", elite_model.score(X_val, y_val))
    joblib.dump(elite_model, ELITE_MODEL_PATH)

# 3. Image Utilities
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    fg = cv2.bitwise_and(img, img, mask=mask)
    fg = cv2.resize(fg, (800, 600))
    return fg

def detect_landmarks(img):
    h, w = img.shape[:2]
    return {
        "withers":     (int(w*0.5), int(h*0.3)),
        "rump":        (int(w*0.5), int(h*0.6)),
        "chest_left":  (int(w*0.4), int(h*0.45)),
        "chest_right": (int(w*0.6), int(h*0.45))
    }

def pixel_to_cm(px):
    return px * CALIBRATION_CM_PER_PIXEL

def compute_measurements(kpts):
    w = kpts["withers"]
    r = kpts["rump"]
    cl = kpts["chest_left"]
    cr = kpts["chest_right"]

    # Height at withers: vertical pixel difference
    height_px = abs(r[1] - w[1])
    body_px   = np.hypot(w[0]-r[0], w[1]-r[1])
    chest_px  = np.hypot(cl[0]-cr[0], cl[1]-cr[1])

    return {
        "height_at_withers": pixel_to_cm(height_px),
        "body_length":       pixel_to_cm(body_px),
        "chest_width":       pixel_to_cm(chest_px)
    }

# 4. Feature Extraction
def extract_image_features(img_path):
    img  = load_image(img_path)
    prep = preprocess(img)
    kpts = detect_landmarks(prep)
    meas = compute_measurements(kpts)
    return meas

# 5. Prediction Workflow
def predict(args):
    # Load trained models
    lon_model   = joblib.load(LON_MODEL_PATH)
    milk_model  = joblib.load(MILK_MODEL_PATH)
    repro_model = joblib.load(REPRO_MODEL_PATH)
    elite_model = joblib.load(ELITE_MODEL_PATH)

    # Image-based traits
    img_feats = extract_image_features(args.image)

    # Manually provided traits
    manual_feats = {
        "age":                    args.age,
        "body_weight":            args.body_weight,
        "parity":                 args.parity,
        "historical_milk_yield":  args.historical_milk_yield
    }

    # Combine features into DataFrame
    feats = {**manual_feats, **img_feats}
    df   = pd.DataFrame([feats])

    # Model predictions
    longevity              = lon_model.predict(df)[0]
    milk_productivity      = milk_model.predict(df)[0]
    reproductive_efficiency = repro_model.predict(df)[0]
    elite_dam_label        = elite_model.predict(df)[0]

    # Output results
    print(f"Predicted Longevity (years): {longevity:.2f}")
    print(f"Predicted Milk Productivity (liters): {milk_productivity:.2f}")
    print(f"Predicted Reproductive Efficiency (services/conception): {reproductive_efficiency:.2f}")
    print(f"Predicted Elite Dam Label (0=No, 1=Yes): {elite_dam_label}")

# 6. CLI Interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ATC Multi-task Pipeline")
    parser.add_argument(
        "--mode", choices=["train", "predict"], required=True
    )
    parser.add_argument("--image", help="Path to animal image")
    parser.add_argument("--age", type=float, help="Age in years")
    parser.add_argument("--body_weight", type=float, help="Weight in kg")
    parser.add_argument("--parity", type=int, help="Number of calvings")
    parser.add_argument(
        "--historical_milk_yield", type=float,
        help="Liters per lactation"
    )
    args = parser.parse_args()

    if args.mode == "train":
        df = load_and_clean_dataset(DATA_PATH)
        train_models(df)
    else:
        if not args.image or args.age is None or args.body_weight is None \
           or args.parity is None or args.historical_milk_yield is None:
            raise ValueError(
                "For prediction, provide --image, --age, --body_weight, "
                "--parity, and --historical_milk_yield"
            )
        predict(args)