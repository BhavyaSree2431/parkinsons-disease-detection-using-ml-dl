import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


def train_voice_model():

    print("Loading dataset...")

    data = pd.read_csv("../dataset/voice/pd_speech_features.csv")

    print("Dataset shape:", data.shape)

    # -----------------------
    # Split features & label
    # -----------------------

    y = data["class"]
    X = data.drop(["id", "class"], axis=1)

    # Save feature names for inference
    feature_names = X.columns

    # -----------------------
    # Feature Scaling
    # -----------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------
    # PCA   
    # -----------------------

    pca = PCA(n_components=80)
    X_pca = pca.fit_transform(X_scaled)

    print("Reduced feature size:", X_pca.shape)

    # -----------------------
    # SMOTE
    # -----------------------

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_pca, y)

    print("Balanced dataset:", X_balanced.shape)

    # -----------------------
    # Train test split
    # -----------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced,
        y_balanced,
        test_size=0.2,
        stratify=y_balanced,
        random_state=42
    )

    # ===============================
    # 1️⃣ Deep Neural Network
    # ===============================

    dnn = Sequential([
        Input(shape=(X_train.shape[1],)),

        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    dnn.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("Training Neural Network...")

    dnn.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    dnn_pred = (dnn.predict(X_test) > 0.5).astype(int).flatten()

    # ===============================
    # 2️⃣ XGBoost
    # ===============================

    print("Training XGBoost...")

    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)

    # ===============================
    # 3️⃣ Random Forest
    # ===============================

    print("Training Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42
    )

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    # ===============================
    # 4️⃣ SVM
    # ===============================

    print("Training SVM...")

    svm = SVC(kernel="rbf", probability=True)

    svm.fit(X_train, y_train)

    svm_pred = svm.predict(X_test)

    # ===============================
    # Weighted Ensemble
    # ===============================

    final_score = (
        0.4 * dnn_pred +
        0.2 * xgb_pred +
        0.2 * rf_pred +
        0.2 * svm_pred
    )

    final_pred = (final_score >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, final_pred)

    print("\nFinal Ensemble Accuracy:", accuracy)

    print("\nClassification Report\n")

    print(classification_report(y_test, final_pred))

    # -----------------------
    # Save models
    # -----------------------

    os.makedirs("models", exist_ok=True)

    dnn.save("models/voice_dnn_model.keras")

    joblib.dump(xgb, "models/voice_xgb.pkl")
    joblib.dump(rf, "models/voice_rf.pkl")
    joblib.dump(svm, "models/voice_svm.pkl")

    joblib.dump(scaler, "models/voice_scaler.pkl")
    joblib.dump(pca, "models/voice_pca.pkl")

    # ⭐ Save feature names (fix warning permanently)
    joblib.dump(feature_names, "models/voice_feature_names.pkl")

    print("\nAll models saved successfully.")


if __name__ == "__main__":
    train_voice_model()