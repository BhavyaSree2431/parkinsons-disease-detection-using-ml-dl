import os
import tempfile
 
import cv2
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
 
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
 
image_model = load_model(os.path.join(MODELS_DIR, "efficientnet_parkinson_model.keras"))
voice_dnn = load_model(os.path.join(MODELS_DIR, "voice_dnn_model.keras"))
 
xgb = joblib.load(os.path.join(MODELS_DIR, "voice_xgb.pkl"))
rf = joblib.load(os.path.join(MODELS_DIR, "voice_rf.pkl"))
svm = joblib.load(os.path.join(MODELS_DIR, "voice_svm.pkl"))
 
scaler = joblib.load(os.path.join(MODELS_DIR, "voice_scaler.pkl"))
pca = joblib.load(os.path.join(MODELS_DIR, "voice_pca.pkl"))
feature_names = list(joblib.load(os.path.join(MODELS_DIR, "voice_feature_names.pkl")))
 
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
 
 
def contains_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0
 
 
def label_from_prediction(prediction):
    return "Parkinson Disease Detected" if int(prediction) == 1 else "Healthy"
 
 
def prediction_scores_from_probs(probabilities):
    probabilities = np.asarray(probabilities, dtype="float32").flatten()
    if probabilities.size != 2:
        raise ValueError("Expected exactly two class probabilities.")
 
    return {
        "Healthy": float(probabilities[0]),
        "Parkinson Disease Detected": float(probabilities[1]),
    }
 
 
def is_handwriting_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
 
    # Valid handwriting/curve images fall within this edge density range
    return 0.05 < edge_ratio < 0.20
 
 
def _save_uploaded_image(image_input):
    if isinstance(image_input, str):
        return image_input, False
 
    if hasattr(image_input, "seek"):
        image_input.seek(0)
 
    image_bytes = image_input.read() if hasattr(image_input, "read") else None
    if image_bytes is None:
        raise ValueError("Unsupported image input.")
 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(image_bytes)
        return temp_file.name, True
 
 
def _prepare_image_for_inference(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
 
    thresholded = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]
 
    stroke_map = 255 - thresholded
    coordinates = cv2.findNonZero(stroke_map)
 
    if coordinates is not None:
        x, y, w, h = cv2.boundingRect(coordinates)
        padding = 20
        y1 = max(y - padding, 0)
        y2 = min(y + h + padding, image_bgr.shape[0])
        x1 = max(x - padding, 0)
        x2 = min(x + w + padding, image_bgr.shape[1])
        thresholded = thresholded[y1:y2, x1:x2]
 
    prepared = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
    prepared = cv2.resize(prepared, (192, 192))
    return prepared.astype("float32")
 
 
def predict_image(image_input):
    return int(np.argmax(predict_image_proba(image_input)))
 
 
def predict_image_proba(image_input):
    image_path, should_cleanup = _save_uploaded_image(image_input)
 
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Uploaded image could not be read.")
 
        # ✅ FIX: Check for face first (clearest error message)
        if contains_face(img):
            raise ValueError(
                "Face detected! Please upload only handwriting or spiral/wave curve images."
            )
 
        # ✅ FIX: Then check if it's a valid handwriting/curve image
        if not is_handwriting_image(img):
            raise ValueError(
                "Invalid image! Please upload only handwriting or spiral/wave curve images."
            )
 
        img = _prepare_image_for_inference(img)
        img = np.expand_dims(img, axis=0)
 
        prediction = image_model.predict(img, verbose=0)[0]
        return prediction
 
    finally:
        if should_cleanup and os.path.exists(image_path):
            os.remove(image_path)
 
 
def _prepare_voice_features(voice_input):
    if isinstance(voice_input, pd.DataFrame):
        voice_df = voice_input.copy()
    elif isinstance(voice_input, pd.Series):
        voice_df = voice_input.to_frame().T
    elif isinstance(voice_input, dict):
        voice_df = pd.DataFrame([voice_input])
    elif hasattr(voice_input, "read"):
        if hasattr(voice_input, "seek"):
            voice_input.seek(0)
        voice_df = pd.read_csv(voice_input)
    else:
        voice_array = np.asarray(voice_input).flatten()
        voice_df = pd.DataFrame([voice_array], columns=feature_names[: len(voice_array)])
 
    if "id" in voice_df.columns:
        voice_df = voice_df.drop(columns=["id"])
    if "class" in voice_df.columns:
        voice_df = voice_df.drop(columns=["class"])
 
    missing_columns = [column for column in feature_names if column not in voice_df.columns]
    if missing_columns:
        raise ValueError(
            f"Voice CSV is missing {len(missing_columns)} required columns. "
            "Please upload the model-compatible speech feature CSV."
        )
 
    voice_df = voice_df[feature_names]
    voice_df = voice_df.apply(pd.to_numeric, errors="coerce")
 
    if voice_df.isnull().any().any():
        raise ValueError("Voice input contains empty or non-numeric feature values.")
 
    return voice_df.iloc[[0]]
 
 
def predict_voice(voice_input):
    return int(predict_voice_proba(voice_input) >= 0.5)
 
 
def predict_voice_proba(voice_input):
    features = _prepare_voice_features(voice_input)
    features = scaler.transform(features)
    features = pca.transform(features)
 
    dnn_prob = float(voice_dnn.predict(features, verbose=0)[0][0])
    xgb_prob = float(xgb.predict_proba(features)[0][1])
    rf_prob = float(rf.predict_proba(features)[0][1])
    svm_prob = float(svm.predict_proba(features)[0][1])
 
    score = (
        0.4 * dnn_prob +
        0.2 * xgb_prob +
        0.2 * rf_prob +
        0.2 * svm_prob
    )
 
    return float(score)
 
 
def final_prediction(image_prediction, voice_prediction):
    final = int((int(image_prediction) + int(voice_prediction)) >= 1)
    return label_from_prediction(final)
 
 
def final_prediction_proba(image_input, voice_input, image_weight=0.5, voice_weight=0.5):
    image_prob = float(predict_image_proba(image_input)[1])
    voice_prob = float(predict_voice_proba(voice_input))
    return (image_weight * image_prob) + (voice_weight * voice_prob)
 
 
def fusion_prediction(image_input, voice_input):
    image_result = predict_image(image_input)
    voice_result = predict_voice(voice_input)
    fusion_prob = final_prediction_proba(image_input, voice_input)
 
    print("\nImage Prediction :", image_result)
    print("Voice Prediction :", voice_result)
    print("Fusion Probability :", round(fusion_prob, 4))
 
    diagnosis = label_from_prediction(int(fusion_prob >= 0.5))
 
    print("Final Diagnosis :", diagnosis)
    return diagnosis
 
 
if __name__ == "__main__":
    image_path = "../test_image.png"
    voice_features = np.random.rand(len(feature_names))
    fusion_prediction(image_path, voice_features)