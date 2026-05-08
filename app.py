import io

import pandas as pd
import streamlit as st
from PIL import Image

from src.fusion_model import (
    final_prediction,
    label_from_prediction,
    predict_image,
    predict_image_proba,
    prediction_scores_from_probs,
    predict_voice,
)


st.set_page_config(
    page_title="Parkinson Detection UI",
    page_icon="PD",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 195, 113, 0.45), transparent 32%),
            radial-gradient(circle at top right, rgba(107, 142, 255, 0.35), transparent 28%),
            linear-gradient(180deg, #f8f4ec 0%, #eef3f9 100%);
    }
    .hero-card, .result-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(28, 39, 64, 0.08);
        border-radius: 24px;
        padding: 1.4rem;
        box-shadow: 0 18px 45px rgba(38, 49, 77, 0.08);
        backdrop-filter: blur(8px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2f46;
        margin-bottom: 0.35rem;
    }
    .hero-copy {
        color: #425466;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0;
    }
    .metric-label {
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6a7a8c;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #17263a;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Parkinson Detection System</div>
            <p class="hero-copy">
                Use handwriting image and voice feature uploads to display prediction results in a simple, clinical-style UI.
All three modes—Image-only, Voice-only, and Fusion—can be run here.

            </p>
            <p class="hero-copy">
                Note: The image model is designed only for handwriting samples. Uploading a person face photo, selfie, or any random image will not produce a valid result.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(title, prediction_value):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{label_from_prediction(prediction_value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header_clean():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Parkinson Detection System</div>
            <p class="hero-copy">Use handwriting image and voice feature uploads to display prediction results in a simple, clinical-style UI. All three modes - Image-only, Voice-only, and Fusion - can be run here.</p>
            <p class="hero-copy">Note: The image model is designed only for handwriting samples. Uploading a person face photo, selfie, or any random image will not produce a valid result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_prediction_section():
    st.subheader("Handwriting Prediction")
    st.caption("Please upload only handwriting or drawing test images. This model will not work with face photos or regular camera images.")
    image_file = st.file_uploader(
        "Upload handwriting image",
        type=["png", "jpg", "jpeg"],
        key="image_only_upload",
    )

    if image_file is None:
        st.info("Once you upload a handwriting sample, the image-based prediction will be displayed here.")
        return

    image_bytes = image_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    left_col, right_col = st.columns([1.2, 1], gap="large")
    with left_col:
        st.image(image, caption="Uploaded handwriting sample", use_container_width=True)
    with right_col:
        if st.button("Predict From Image", type="primary", use_container_width=True):
            try:
                probabilities = predict_image_proba(io.BytesIO(image_bytes))
                prediction = int(probabilities.argmax())
                scores = prediction_scores_from_probs(probabilities)
                render_result("Image Model Result", prediction)
                st.progress(scores["Healthy"], text=f"Healthy confidence: {scores['Healthy']:.2%}")
                st.progress(
                    scores["Parkinson Disease Detected"],
                    text=f"Parkinson confidence: {scores['Parkinson Disease Detected']:.2%}",
                )
            except Exception as error:
                st.error(f"Image prediction failed: {error}")


def voice_prediction_section():
    st.subheader("Voice Prediction")
    st.caption("The current model does not accept raw audio files (`.wav`, `.mp3`). It expects a speech feature CSV file instead.")

    voice_file = st.file_uploader(
        "Upload voice feature CSV",
        type=["csv"],
        key="voice_only_upload",
    )

    if voice_file is None:
        st.info("When you upload a speech feature CSV file, the voice-based prediction result will be displayed here.")
        return

    voice_bytes = voice_file.getvalue()
    try:
        voice_df = pd.read_csv(io.BytesIO(voice_bytes))
        st.dataframe(voice_df.head(1), use_container_width=True)
    except Exception as error:
        st.error(f"CSV preview failed: {error}")
        return

    if st.button("Predict From Voice", type="primary", use_container_width=True):
        try:
            prediction = predict_voice(io.BytesIO(voice_bytes))
            render_result("Voice Model Result", prediction)
        except Exception as error:
            st.error(f"Voice prediction failed: {error}")


def fusion_prediction_section():
    st.subheader("Fusion Prediction")
    st.caption("For the best results, upload both the handwriting image and the voice feature CSV file.")

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        image_file = st.file_uploader(
            "Upload handwriting image",
            type=["png", "jpg", "jpeg"],
            key="fusion_image_upload",
        )
    with right_col:
        voice_file = st.file_uploader(
            "Upload voice feature CSV",
            type=["csv"],
            key="fusion_voice_upload",
        )

    if image_file:
        st.image(Image.open(io.BytesIO(image_file.getvalue())).convert("RGB"), caption="Handwriting preview", width=320)

    if not image_file or not voice_file:
        st.info("Both files are required to generate the fusion result.")
        return

    if st.button("Run Fusion Prediction", type="primary", use_container_width=True):
        try:
            image_probabilities = predict_image_proba(io.BytesIO(image_file.getvalue()))
            image_prediction = int(image_probabilities.argmax())
            voice_prediction = predict_voice(io.BytesIO(voice_file.getvalue()))
            fusion_result = final_prediction(image_prediction, voice_prediction)
            scores = prediction_scores_from_probs(image_probabilities)

            metrics_col1, metrics_col2, metrics_col3 = st.columns(3, gap="large")
            with metrics_col1:
                render_result("Image Result", image_prediction)
                st.caption(
                    f"Healthy {scores['Healthy']:.2%} | Parkinson {scores['Parkinson Disease Detected']:.2%}"
                )
            with metrics_col2:
                render_result("Voice Result", voice_prediction)
            with metrics_col3:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="metric-label">Fusion Result</div>
                        <div class="metric-value">{fusion_result}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as error:
            st.error(f"Fusion prediction failed: {error}")


def main():
    render_header_clean()

    mode = st.radio(
        "Choose prediction mode",
        ["Fusion", "Image", "Voice"],
        horizontal=True,
    )

    if mode == "Image":
        image_prediction_section()
    elif mode == "Voice":
        voice_prediction_section()
    else:
        fusion_prediction_section()


if __name__ == "__main__":
    main()