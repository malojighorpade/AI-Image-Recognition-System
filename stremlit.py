import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras


# Load the trained model

MODEL_PATH = "cat_dog_model.keras"

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

model = load_model()


# Preprocessing Function
#
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------
# Prediction Function
# -------------------------
def predict(image):
    prediction = model.predict(image, verbose=0)[0]
    prediction_value = prediction if np.isscalar(prediction) else prediction[0]
    label = 1 if prediction_value > 0.5 else 0
    confidence = prediction_value if label == 1 else 1 - prediction_value
    return {"label": label, "confidence": float(confidence)}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Cat vs Dog Classifier Image classifier", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #2E86C1;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #e8f6ff;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #1F618D;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Subtitle
st.markdown('<div class="main-title">🐶🐱 Cat vs Dog Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to find out whether it’s a cat or a dog!</div>', unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("---")
    st.write("### 🔍 Processing...")

    processed_image = preprocess_image(image)
    with st.spinner("Classifying..."):
        result = predict(processed_image)

    label = "Dog 🐶" if result["label"] == 1 else "Cat 🐱"
    confidence = result["confidence"] * 100

    # Stylish Output
    st.markdown(f"<div class='prediction-box'>✅ Prediction: <span style='font-size: 2rem;'>{label}</span></div>", unsafe_allow_html=True)

    # Confidence Bar
    st.write("### 🔢 Confidence Level")
    st.progress(confidence / 100)
    st.write(f"**Confidence Score:** `{confidence:.2f}%`")

else:
    st.info("⬆ Upload a JPG or PNG image to start classification.")
