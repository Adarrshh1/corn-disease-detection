import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image
import datetime

# ✅ Page config
st.set_page_config(page_title="Corn Disease Detector", layout="wide")

# ✅ Load model
model = tf.keras.models.load_model("corn_model.h5")

classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

disease_info = {
    "Blight": "Fungal disease. Use fungicide spray.",
    "Common_Rust": "Rust spots appear. Use resistant varieties.",
    "Gray_Leaf_Spot": "Gray lesions. Needs proper treatment.",
    "Healthy": "No disease detected."
}

# 🌙 Sidebar
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio("Select Mode", ["Upload Image", "Use Camera"])

# 🎨 Title
st.markdown("<h1 style='text-align: center;'>🌽 Corn Disease Detection</h1>", unsafe_allow_html=True)

def predict_image(img):
    img = img.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return result, confidence, prediction

# 📤 Upload Mode
if mode == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded Image")

        result, confidence, prediction = predict_image(img)

        with col2:
            st.success(f"Prediction: {result}")
            st.info(f"Confidence: {confidence:.2f}%")

        st.write("### 🩺 Disease Info")
        st.write(disease_info[result])

        df = pd.DataFrame(prediction[0], index=classes, columns=["Probability"])
        st.bar_chart(df)

        # 📄 Download report
        report = f"""
        Corn Disease Report
        -------------------
        Prediction: {result}
        Confidence: {confidence:.2f}%
        Time: {datetime.datetime.now()}
        """

        st.download_button("📄 Download Report", report, file_name="report.txt")

# 📸 Camera Mode
elif mode == "Use Camera":
    img_file = st.camera_input("Take a picture")

    if img_file:
        img = Image.open(img_file)

        st.image(img, caption="Captured Image")

        result, confidence, prediction = predict_image(img)

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.write("### 🩺 Disease Info")
        st.write(disease_info[result])

        df = pd.DataFrame(prediction[0], index=classes, columns=["Probability"])
        st.bar_chart(df)