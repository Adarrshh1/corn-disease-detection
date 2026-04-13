import os
import urllib.request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# =========================
# MODEL SETUP
# =========================
MODEL_PATH = "corn_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1lCyqvrpGGBGTJnT3ulkZ25ufT9-kOZUD"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    classes = ['Blight','Common_Rust','Gray_Leaf_Spot','Healthy']

    return classes[np.argmax(prediction)]