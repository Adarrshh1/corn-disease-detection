import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image   # ✅ IMPORTANT

model = tf.keras.models.load_model("corn_model.h5")

img = image.load_img("test.jpg", target_size=(224,224))
img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0   # normalization

prediction = model.predict(img_array)

classes = ['Blight','Common_Rust','Gray_Leaf_Spot','Healthy']

print("Prediction:", classes[np.argmax(prediction)])