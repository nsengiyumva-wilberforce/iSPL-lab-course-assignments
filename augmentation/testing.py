import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model you trained
model = tf.keras.models.load_model("color_model.h5")

# Load and preprocess your own image
img = image.load_img("test_image.jpeg", target_size=(32, 32))  # resize to CIFAR-10 size
img_array = image.img_to_array(img)
img_array = img_array.astype("float32") / 255.0  # normalize 0–1
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
print("Predicted class:", class_names[predicted_class])
print("Confidence:", confidence)
