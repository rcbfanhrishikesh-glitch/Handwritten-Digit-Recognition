import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("digit_model.h5")

img = cv2.imread("my_digit.jpeg", cv2.IMREAD_GRAYSCALE)

# Preprocess
img = cv2.resize(img, (28, 28))
img = 255 - img
img = img / 255.0
img = img.reshape(1, 784)

prediction = model.predict(img)
print("Predicted digit is:", np.argmax(prediction))
