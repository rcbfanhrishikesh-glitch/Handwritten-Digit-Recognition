import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Flatten images (28x28 → 784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 4. Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 5. Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train
model.fit(x_train, y_train, epochs=5)

# 7. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Model Accuracy:", acc)

# 8. Save the model
model.save("digit_model.h5")
print("Model saved as digit_model.h5")

# 9. Test prediction with one sample
index = 0
plt.imshow(x_test[index].reshape(28, 28), cmap="gray")
plt.show()

prediction = model.predict(x_test[index].reshape(1, 784))
print("Predicted digit:", np.argmax(prediction))
