import tkinter as tk
from tkinter import *
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

# Load the trained MNIST model
model = tf.keras.models.load_model("digit_model.h5")

# Create main window
root = tk.Tk()
root.title("Handwritten Digit Recognizer")
root.geometry("400x500")

# Canvas for drawing
canvas = tk.Canvas(root, width=280, height=280, bg='white')
canvas.pack(pady=20)

# PIL image to store drawings
image = Image.new("L", (280, 280), color=255)
draw = ImageDraw.Draw(image)


# ----------------------------
# DRAWING FUNCTIONS
# ----------------------------

def start_pos(event):
    canvas.x = event.x
    canvas.y = event.y

def draw_lines(event):
    canvas.create_line(canvas.x, canvas.y, event.x, event.y, width=12, fill='black', capstyle=ROUND, smooth=True)
    draw.line([canvas.x, canvas.y, event.x, event.y], fill=0, width=20)
    canvas.x = event.x
    canvas.y = event.y


# ----------------------------
# CLEAR CANVAS
# ----------------------------
def clear_canvas():
    canvas.delete("all")
    global image, draw
    image = Image.new("L", (280, 280), color=255)
    draw = ImageDraw.Draw(image)
    result_label.config(text="Draw a digit and click Predict")


# ----------------------------
# PREDICT DIGIT
# ----------------------------
def predict_digit():
    # Resize to 28x28
    img = image.resize((28, 28))

    # Invert (MNIST digits are white on black)
    img = ImageOps.invert(img)

    # Convert to array
    img_array = np.array(img) / 255.0

    # Flatten for dense model (784 features)
    img_array = img_array.reshape(1, 784)

    # Predict
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    result_label.config(text=f"Digit: {digit} | Confidence: {confidence:.2f}%")


# Bind drawing events
canvas.bind("<Button-1>", start_pos)
canvas.bind("<B1-Motion>", draw_lines)

# Buttons
btn_predict = tk.Button(root, text="Predict Digit", command=predict_digit, font=("Arial", 14))
btn_predict.pack(pady=10)

btn_clear = tk.Button(root, text="Clear", command=clear_canvas, font=("Arial", 14))
btn_clear.pack(pady=5)

# Result label
result_label = tk.Label(root, text="Draw a digit and click Predict", font=("Arial", 16))
result_label.pack(pady=20)

root.mainloop()
