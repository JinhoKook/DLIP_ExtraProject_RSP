from PIL import Image
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sys
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
model = load_model("rsp_model.h5")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Resize the frame for the model
    frame_resized = cv2.resize(frame, (56, 56))
    img = np.array(frame_resized, dtype=np.float32) / 255.0     # Normalize image

    # Reshape the image dimension
    img = np.expand_dims(img, axis=0)  # Image dimension reshaped: (1, 56, 56, 3)

    # Predict the result using the model
    predicted_result = model.predict(img)
    predicted_labels = np.argmax(predicted_result, axis=1)

    # Label text mapping
    labels = ["Scissors", "Rock", "Paper"]
    predicted_text = labels[predicted_labels[0]]

    win_labels = []

    if predicted_text == 'Scissors':
        win_labels = 'Rock'
    if predicted_text == 'Rock':
        win_labels = 'Paper'
    if predicted_text == 'Paper':
        win_labels = 'Scissors'            

    # Display the predicted label on the image
    cv2.putText(frame, f" This is: {predicted_text}.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, " If you want win", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 255), 2)
    cv2.putText(frame, f" you have to {win_labels}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 255), 2)  
    

    # Display the frame
    cv2.imshow("Rock Scissor Paper ! ", frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Quit code
cap.release()
cv2.destroyAllWindows()