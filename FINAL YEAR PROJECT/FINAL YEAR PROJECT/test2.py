import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
import os
import random
import tkinter as tk

# Path to the emotion detection model
emotion_model_path = './Model2/emotion_detection_model2.h5'
# emotion_model_path = './models/emotion_model.hdf5'

# Get emotion labels
emotion_labels = get_labels('fer2013')

# Function to load the face detection model
def load_detection_model(model_path):
    return cv2.CascadeClassifier(model_path)

# Path to the face detection model
face_cascade_path = './Model2/haarcascade_frontalface_default.xml'
# face_cascade_path = './models/haarcascade_frontalface_default.xml'

# Load the face detection model
face_cascade = load_detection_model(face_cascade_path)

# Load the emotion detection model
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]

# Hyperparameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# Function to select a random image from the folder
def select_random_image(folder_path):
    image_files = os.listdir(folder_path)
    random_image = random.choice(image_files)
    return os.path.join(folder_path, random_image)

# Function to detect emotion from the user's video feed
def detect_emotion(frame, expected_emotion):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        try:
            face = cv2.resize(face, emotion_target_size)
        except Exception as e:
            print("Error:", e)
            continue

        face = preprocess_input(face, True)
        face = np.expand_dims(face, 0)
        face = np.expand_dims(face, -1)
        
        emotion_prediction = emotion_classifier.predict(face)
        emotion_label_arg = np.argmax(emotion_prediction)
        detected_emotion = emotion_labels[emotion_label_arg]
        
        print("Expected emotion:", expected_emotion, "Detected emotion:", detected_emotion)
        
        if detected_emotion == expected_emotion:
            return True

    return False

# Function to display the image and user video
def display_image_and_user_video(image_path):
    image = cv2.imread(image_path)
    cv2.namedWindow('Image and User Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image and User Video', 1280, 480)

    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (image.shape[1], image.shape[0]))
        combined_frame = np.hstack((image, frame_resized))
        cv2.imshow('Image and User Video', combined_frame)
        
        expected_emotion = os.path.splitext(os.path.basename(image_path))[0]
        if detect_emotion(frame, expected_emotion):
            show_result_message("Success")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Function to show result message in a separate window
def show_result_message(message):
    root = tk.Tk()
    root.title("Validation Result")
    label = tk.Label(root, text=message)
    label.pack()
    root.mainloop()

# Main function
def main():
    folder_path = 'Images'
    while True:
        image_path = select_random_image(folder_path)
        display_image_and_user_video(image_path)
        
if __name__ == "__main__":
    main()