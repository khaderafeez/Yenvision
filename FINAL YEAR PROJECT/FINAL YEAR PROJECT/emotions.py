# import cv2
# import numpy as np
# from keras.models import load_model
# from statistics import mode
# from utils.datasets import get_labels
# from utils.inference import detect_faces
# from utils.inference import draw_text
# from utils.inference import draw_bounding_box
# from utils.inference import apply_offsets
# from utils.inference import load_detection_model
# from utils.preprocessor import preprocess_input

# USE_WEBCAM = True # If false, loads video file source

# # parameters for loading data and images
# emotion_model_path = './models/emotion_model.hdf5'
# emotion_labels = get_labels('fer2013')

# # hyper-parameters for bounding boxes shape
# frame_window = 10
# emotion_offsets = (20, 40)

# # loading models
# face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
# emotion_classifier = load_model(emotion_model_path)

# # getting input model shapes for inference
# emotion_target_size = emotion_classifier.input_shape[1:3]

# # starting lists for calculating modes
# emotion_window = []

# # starting video streaming

# cv2.namedWindow('window_frame')
# video_capture = cv2.VideoCapture(0)

# # Select video or webcam feed
# cap = None
# if (USE_WEBCAM == True):
#     cap = cv2.VideoCapture(0) # Webcam source
# else:
#     cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source

# while cap.isOpened(): # True:
#     ret, bgr_image = cap.read()

#     #bgr_image = video_capture.read()[1]

#     gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
#     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

#     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
# 			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

#     for face_coordinates in faces:

#         x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
#         gray_face = gray_image[y1:y2, x1:x2]
#         try:
#             gray_face = cv2.resize(gray_face, (emotion_target_size))
#         except:
#             continue

#         gray_face = preprocess_input(gray_face, True)
#         gray_face = np.expand_dims(gray_face, 0)
#         gray_face = np.expand_dims(gray_face, -1)
#         emotion_prediction = emotion_classifier.predict(gray_face)
#         emotion_probability = np.max(emotion_prediction)
#         emotion_label_arg = np.argmax(emotion_prediction)
#         emotion_text = emotion_labels[emotion_label_arg]
#         emotion_window.append(emotion_text)

#         if len(emotion_window) > frame_window:
#             emotion_window.pop(0)
#         try:
#             emotion_mode = mode(emotion_window)
#         except:
#             continue

#         if emotion_text == 'angry':
#             color = emotion_probability * np.asarray((255, 0, 0))
#         elif emotion_text == 'sad':
#             color = emotion_probability * np.asarray((0, 0, 255))
#         elif emotion_text == 'happy':
#             color = emotion_probability * np.asarray((255, 255, 0))
#         elif emotion_text == 'surprise':
#             color = emotion_probability * np.asarray((0, 255, 255))
#         else:
#             color = emotion_probability * np.asarray((0, 255, 0))

#         color = color.astype(int)
#         color = color.tolist()

#         draw_bounding_box(face_coordinates, rgb_image, color)
#         draw_text(face_coordinates, rgb_image, emotion_mode,
#                   color, 0, -45, 1, 1)

#     bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
#     cv2.imshow('window_frame', bgr_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import os

# Set this flag to True if using webcam, False if using a video file source
USE_WEBCAM = True  

# Path to the emotion detection model
emotion_model_path = './models/emotion_model.hdf5'

# Get emotion labels
emotion_labels = get_labels('fer2013')

# Hyperparameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# Load models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]

# Start video capture
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0) if USE_WEBCAM else cv2.VideoCapture('./demo/dinner.mp4')

# Path to the folder containing the images
images_folder = 'Images'

# Initialize emotion window
emotion_window = []

# Function to select the image corresponding to the detected emotion
def select_emotion_image(emotion_label):
    image_path = os.path.join(images_folder, emotion_label + '.jpg')
    return cv2.imread(image_path)

# Main loop
while video_capture.isOpened():
    ret, bgr_image = video_capture.read()

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        # Determine color based on emotion
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        # Draw bounding box and emotion text
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

        # Display image corresponding to the detected emotion
        emotion_image = select_emotion_image(emotion_text)
        if emotion_image is not None:
            rgb_image[10:110, 10:110] = emotion_image

    # Display the processed frame
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
