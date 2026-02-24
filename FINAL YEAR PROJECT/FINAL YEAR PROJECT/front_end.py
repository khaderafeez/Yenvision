# import tkinter as tk
# import cv2
# from PIL import Image, ImageTk
# import test2

# # Function to load next image and run emotion detection
# def load_next_image(root):
#     root.destroy()
#     test2.main()

# # Function to display real-time video feed with emotion detection
# def display_realtime_emotion(root):
#     cap = cv2.VideoCapture(0)
#     root.title("Real-time Emotion Detection")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Display the video feed
#         cv2.imshow("Real-time Emotion Detection", frame)

#         # Perform emotion detection
#         emotion_detected = test2.detect_realtime_emotion(frame)
#         if emotion_detected:
#             display_emotion_image(root, emotion_detected)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Function to display the image corresponding to the detected emotion
# def display_emotion_image(root, emotion):
#     root.withdraw()  # Hide the main window temporarily

#     # Load the image corresponding to the detected emotion
#     image_path = f"Images/{emotion}.jpg"
#     image = Image.open(image_path)
#     image = image.resize((300, 300), Image.ANTIALIAS)
#     photo = ImageTk.PhotoImage(image)

#     # Display the image in a separate window
#     image_window = tk.Toplevel(root)
#     image_window.title("Emotion Image")
#     label = tk.Label(image_window, image=photo)
#     label.image = photo
#     label.pack()

#     # Display success message if emotion matches expected emotion
#     if emotion == test2.expected_emotion:
#         show_success_message()

# # Function to display success message in a separate window
# def show_success_message():
#     success_window = tk.Toplevel()
#     success_window.title("Success!")
#     success_label = tk.Label(success_window, text="Emotion Detected Successfully!")
#     success_label.pack()

# # Main function
# def main():
#     root = tk.Tk()
#     root.title("Emotion Detection")

#     # Button to load next image and run emotion detection
#     next_button = tk.Button(root, text="Next Emotion", command=lambda: load_next_image(root))
#     next_button.pack()

#     # Button to start real-time emotion detection
#     start_button = tk.Button(root, text="Start Real-time Emotion Detection", command=lambda: display_realtime_emotion(root))
#     start_button.pack()

#     root.mainloop()

# if __name__ == "__main__":
#     main()

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import test2

# Function to load next image and run emotion detection
def load_next_image(root):
    root.destroy()
    test2.main()

# Function to display real-time video feed with emotion detection
def display_realtime_emotion(root):
    cap = cv2.VideoCapture(0)
    root.title("Real-time Emotion Detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the video feed
        cv2.imshow("Real-time Emotion Detection", frame)

        # Perform emotion detection
        emotion_detected = test2.detect_realtime_emotion(frame)
        if emotion_detected:
            display_emotion_image(root, emotion_detected)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to display the image corresponding to the detected emotion
def display_emotion_image(root, emotion):
    root.withdraw()  # Hide the main window temporarily

    # Load the image corresponding to the detected emotion
    image_path = f"Images/{emotion}.jpg"
    image = Image.open(image_path)
    image = image.resize((300, 300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)

    # Display the image in a separate window
    image_window = tk.Toplevel(root)
    image_window.title("Emotion Image")
    label = tk.Label(image_window, image=photo)
    label.image = photo
    label.pack()

    # Display success message if emotion matches expected emotion
    if emotion == test2.expected_emotion:
        show_success_message()

# Function to display success message in a separate window
def show_success_message():
    success_window = tk.Toplevel()
    success_window.title("Success!")
    success_label = tk.Label(success_window, text="Emotion Detected Successfully!")
    success_label.pack()

# Main function
def main():
    root = tk.Tk()
    root.title("Emotion Detection")

    # Button to load next image and run emotion detection
    next_button = tk.Button(root, text="START", command=lambda: load_next_image(root))
    next_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()