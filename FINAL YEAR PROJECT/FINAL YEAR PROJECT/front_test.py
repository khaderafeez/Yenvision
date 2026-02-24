import tkinter as tk
import cv2
from PIL import Image, ImageTk
import test2

# Function to load next image and run emotion detection
def load_next_image(root):
    root.destroy()
    test2.main()

# Function to start real-time emotion detection
def start_detection(root, frame_label, stop_flag):
    cap = cv2.VideoCapture(0)
    root.title("Real-time Emotion Detection")
    stop_flag["stop"] = False  # Reset stop flag

    def update_frame():
        ret, frame = cap.read()
        if ret and not stop_flag["stop"]:
            # Display the video feed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            frame_label.config(image=photo)
            frame_label.image = photo
            frame_label.after(10, update_frame)  # Update every 10 milliseconds
        else:
            cap.release()
            cv2.destroyAllWindows()

    update_frame()

# Function to stop real-time emotion detection
def stop_detection(stop_flag):
    stop_flag["stop"] = True

# Main function
def main():
    root = tk.Tk()
    root.title("Emotion Detection")

    frame_label = tk.Label(root)
    frame_label.pack(pady=10)

    # Dictionary to hold stop flag
    stop_flag = {"stop": False}

    # Button to load next image and run emotion detection
    next_button = tk.Button(root, text="Start", command=lambda: load_next_image(root))
    next_button.pack(pady=10)

    # Button to start real-time emotion detection
    start_button = tk.Button(root, text="Camera", command=lambda: start_detection(root, frame_label, stop_flag))
    start_button.pack(pady=10)

    # Button to stop real-time emotion detection
    # stop_button = tk.Button(root, text="Stop", command=lambda: stop_detection(stop_flag))
    # stop_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
