import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
model = load_model(r'C:\CodeShell_Core\GitHub_Repository\emotion_detection_cnn\emotion-detection-cnn\model\vgg16_emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion color themes
emotion_colors = {
    'Angry': ('red', (0, 0, 255)),        # Tkinter color, OpenCV BGR
    'Disgust': ('green', (0, 255, 0)),
    'Fear': ('purple', (128, 0, 128)),
    'Happy': ('orange', (0, 165, 255)),
    'Neutral': ('gray', (128, 128, 128)),
    'Sad': ('blue', (255, 0, 0)),
    'Surprise': ('magenta', (255, 0, 255))
}

# Emojis per emotion
emotion_emojis = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😲'
}

# Load face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Real-time Emotion Detection")
        self.window.geometry("720x580")
        self.window.resizable(False, False)

        # Title Banner
        self.title_label = Label(window, text="Emotion Detector AI", font=("Arial", 20, "bold"), bg="black", fg="white", pady=10)
        self.title_label.pack(fill="x")

        # Video Frame
        self.video_label = Label(window)
        self.video_label.pack(pady=10)

        # Emotion Output Label
        self.emotion_label = Label(window, text="", font=("Helvetica", 18, "bold"))
        self.emotion_label.pack(pady=10)

        # Start webcam
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        label = "No Face Detected"
        emoji = ''
        tk_color = 'black'

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
                roi = img_to_array(roi_color)
                roi = roi.astype("float") / 255.0
                roi = np.expand_dims(roi, axis=0)

                prediction = model.predict(roi)[0]
                label_index = np.argmax(prediction)
                label = emotion_labels[label_index]
                emoji = emotion_emojis.get(label, '')
                tk_color, cv_color = emotion_colors.get(label, ('black', (255, 255, 255)))

                # Draw rectangle and label on OpenCV frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), cv_color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cv_color, 2)
                break

        # Convert OpenCV frame to PIL Image and show
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Update emotion label in GUI (with emoji + color)
        self.emotion_label.configure(
            text=f"Detected Emotion: {label} {emoji}",
            fg=tk_color
        )

        self.window.after(10, self.update_frame)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == '__main__':
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
