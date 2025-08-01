import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load trained model
MODEL_PATH = r'C:\CodeShell_Core\GitHub_Repository\emotion_detection_cnn\emotion-detection-cnn\model\emotion_model.h5'
model = load_model(MODEL_PATH)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict_emotion(image: Image.Image):
    image = image.convert('RGB')
    image = image.resize((48, 48))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    prediction = model.predict(img_array)[0]
    return emotion_labels[np.argmax(prediction)], prediction

# Streamlit UI
st.title("🧠 Emotion Detection from Image")
st.write("Upload an image with a face and see the predicted emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    label, probabilities = predict_emotion(image)
    
    st.success(f"Predicted Emotion: **{label}**")
    st.write("Confidence:")
    for i in range(len(emotion_labels)):
        st.write(f"{emotion_labels[i]}: {probabilities[i]:.2f}")
