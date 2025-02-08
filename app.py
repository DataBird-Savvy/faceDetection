import streamlit as st
import cv2
import numpy as np
import joblib
import tempfile
import os
from deepface import DeepFace
from mtcnn import MTCNN
from io import BytesIO
import shutil

# Initialize MTCNN for face detection
detector = MTCNN()

# Label mapping for predictions
label_mapping = {
    0: "Lionel Messi",
    1: "Maria Sharapova",
    2: "Roger Federer",
    3: "Serena Williams",
    4: "Virat Kohli"
}

# Load trained SVM model and label encoder
svm_model = joblib.load("artifacts/face_recognition_model.pkl")
label_encoder = joblib.load("artifacts/label_encoder.pkl")

# Set DeepFace model
deepface_model = "Facenet512"

def get_embedding(face_crop):
    try:
        # Convert face crop to in-memory image
        _, buffer = cv2.imencode(".jpg", face_crop)
        img_bytes = BytesIO(buffer)

        # Save the image as a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(img_bytes.getvalue())
            temp_face_path = temp_file.name

        # Get embedding using DeepFace from the temporary file path
        embedding = DeepFace.represent(
            img_path=temp_face_path,
            model_name=deepface_model,
            enforce_detection=False,
            detector_backend="mtcnn"
        )[0]['embedding']

        # Clean up the temporary file
        os.remove(temp_face_path)

        return np.array(embedding).reshape(1, -1)

    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


# Function to recognize faces in images
def recognize_and_draw_faces(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)

    for face in faces:
        x, y, w, h = face["box"]
        face_crop = image[y:y+h, x:x+w]
        
        embeddings = get_embedding(face_crop)
        if embeddings is None:
            label = "Unknown"
        else:
            probabilities = svm_model.predict_proba(embeddings)[0]
            max_prob = np.max(probabilities)
            predicted_label = np.argmax(probabilities)
            label = label_mapping.get(predicted_label, "Unknown") if max_prob >= 0.8 else "Unknown"

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_skip = 5  
    frame_count = 0

    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:  
            frame = recognize_and_draw_faces(frame)
        
        # Display frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        frame_count += 1

    cap.release()

# Function to save processed video
def save_video(frames, output_path="output_video.mp4"):
    if len(frames) == 0:
        print("Error: No frames to save!")
        return None

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # "avc1" may not work on all devices
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for frame in frames:
        out.write(frame)  # No need to convert back to BGR

    out.release()
    return output_path

# Function to safely remove files
def safe_remove(file_path):
    try:
        os.remove(file_path)
    except PermissionError:
        shutil.move(file_path, tempfile.gettempdir())

# Streamlit UI
st.title("🚀 Face Recognition App (Image & Video)")
st.write("Upload an image or video to recognize faces.")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]

    # Process image
    if file_extension in ["jpg", "png"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp:
            temp.write(uploaded_file.read())
            temp_path = temp.name

        image = cv2.imread(temp_path)
        result_image = recognize_and_draw_faces(image)

        st.image(result_image, caption="Face Recognition Result", use_container_width=True)
        safe_remove(temp_path)

    # Process video
    elif file_extension == "mp4":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp.write(uploaded_file.read())
            temp_path = temp.name

        st.video(temp_path)  # Show uploaded video before processing
        st.write("🔄 Processing video, please wait...")

        frames = process_video(temp_path)

        # Save processed video
        output_video_path = save_video(frames, "processed_video.mp4")

        if output_video_path:
            st.video(output_video_path)  # Display processed video
            safe_remove(temp_path)
            safe_remove(output_video_path)
