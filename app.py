import streamlit as st
import cv2
import numpy as np
import joblib
import tempfile
import os
from deepface import DeepFace
from mtcnn import MTCNN

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

# Function to extract embeddings using DeepFace
def get_embedding(face_crop):
    try:
        temp_face_path = "temp_face.jpg"
        cv2.imwrite(temp_face_path, face_crop)

        embedding = DeepFace.represent(
            img_path=temp_face_path,
            model_name=deepface_model,
            enforce_detection=False,
            detector_backend="mtcnn"
        )[0]['embedding']

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
    output_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:  
            frame = recognize_and_draw_faces(frame)

        output_frames.append(frame)
        frame_count += 1

    cap.release()
    return output_frames

# Function to save processed video
def save_video(frames, output_path="output_video.mp4"):
    if len(frames) == 0:
        print("Error: No frames to save!")
        return None

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Use "XVID" for AVI
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    return output_path

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
        os.remove(temp_path)

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
            os.remove(temp_path)
            os.remove(output_video_path)