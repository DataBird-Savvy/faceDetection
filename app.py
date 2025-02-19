import streamlit as st
import cv2
import numpy as np
import joblib
import tempfile
import os
from deepface import DeepFace
from mtcnn import MTCNN


detector = MTCNN()

label_mapping = {
    0: "Lionel Messi",
    1: "Maria Sharapova",
    2: "Roger Federer",
    3: "Serena Williams",
    4: "Virat Kohli"
}

svm_model = joblib.load("artifacts/face_recognition_model.pkl")
label_encoder = joblib.load("artifacts/label_encoder.pkl")
pca = joblib.load("artifacts/pca.pkl")
scalar=joblib.load("artifacts/scaler.pkl")


deepface_model = "Facenet512"


def get_embedding(face_crop):
    try:
        temp_face_path = "temp_face.jpg"
        cv2.imwrite(temp_face_path, face_crop)

        # Extract embedding using DeepFace
        embedding = DeepFace.represent(
            img_path=temp_face_path,
            model_name=deepface_model,
            enforce_detection=False,
            detector_backend="mtcnn"
        )[0]['embedding']

        embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)

        # Apply feature scaling and PCA
        embedding_scaled = scalar.transform(embedding)  # Standardization
        embedding_pca = pca.transform(embedding_scaled)  # Dimensionality Reduction

        return embedding_pca

    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


def recognize_and_draw_faces(image, is_video):  
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)

    if not faces:
        if not is_video:
            st.warning("⚠ No faces detected!")
        return image  # Return original image if no face found

    for face in faces:
        x, y, w, h = face["box"]
        face_crop = image[max(0, y):max(0, y+h), max(0, x):max(0, x+w)]

        if face_crop.size == 0:
            continue  # Skip invalid faces

        face_crop = cv2.resize(face_crop, (160, 160))  # Resize for DeepFace
        embeddings = get_embedding(face_crop)

        if embeddings is None:
            label = "Unknown"
            max_prob = 0.0  # Assign default confidence
        else:
            probabilities = svm_model.predict_proba(embeddings)[0]
            max_prob = np.max(probabilities)  
            predicted_label = np.argmax(probabilities)
            label = label_mapping.get(predicted_label, "Unknown") if max_prob >= 0.8 else "Unknown"

        # Draw bounding box and label
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(image, f"{label} ({max_prob:.2f})", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        # Show prediction message **only for images**
        if not is_video:
            st.write(f"✅ Predicted: **{label}** | Confidence: **{max_prob:.2f}**")

    return image



def process_video(video_path, frame_skip=2):
    cap = cv2.VideoCapture(video_path)
    output_frames = []
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    progress_bar = st.progress(0)

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  

        if current_frame % frame_skip == 0:  
            frame = recognize_and_draw_faces(frame, is_video=True)
            output_frames.append(frame)

        current_frame += 1
        progress_bar.progress(current_frame / total_frames)

    cap.release()
    progress_bar.empty()
    return output_frames,fps



def save_video(frames, output_path="processed_video.mp4", fps=25):
    if not frames:
        print("No frames to save.")
        return None

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    return output_path  # Return file path

# Streamlit UI
st.title("🚀 Face Recognition App (Image & Video)")
st.write("Upload an image or video to recognize faces.")

# File uploader for images and videos
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4"])



if uploaded_file:
    
    file_extension = uploaded_file.name.split(".")[-1]

    # Process image
    if file_extension in ["jpg", "png"]:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp:
            temp.write(uploaded_file.read())
            temp_path = temp.name

        image = cv2.imread(temp_path)
        result_image = recognize_and_draw_faces(image,is_video=False)

        st.image(result_image, caption="Face Recognition Result", use_container_width=True)
        os.remove(temp_path)

    # Process video
    elif file_extension == "mp4":
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp.write(uploaded_file.read())
            temp_path = temp.name

        st.video(temp_path)  # Show uploaded video before processing
        st.write("🔄 Processing video, please wait...")

        frames,fps = process_video(temp_path)

        if frames:
            output_video_path = save_video(frames, "processed_output.mp4",fps)
            st.video(output_video_path)  # Display processed video

            # Add download button
            with open(output_video_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                label="📥 Download Processed Video",
                data=video_bytes,
                file_name="processed_output.mp4",
                mime="video/mp4",
            )
