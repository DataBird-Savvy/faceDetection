Face Recognition App (Image & Video) 🎭🚀

This Streamlit-based Face Recognition App utilizes DeepFace (Facenet512) for face embeddings, MTCNN for face detection, and an SVM classifier for identity recognition. The app supports both image and video-based face recognition, allowing users to upload files and see real-time predictions.
✨ Features

✔ Recognizes Faces in Images & Videos
✔ Uses DeepFace (Facenet512) for Face Embeddings
✔ MTCNN for Face Detection
✔ SVM Classifier for Face Recognition
✔ Processes Videos with Frame Skipping for Efficiency
✔ Interactive Streamlit Web Interface
📂 How It Works

1️⃣ Upload an image or video (JPG, PNG, or MP4).
2️⃣ The app detects faces using MTCNN and extracts embeddings using DeepFace.
3️⃣ The SVM model predicts the person’s identity.
4️⃣ Recognized faces are displayed with bounding boxes and labels.
5️⃣ For videos, the app processes frames and generates an output video with face annotations.
🛠️ Tech Stack

    Python 🐍
    OpenCV 📸
    DeepFace 🤖
    MTCNN 🔍
    Streamlit 🌐
    SVM Classifier 📊
    Joblib (for model persistence)

🚀 Setup Instructions

1️⃣ Clone this repository:

git clone https://github.com/yourusername/face-recognition-app.git
cd face-recognition-app

2️⃣ Install dependencies:

pip install -r requirements.txt

3️⃣ Run the Streamlit app:

streamlit run app.py

📌 Future Improvements

🔹 Add support for real-time webcam face recognition
🔹 Improve model accuracy with more training data
🔹 Implement multiple backend models for comparison
🔹 Deploy on Streamlit Cloud or Hugging Face Spaces
🤝 Contribution

Contributions are welcome! Feel free to fork the repo, make improvements, and submit a PR.
