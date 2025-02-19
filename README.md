# Face Recognition App (Image & Video) 🎭🚀

This is a Streamlit-based Face Recognition App that leverages DeepFace (Facenet512) for face embeddings, MTCNN for face detection, and an SVM classifier for identity recognition. The app supports both image and video-based face recognition, allowing users to upload files and see real-time predictions.

**Note:** The training dataset focuses on five iconic personalities:
- **Virat Kohli** 🏏
- **Maria Sharapova** 🎾
- **Lionel Messi** ⚽
- **Serena Williams** 🎾
- **Roger Federer** 🎾

Due to the limited number of training images, data augmentation was applied using additional images of Lionel Messi to boost model performance.

---

## ✨ Features
- Recognizes faces in images & videos
- Uses DeepFace (Facenet512) for face embeddings
- Detects faces using MTCNN
- Classifies faces with an SVM model
- Processes videos with frame skipping for efficiency
- Interactive Streamlit web interface

---

## 📂 How It Works
1. **Upload** an image or video (JPG, PNG, or MP4).
2. **Detect** faces using MTCNN and extract embeddings using DeepFace.
3. **Predict** the identity using an SVM model.
4. **Display** recognized faces with bounding boxes and labels.
5. **Process videos** by analyzing frames and generating an output video with annotations.

---

## 🛠️ Tech Stack
- **Python** 🐍
- **OpenCV** 📸
- **DeepFace** 🤖
- **MTCNN** 🔍
- **Streamlit** 🌐
- **SVM Classifier** 📊
- **Joblib** (for model persistence)
- **scikit-learn** (for data preprocessing and model training)

---

## 🚀 Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone git@github.com:DataBird-Savvy/faceDetection.git



    ![image](https://github.com/user-attachments/assets/0b704859-a0a0-4862-b0d6-d70a17290b15)



📌 Future Improvements

🔹 Add support for real-time webcam face recognition
🔹 Improve model accuracy with more training data
🔹 Implement multiple backend models for comparison
🤝 Contribution

Contributions are welcome! Feel free to fork the repo, make improvements, and submit a PR.
