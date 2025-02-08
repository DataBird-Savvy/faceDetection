import os
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder

# Define dataset path
dataset_path = "/content/drive/MyDrive/processed_faces"

# Function to extract FaceNet512 embeddings using DeepFace
def get_embedding(img_path):
    try:
        embedding = DeepFace.represent(img_path, model_name="Facenet512", enforce_detection=False,detector_backend="skip")[0]['embedding']
        return np.array(embedding)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Load dataset and extract embeddings
X, y = [], []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)
        embedding = get_embedding(img_path)

        if embedding is not None:
            X.append(embedding)
            y.append(person)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"X shape: {X.shape}")  # Should be (num_samples, embedding_size)
print(f"y shape: {y_encoded.shape}")  # Should be (num_samples,)

# Ensure the "artifacts" directory exists
os.makedirs("artifacts", exist_ok=True)

# Save embeddings and labels
np.savez_compressed("face_embeddings.npz", embeddings=X, labels=y_encoded)
print("Embeddings saved successfully! ✅")
