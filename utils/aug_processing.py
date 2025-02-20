import os
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder


dataset_path = "/content/drive/MyDrive/augmented_faces"


def get_embedding(img_path):
    try:
        embedding = DeepFace.represent(img_path, model_name="Facenet512", enforce_detection=False,detector_backend="skip")[0]['embedding']
        return np.array(embedding)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


X, y = [], []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)
        embedding = get_embedding(img_path)

        if embedding is not None:
            X.append(embedding)
            y.append(person)


X = np.array(X)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"X shape: {X.shape}")  
print(f"y shape: {y_encoded.shape}")  

os.makedirs("artifacts", exist_ok=True)


np.savez_compressed("augface_embeddings.npz", embeddings=X, labels=y_encoded)
print("Embeddings saved successfully! ✅")
