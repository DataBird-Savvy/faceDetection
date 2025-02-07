from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  # For saving the model
import numpy as np
import os

# Load embeddings and labels
data = np.load("/content/face_embeddings.npz")
X, y = data["embeddings"], data["labels"]

print(f"X shape: {X.shape if hasattr(X, 'shape') else len(X)}")



# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"y_encoded shape: {y_encoded.shape if hasattr(y_encoded, 'shape') else len(y_encoded)}")


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)
print("Model training complete! ✅")

# Convert numeric labels to strings for classification_report
target_names = [str(class_name) for class_name in label_encoder.classes_]
y_pred = svm_model.predict(X_test)
# Print classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))



# Save the model and label encoder
os.makedirs("artifacts", exist_ok=True)
joblib.dump(svm_model, "artifacts/face_recognition_model.pkl")
joblib.dump(label_encoder, "artifacts/label_encoder.pkl")
print("Model and label encoder saved successfully! ✅")
