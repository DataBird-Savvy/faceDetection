from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
import joblib
import numpy as np
import os


data = np.load("/content/face_embeddings.npz")
aug_data = np.load("/content/augface_embeddings.npz")

X = np.concatenate((data["embeddings"], aug_data["embeddings"]), axis=0)
y = np.concatenate((data["labels"], aug_data["labels"]), axis=0)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


scaler = StandardScaler()
X = scaler.fit_transform(X)


pca = PCA(n_components=100)
X = pca.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


svm_model = SVC(kernel="linear", C=0.1, probability=True)
svm_model.fit(X_train, y_train)


cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
print(f"Cross-validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")


y_pred = svm_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")

target_names = [str(label) for label in label_encoder.classes_]
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))



os.makedirs("artifacts", exist_ok=True)
joblib.dump(svm_model, "artifacts/face_recognition_model.pkl")
joblib.dump(label_encoder, "artifacts/label_encoder.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(pca, "artifacts/pca.pkl")
print("Model, label encoder, scaler, and PCA saved successfully! ✅")

