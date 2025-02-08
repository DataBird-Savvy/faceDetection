import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize MTCNN
detector = MTCNN()

# Define source and destination folders
source_dir = "data/images_dataset"  # Folder with identity subfolders
destination_dir = "data/processed_faces"  # Output folder
os.makedirs(destination_dir, exist_ok=True)

def process_images():
    for person_name in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person_name)
        
        # Check if it's a directory (valid person folder)
        if not os.path.isdir(person_path):
            continue  

        # Create a destination folder for this person
        save_path = os.path.join(destination_dir, person_name)
        os.makedirs(save_path, exist_ok=True)

        img_id = 0
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            # Ensure it's an image file (skip non-image files)
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue  

            try:
                image = load_img(img_path)  # Loads in RGB format
                image = img_to_array(image)  # Convert to NumPy array (still RGB)

                # Detect faces
                faces = detector.detect_faces(image)
                if not faces:
                    print(f"No face detected in {img_name}. Skipping...")
                    continue

                for face in faces:
                    x, y, w, h = face['box']
                    if x < 0 or y < 0:  # Ignore invalid bounding boxes
                        continue
                    
                    cropped_face = image[y:y+h, x:x+w]
                    cropped_face = cv2.resize(cropped_face, (200, 200))  # Resize for consistency
                    
                    # **Fix: Convert RGB to BGR before saving**
                    cropped_face_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
                    
                    img_id += 1
                    save_filename = os.path.join(save_path, f"{person_name}_{img_id}.jpg")
                    cv2.imwrite(save_filename, cropped_face_bgr)

                print(f"Processed {img_name} -> Saved to {save_path}")

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

process_images()
print("Processing complete! 🎉")