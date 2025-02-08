from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os

# Define augmentation techniques
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Random brightness
    fill_mode='nearest'
)

# Folder containing cropped face images
folder_path = "data/processed_faces/lionel_messi"

# Get all subfolders (if you have subfolders inside the 'lionel_messi' folder)
subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

# Iterate over each subfolder
for subfolder in subfolders:
    subfolder_path = os.path.join(folder_path, subfolder)
    images = [img for img in os.listdir(subfolder_path) if img.endswith('.jpg')]

    # Process each image in the subfolder
    for img_name in images:
        img_path = os.path.join(subfolder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (160, 160))  # Resize for FaceNet

        # Convert to numpy and expand dimensions
        img = np.expand_dims(img, axis=0)

        # Generate augmented images and save them
        aug_iter = datagen.flow(img, batch_size=1, save_to_dir=subfolder_path, save_prefix="augmented_", save_format="jpg")
        
        # Save 5 augmented images
        for i in range(5):  # Save 5 augmented images
            next(aug_iter)  # Generate and save the augmented image
