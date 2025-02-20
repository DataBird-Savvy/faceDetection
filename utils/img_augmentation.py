from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  
    fill_mode='nearest'
)


folder_path = "data/processed_faces/lionel_messi"
result_path = "data/augmented_faces/lionel_messi"


if not os.path.exists(result_path):
    os.makedirs(result_path)


images = [img for img in os.listdir(folder_path) if img.endswith('.jpg')]


for img_name in images:
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Warning: Unable to read image {img_path}")
        continue

    img = cv2.resize(img, (160, 160))  


    img = np.expand_dims(img, axis=0)

    aug_iter = datagen.flow(img, batch_size=1, save_to_dir=result_path, save_prefix="augmented_", save_format="jpg")
    
    
    for i in range(2):
        next(aug_iter)  

print("Augmentation completed successfully!")
