import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
eye_open_path = "D:\project\dataset\open_eyes"
eye_closed_path = "D:\project\dataset\closed_eyes"
yawn_path = "D:\project\extracted_mouth_frames\yawn"
no_yawn_path =r"D:\project\extracted_mouth_frames\no_yawn"

# Function to load images and assign label
def load_images_from_folder(folder, label, img_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return images, labels

# Load Eye images
eye_open_imgs, eye_open_labels = load_images_from_folder(eye_open_path, 1)
eye_closed_imgs, eye_closed_labels = load_images_from_folder(eye_closed_path, 0)

# Load Mouth images
yawn_imgs, yawn_labels = load_images_from_folder(yawn_path, 1)
no_yawn_imgs, no_yawn_labels = load_images_from_folder(no_yawn_path, 0)

# Combine and shuffle datasets
eye_images = np.array(eye_open_imgs + eye_closed_imgs)
eye_labels = np.array(eye_open_labels + eye_closed_labels)

mouth_images = np.array(yawn_imgs + no_yawn_imgs)
mouth_labels = np.array(yawn_labels + no_yawn_labels)

# Normalize images
eye_images = eye_images.astype("float32") / 255.0
mouth_images = mouth_images.astype("float32") / 255.0

# Add channel dimension
eye_images = np.expand_dims(eye_images, axis=-1)
mouth_images = np.expand_dims(mouth_images, axis=-1)

# Ensure equal number of eye and mouth samples
min_len = min(len(eye_images), len(mouth_images))
eye_images = eye_images[:min_len]
eye_labels = eye_labels[:min_len]
mouth_images = mouth_images[:min_len]
mouth_labels = mouth_labels[:min_len]

# Train-test split
(X_eye_train, X_eye_test,
 X_mouth_train, X_mouth_test,
 y_eye_train, y_eye_test,
 y_yawn_train, y_yawn_test) = train_test_split(
    eye_images, mouth_images, eye_labels, mouth_labels,
    test_size=0.2, random_state=42
)

print("Eye Train shape:", X_eye_train.shape)
print("Mouth Train shape:", X_mouth_train.shape)
