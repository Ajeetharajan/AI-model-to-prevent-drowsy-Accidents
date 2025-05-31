import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load preprocessed mouth image data
X_mouth = np.load("X_mouth.npy")  # Features: mouth images
y_mouth = np.load("y_mouth.npy")  # Labels: 0 = no yawn, 1 = yawn

# Ensure shape is (batch_size, height, width, 1)
if len(X_mouth.shape) == 3:
    X_mouth = np.expand_dims(X_mouth, axis=-1)

X_mouth = X_mouth.reshape(-1, 64, 64, 1).astype(np.float32)
X_mouth /= 255.0  # Normalize

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_mouth, y_mouth, test_size=0.2, random_state=42, stratify=y_mouth)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Define CNN model for mouth
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classifier: yawn or not
])

# Compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_test, y_test),
          epochs=50,
          callbacks=[early_stopping])

# Save the model
model.save("yawn_model.h5")
print("Yawn detection model training complete! Saved as yawn_model.h5")
