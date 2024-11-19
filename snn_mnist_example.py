import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up paths
dataset_dir = "dataset"
image_size = (224, 224)
batch_size = 32

# Data Augmentation and Loading
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    dataset_dir, target_size=image_size, batch_size=batch_size, class_mode="categorical", subset="training"
)

val_gen = datagen.flow_from_directory(
    dataset_dir, target_size=image_size, batch_size=batch_size, class_mode="categorical", subset="validation"
)

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')  # Number of classes = number of objects
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save the model
model.save("trained_model.h5")
print("Model trained and saved as trained_model.h5")
