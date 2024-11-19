import os
import numpy as np
from PIL import Image

# Create a directory for the dataset
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Generate 100 objects with 72 images each (dummy data)
for obj in range(1, 101):
    obj_dir = os.path.join(dataset_dir, f"object_{obj}")
    os.makedirs(obj_dir, exist_ok=True)
    
    for img_id in range(1, 73):
        # Create a random RGB image of size 224x224
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img_path = os.path.join(obj_dir, f"image_{img_id}.png")
        Image.fromarray(img).save(img_path)

print(f"Dataset generated at: {dataset_dir}")
