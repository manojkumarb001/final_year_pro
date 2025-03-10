import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set dataset path (Fix Windows path issue)
DATASET_PATH = r"D:\Indian"
IMG_SIZE = 64  # Resize images to 64x64

# Initialize lists for images (X) and labels (y)
X, y = [], []
labels = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])

print(f"📂 Found {len(labels)} Classes: {labels}")

# Loop through each sign label folder
for label_idx, label in enumerate(labels):
    class_path = os.path.join(DATASET_PATH, label)
    
    # Skip if the folder is empty
    if not os.listdir(class_path):
        print(f"⚠️ Skipping empty folder: {class_path}")
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Check if it's an image file
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  
            continue  

        # Read and process image
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Skipping unreadable image: {img_path}")
            continue
        
        # Convert grayscale images to RGB
        if len(image.shape) == 2 or image.shape[-1] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize to 64x64
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        X.append(image)
        y.append(label_idx)  # Convert label to numerical index

# Check if dataset is empty before proceeding
if len(X) == 0:
    raise ValueError("🚨 No valid images found! Check dataset path and file formats.")

# Convert lists to NumPy arrays
X = np.array(X, dtype=np.float32) / 255.0  # Normalize pixel values (0 to 1)
y = to_categorical(np.array(y), num_classes=len(labels))  # One-hot encode labels

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Print dataset details
print(f"\n✅ Dataset Loaded Successfully!")
print(f"🔹 {len(X_train)} Training Samples, {len(X_test)} Testing Samples")
print(f"🔹 Image Shape: {X_train.shape}, Labels Shape: {y_train.shape}")
np.save("X.npy", X)  # Save image data
np.save("y.npy", y)  # Save labels
