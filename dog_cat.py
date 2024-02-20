import os
import numpy as np
from sklearn.svm import SVC
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
import pandas as pd  # Importing pandas for DataFrame manipulation

# Function to load and preprocess images using VGG-16 preprocessing
def load_images(directory, target_size=(224, 224), batch_size=32):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            try:
                img = image.load_img(image_path, target_size=target_size)
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                images.append(img)
                label = 0 if "cat" in filename else 1  # Assuming filenames contain "cat" or "dog"
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        if len(images) >= batch_size:
            yield np.vstack(images), np.array(labels)
            images = []
            labels = []
    if images:  # Process remaining images
        yield np.vstack(images), np.array(labels)

# Load pre-trained VGG-16 model without top layer (include_top=False)
base_model = VGG16(weights='imagenet', include_top=False)

# Function to extract features using VGG-16
def extract_features(images):
    features = base_model.predict(images)
    return features.reshape(features.shape[0], -1)

# Load training images and labels
train_dir = r"C:\Users\Dell.com\Desktop\train"
train_features, train_labels = [], []
for batch_images, batch_labels in load_images(train_dir):
    train_features.append(extract_features(batch_images))
    train_labels.append(batch_labels)
X_train = np.vstack(train_features)
y_train = np.concatenate(train_labels)

# Load test images
test_dir = r"C:\Users\Dell.com\Desktop\test1"
test_features, test_labels = [], []
for batch_images, batch_labels in load_images(test_dir):
    test_features.append(extract_features(batch_images))
    test_labels.append(batch_labels)
test_features = np.vstack(test_features)
test_labels = np.concatenate(test_labels)

# Initialize SVM model
svm_model = SVC(kernel='linear')

# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
test_predictions = svm_model.predict(test_features)

# Save predictions to a CSV file for submission
submission_df = pd.DataFrame({'Id': os.listdir(test_dir), 'label': test_predictions})
submission_df.to_csv("submission.csv", index=False)
