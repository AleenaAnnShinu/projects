import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (100, 100)

def extract_age_gender_from_filename(filename):
    parts = filename.split('_')
    age = int(parts[0])
    gender = int(parts[1])  # 0: Male, 1: Female
    return age, gender

def estimate_hair_length(image_array):
    gray = image_array.mean(axis=2)
    avg_brightness = gray.mean()
    return "long" if avg_brightness < 100 else "short"

def load_data(data_dir):
    X, y = [], []
    for file in os.listdir(data_dir):
        if not file.endswith(".jpg"):
            continue
        try:
            img = load_img(os.path.join(data_dir, file), target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0

            age, gender = extract_age_gender_from_filename(file)
            hair = estimate_hair_length(img_array)  # Fixed: use preprocessed array

            # Label: 0 = Female, 1 = Male
            label = 0 if gender == 0 else 1

            # Override if age between 20–30
            if 20 <= age <= 30:
                label = 0 if hair == "long" else 1

            X.append(img_array)
            y.append(label)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    return np.array(X), np.array(y)

# Load data
X, y = load_data(r"C:\Users\ALEEN\OneDrive\Desktop\Kaggle\Age and gender\utkface_dataset\UTKFace")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile & Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/hair_gender_model.h5")
