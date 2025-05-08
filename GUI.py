import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2

def extract_age_gender_from_filename(filename):
    parts = filename.split('_')
    age = int(parts[0])
    gender = int(parts[1])  # 0: Male, 1: Female
    return age, gender

def estimate_hair_length(image_array):
    gray = image_array.mean(axis=2)
    avg_brightness = gray.mean()
    return "long" if avg_brightness < 100 else "short"

model = load_model("C:/Users/ALEEN/OneDrive/Desktop/Kaggle/Hair detection/model/hair_gender_model.h5")


def predict(image_path):
    img = load_img(image_path, target_size=(100,100))
    img_array = img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    filename = image_path.split('/')[-1]
    age, gender_actual = extract_age_gender_from_filename(filename)
    hair = estimate_hair_length(np.array(img))

    if 20 <= age <= 30:
        predicted = "Female" if hair == "long" else "Male"
    else:
        prob = model.predict(img_input)[0][0]
        predicted = "Male" if prob > 0.5 else "Female"
    
    return f"Age: {age} | Hair: {hair} | Predicted: {predicted}"

def open_file():
    file_path = filedialog.askopenfilename()
    result = predict(file_path)
    result_label.config(text=result)

app = tk.Tk()
app.title("Hair-Length Gender Detector")

btn = tk.Button(app, text="Upload Image", command=open_file)
btn.pack(pady=10)

result_label = tk.Label(app, text="Upload an image to start", font=("Arial", 14))
result_label.pack(pady=20)

app.mainloop()
