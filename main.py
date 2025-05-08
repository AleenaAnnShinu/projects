def preprocess_image(img_path):
    import cv2
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)




from tensorflow.keras.models import load_model
import numpy as np

nationality_model = load_model('models/nationality_model.h5')
emotion_model = load_model('models/emotion_model.h5')
age_model = load_model('models/age_model.h5')

nationality_labels = ['Indian', 'United States', 'African', 'Other']
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

def predict_all(img_path):
    img = preprocess_image(img_path)

    nat = nationality_model.predict(img)
    nationality = nationality_labels[np.argmax(nat)]

    emo = emotion_model.predict(img)
    emotion = emotion_labels[np.argmax(emo)]

    result = {'Nationality': nationality, 'Emotion': emotion}

    if nationality in ['Indian', 'United States']:
        age = age_model.predict(img)
        result['Age'] = age_labels[np.argmax(age)]

    return result




###GUI
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

def detect_dress_color(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (100, 100))
    img = img.reshape((-1, 3))
    colors, counts = np.unique(img, axis=0, return_counts=True)
    dom = colors[counts.argmax()]
    r, g, b = dom[2], dom[1], dom[0]

    if r > 150 and g < 100:
        return "Red"
    elif g > 150:
        return "Green"
    elif b > 150:
        return "Blue"
    return "Other"

root = tk.Tk()
root.title("Nationality Detection Model")

def upload():
    global img_path
    img_path = filedialog.askopenfilename()
    img = Image.open(img_path).resize((250, 250))
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img

def predict():
    res = predict_all(img_path)
    out = f"Nationality: {res['Nationality']}\nEmotion: {res['Emotion']}"
    if 'Age' in res:
        out += f"\nAge: {res['Age']}"
    if res['Nationality'] in ['Indian', 'African']:
        out += f"\nDress Color: {detect_dress_color(img_path)}"
    output_label.config(text=out)

tk.Button(root, text="Upload Image", command=upload).pack(pady=10)
img_label = tk.Label(root)
img_label.pack()
tk.Button(root, text="Predict", command=predict).pack(pady=10)
output_label = tk.Label(root, text="", font=('Arial', 14))
output_label.pack(pady=10)

root.mainloop()
