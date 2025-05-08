from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
### from utils.preprocessing import load_utkface_data
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

# Parse UTKFace filename
def parse_filename(filename):
    parts = filename.split('_')
    age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
    return age, gender, race

def load_utkface_data(dataset_path, target='age', image_size=(224, 224), max_items=10000):
    X, y = [], []
    for i, file in enumerate(os.listdir(dataset_path)):
        if not file.endswith('.jpg'):
            continue
        if i >= max_items:
            break
        try:
            age, gender, race = parse_filename(file)
            img_path = os.path.join(dataset_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            img = img / 255.0
            X.append(img)

            if target == 'age':
                y.append(age // 10)  # Grouped by decade (0-9 classes)
            elif target == 'emotion':
                y.append(age % 7)  # Synthetic: 0-6 as emotion (not perfect)
            elif target == 'nationality':
                if race == 0:        # White
                    y.append(1)      # USA
                elif race == 2:      # Asian
                    y.append(0)      # Indian
                elif race == 3:      # Others
                    y.append(3)      # Other
                else:                # Black etc
                    y.append(2)      # African
        except Exception as e:
            continue

    return np.array(X), to_categorical(np.array(y))

###AGE
X, y = load_utkface_data(r'C:\Users\ALEEN\OneDrive\Desktop\Kaggle\Age and gender\utkface_dataset\UTKFace', target='age')




base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
out = Dense(y.shape[1], activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)


os.makedirs(r"C:\Users\ALEEN\OneDrive\Desktop\Kaggle\Nationality detection\models", exist_ok=True)
model.save(r'C:\Users\ALEEN\OneDrive\Desktop\Kaggle\Nationality detection\models\age_model.h5')

###EMOTION
# Uses fake emotion mapping from age % 7 (for demo)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

X, y = load_utkface_data(r'C:\Users\ALEEN\OneDrive\Desktop\Kaggle\Age and gender\utkface_dataset\UTKFace', target='emotion')




base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
out = Dense(y.shape[1], activation='softmax')(x)



model = Model(inputs=base.input, outputs=out)
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

os.makedirs(r"C:\Users\ALEEN\OneDrive\Desktop\Kaggle\Nationality detection\models", exist_ok=True)
model.save(r'C:\Users\ALEEN\OneDrive\Desktop\Kaggle\Nationality detection\models\emotion_model.h5')


###NATIONALITY

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

X, y = load_utkface_data(r'C:\Users\ALEEN\OneDrive\Desktop\Kaggle\Age and gender\utkface_dataset\UTKFace', target='nationality')


base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
out = Dense(y.shape[1], activation='softmax')(x)


model = Model(inputs=base.input, outputs=out)
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)


os.makedirs(r"C:\Users\ALEEN\OneDrive\Desktop\Kaggle\Nationality detection\models", exist_ok=True)
model.save(r'C:\Users\ALEEN\OneDrive\Desktop\Kaggle\Nationality detection\models\nationality_model.h5')
