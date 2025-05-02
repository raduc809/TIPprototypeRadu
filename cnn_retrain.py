import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# === Load Data ===
df = pd.read_csv("available_train_samples.csv")
image_folder = "cnn_images"
image_size = (64, 64)

X, y = [], []
for _, row in df.iterrows():
    path = os.path.join(image_folder, f"{row['Id']}.npy")
    if os.path.exists(path):
        img = np.load(path).astype("float32") / 255.0
        X.append(img)
        y.append(row["Class"])

X = np.array(X).reshape(-1, image_size[0], image_size[1], 1)
y = to_categorical(np.array(y) - 1, num_classes=9)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Define CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(9, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Train CNN ===
model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test))

# === Save model for fusion ===
model.save("cnn_model.h5")
