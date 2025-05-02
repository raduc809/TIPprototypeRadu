import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model

# === CONFIG ===
model_path = "cnn_model.h5"
image_folder = "cnn_images"
csv_path = "available_train_samples.csv"
output_path = "cnn_features.npy"

# === Load CNN model and feature extractor ===
cnn_model = load_model(model_path)
# Manually build model by calling it on dummy input
dummy_input = np.zeros((1, 64, 64, 1), dtype=np.float32)
cnn_model(dummy_input)  # triggers building

# Now you can extract intermediate layer output
from tensorflow.keras import Model
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("flatten").output)


# === Load sample list ===
df = pd.read_csv(csv_path)
X = []
ids = []

for _, row in df.iterrows():
    sample_id = row["Id"]
    npy_path = os.path.join(image_folder, f"{sample_id}.npy")

    if os.path.exists(npy_path):
        img = np.load(npy_path).astype("float32") / 255.0
        img = img.reshape(1, 64, 64, 1)
        features = feature_extractor.predict(img, verbose=0)
        X.append(features[0])
        ids.append(sample_id)

# === Save features
X = np.array(X)
np.save(output_path, X)
pd.DataFrame({"Id": ids}).to_csv("cnn_feature_ids.csv", index=False)

print(f"\n✅ Saved CNN features to: {output_path}")
