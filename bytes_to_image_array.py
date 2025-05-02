import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image

# === Config ===
csv_path = "available_train_samples.csv"
data_folder = "train" 
output_folder = "cnn_images"
image_size = (64, 64)  # You can increase to (128, 128) for higher resolution



# === Setup ===
os.makedirs(output_folder, exist_ok=True)
df = pd.read_csv(csv_path)

# === Convert bytes to grayscale image ===
def bytes_to_image_array(file_path, size=(64, 64)):
    pixels = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()[1:]  # skip memory address
            for p in parts:
                if p == '??':
                    pixels.append(0)
                else:
                    pixels.append(int(p, 16))  # convert hex to int

    # Pad or truncate to fixed size
    total_pixels = size[0] * size[1]
    if len(pixels) < total_pixels:
        pixels.extend([0] * (total_pixels - len(pixels)))
    else:
        pixels = pixels[:total_pixels]

    img = np.array(pixels, dtype=np.uint8).reshape(size)
    return img

# === Process each sample ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    sample_id = row["Id"]
    bytes_path = os.path.join(data_folder, f"{sample_id}.bytes")

    if not os.path.exists(bytes_path):
        continue

    try:
        img_array = bytes_to_image_array(bytes_path, image_size)
        np.save(os.path.join(output_folder, f"{sample_id}.npy"), img_array)
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
