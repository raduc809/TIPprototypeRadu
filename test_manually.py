# === Early suppression of warnings and logs ===
import warnings
import os

# Suppress TensorFlow logs and urllib3 warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO and WARNING logs
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

# Optional: hide all UserWarnings if needed
# warnings.simplefilter("ignore", UserWarning)

# === Imports ===
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from collections import Counter
import joblib
import tensorflow as tf

# Suppress TensorFlow logger messages
tf.get_logger().setLevel('ERROR')

# === Malware Labels ===
MALWARE_CLASSES = {
    1: "Ramnit",
    2: "Lollipop",
    3: "Kelihos_ver3",
    4: "Vundo",
    5: "Simda",
    6: "Tracur",
    7: "Kelihos_ver1",
    8: "Obfuscator.ACY",
    9: "Gatak"
}

MALWARE_DESCRIPTIONS = {
    1: "Trojan Downloader",
    2: "Adware",
    3: "Botnet Malware",
    4: "Spyware",
    5: "Worm",
    6: "Trojan Dropper",
    7: "Botnet Variant",
    8: "Obfuscation Tool",
    9: "Backdoor Trojan"
}

# === Load Trained Models ===
cnn_model = load_model("cnn_model.h5", compile=False)
rf_model = joblib.load("rf_model.pkl")

# === Convert .bytes File to Image Array ===
def bytes_to_image_array(bytes_path, size=(64, 64)):
    with open(bytes_path, 'r') as f:
        content = []
        for line in f:
            parts = line.strip().split()[1:]  # skip address
            content.extend([int(b, 16) if b != '??' else 0 for b in parts])
    array = np.array(content, dtype=np.uint8)
    padded_length = size[0] * size[1]
    if len(array) < padded_length:
        array = np.pad(array, (0, padded_length - len(array)))
    else:
        array = array[:padded_length]
    image = array.reshape(size)
    image = np.expand_dims(image, axis=(0, -1))  # shape (1, 64, 64, 1)
    return image / 255.0

# === Extract Features from .asm File ===
def extract_asm_features(asm_path):
    opcodes = ['mov', 'push', 'call', 'jmp', 'cmp', 'add', 'sub', 'lea', 'xor', 'test']
    sections = ['.text', '.data', '.rdata', '.idata']

    opcode_counts = Counter()
    section_flags = dict.fromkeys(sections, 0)

    with open(asm_path, 'r', errors='ignore') as f:
        for line in f:
            tokens = line.lower().split()
            for token in tokens:
                if token in opcodes:
                    opcode_counts[token] += 1
            for sec in sections:
                if sec in line:
                    section_flags[sec] = 1

    features = [opcode_counts.get(op, 0) for op in opcodes]
    features += [section_flags[sec] for sec in sections]

    column_names = opcodes + [f"section_{sec}" for sec in sections]
    return pd.DataFrame([features], columns=column_names)

# === Predict From .asm and .bytes Pair ===
def predict_from_files(sample_id):
    asm_file = f"{sample_id}.asm"
    bytes_file = f"{sample_id}.bytes"
    print(f"📥 Using files: {asm_file}, {bytes_file}")

    # Prepare features
    X_cnn = bytes_to_image_array(bytes_file)
    X_rf = extract_asm_features(asm_file)

    # Make predictions
    cnn_probs = cnn_model.predict(X_cnn)
    rf_probs = rf_model.predict_proba(X_rf)
    combined_probs = 0.5 * cnn_probs + 0.5 * rf_probs
    predicted_class = np.argmax(combined_probs, axis=1)[0] + 1  # 1-based class index

    # Show result
    print(f"✅ Predicted malware class: {predicted_class} ({MALWARE_CLASSES.get(predicted_class, 'Unknown')})")
    print(f"🧠 Malware type: {MALWARE_DESCRIPTIONS.get(predicted_class, 'Unknown Type')}")
    return predicted_class
