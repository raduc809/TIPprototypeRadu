import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# === Load Models ===
cnn_model = load_model("cnn_model.h5")
rf_model = joblib.load("rf_model.pkl")

# === Load Data ===
asm_df = pd.read_csv("asm_features.csv")
cnn_labels = pd.read_csv("available_train_samples.csv")

# Check if "Class" exists in asm_df; if not, merge it from labels
if "Class" not in asm_df.columns:
    asm_df = asm_df.merge(cnn_labels, on="Id")

# Now it's safe to drop
X_rf = asm_df.drop(columns=["Id", "Class"])
y_true = asm_df["Class"].values

# === Load CNN images from .npy ===
def load_image(image_id):
    path = f"cnn_images/{image_id}.npy"
    return np.load(path)

X_cnn = np.array([load_image(id_) for id_ in asm_df["Id"]])

# === Predict ===
cnn_probs = cnn_model.predict(X_cnn)
rf_probs = rf_model.predict_proba(X_rf)

# === Confidence Fusion ===
cnn_weight = 0.5
rf_weight = 0.5
combined_probs = cnn_weight * cnn_probs + rf_weight * rf_probs
y_pred = np.argmax(combined_probs, axis=1) + 1  # Class labels are 1-based

# === Evaluate ===
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Hybrid Model Accuracy: {accuracy:.4f}")
