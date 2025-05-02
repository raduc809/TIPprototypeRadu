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

# Merge labels if needed
if "Class" not in asm_df.columns:
    asm_df = asm_df.merge(cnn_labels, on="Id")

# Separate features and true labels
X_rf = asm_df.drop(columns=["Id", "Class"])
y_true = asm_df["Class"].values

# === Load CNN image arrays (from .npy files) ===
def load_npy_image(image_id):
    path = f"cnn_images/{image_id}.npy"
    return np.load(path)

X_cnn = np.array([load_npy_image(id_) for id_ in asm_df["Id"]])

# === Predict ===
cnn_probs = cnn_model.predict(X_cnn)
rf_probs = rf_model.predict_proba(X_rf)

# === Re-weighted Confidence Fusion ===
cnn_weight = 0.4  # CNN gets 40%
rf_weight = 0.6   # RF gets 60%
combined_probs = cnn_weight * cnn_probs + rf_weight * rf_probs

# === Apply Confidence Threshold (optional bonus) ===
max_probs = np.max(combined_probs, axis=1)
y_pred = np.argmax(combined_probs, axis=1) + 1  # Class labels are 1-based

# Backup with RF predictions if low confidence (optional)
fallback_rf = np.argmax(rf_probs, axis=1) + 1
threshold = 0.5
low_confidence = max_probs < threshold
y_pred[low_confidence] = fallback_rf[low_confidence]

# === Evaluate ===
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Improved Hybrid Model Accuracy: {accuracy:.4f}")

# === Save Predictions (Optional) ===
predictions_df = pd.DataFrame({"Id": asm_df["Id"], "True_Class": y_true, "Predicted_Class": y_pred})
predictions_df.to_csv("hybrid_predictions.csv", index=False)
print("\n📄 Saved predictions to hybrid_predictions.csv")
