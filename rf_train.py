import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === Load features ===
df = pd.read_csv("asm_features.csv")

# === Prepare feature matrix and labels ===
X = df.drop(columns=["Id", "Class"])
y = df["Class"]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# === Evaluate model ===
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ RF Accuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred))

# === Feature importance (top 10) ===
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_feats = importances.sort_values(ascending=False).head(10)

print("\n🔍 Top Features:\n")
print(top_feats)

# === Save model for reuse ===
joblib.dump(rf, "rf_model.pkl")
print("\n💾 RF model saved to rf_model.pkl")
