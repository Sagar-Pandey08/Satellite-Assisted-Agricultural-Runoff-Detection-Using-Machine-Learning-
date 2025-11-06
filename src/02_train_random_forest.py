# =====================================
# 02_train_random_forest.py
# =====================================
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
X = np.load("X_dataset_small1.npy")
y = np.load("y_dataset_small1.npy")
print("Dataset loaded:", X.shape, y.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Save model
joblib.dump(rf, "rfmodel1.pkl")
print("✅ Model trained and saved")
