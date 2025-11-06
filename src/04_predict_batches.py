# =====================================
# 04_predict_batches.py
# =====================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import joblib

def classify_npy_in_batches(npy_file, model_path="rfmodel1.pkl", batch_size=25000):
    rf = joblib.load(model_path)
    bands = np.load(npy_file)
    H, W, _ = bands.shape
    flat = bands.reshape(-1, 4)
    preds = np.zeros(flat.shape[0], dtype=np.int32)

    for start in range(0, flat.shape[0], batch_size):
        end = min(start + batch_size, flat.shape[0])
        preds[start:end] = rf.predict(flat[start:end])
        print(f"Processed {end}/{flat.shape[0]} pixels...")

    labels = preds.reshape(H, W)
    return labels, bands

CLASS_COLORS = {
    0: "#0000FF", 1: "#FFC0CB", 2: "#8B4513", 3: "#00FF00",
    4: "#228B22", 5: "#006400", 6: "#000000"
}
cmap = ListedColormap([CLASS_COLORS[i] for i in range(len(CLASS_COLORS))])

def plot_results(labels, bands):
    plt.figure(figsize=(12,6))
    rgb = np.stack([bands[:,:,2], bands[:,:,1], bands[:,:,0]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    plt.subplot(1,2,1)
    plt.imshow(rgb)
    plt.title("Original RGB Image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(labels, cmap=cmap)
    plt.title("Classified Land Cover Map")
    plt.axis("off")
    plt.show()
