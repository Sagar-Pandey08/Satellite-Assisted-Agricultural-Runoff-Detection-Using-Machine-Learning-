# =====================================
# 01_dataset_builder.py
# =====================================
import os
import numpy as np
import rasterio

# ==== PATHS ====
BANDS_DIR = "/kaggle/input/newagri2/NewAgri2"         # .npy files
LABELS_DIR = "/kaggle/input/tifflabels/TIFF_LABELS"   # .tif files

# ==== LABEL MAPPING ====
dw_map = {
    0: 0, 1: 3, 2: 3, 3: 3, 4: 1, 5: 6,
    6: 4, 7: 6, 8: 4, 9: 2
}
wc_map = {
    10: 3, 20: 3, 30: 3, 40: 1, 50: 6,
    60: 2, 70: 3, 80: 0, 90: 5, 95: 5
}

# ==== SETTINGS ====
MAX_SAMPLES_PER_FILE = 50_000

# ==== BUILD DATASET ====
X, y = [], []

for f in os.listdir(BANDS_DIR):
    if not f.endswith(".npy"):
        continue

    base = f.replace(".npy", "")
    npy_path = os.path.join(BANDS_DIR, f)
    dw_tif = os.path.join(LABELS_DIR, base + "_dw.tif")
    wc_tif = os.path.join(LABELS_DIR, base + "_wc.tif")

    if not os.path.exists(dw_tif) or not os.path.exists(wc_tif):
        print("⚠️ Missing labels for", f)
        continue

    # Load data
    bands = np.load(npy_path)
    H, W, _ = bands.shape

    with rasterio.open(dw_tif) as src:
        dw_labels = src.read(1, out_shape=(H, W))
    with rasterio.open(wc_tif) as src:
        wc_labels = src.read(1, out_shape=(H, W))

    unified = np.full((H, W), -1)

    for wc_val, mapped in wc_map.items():
        unified[wc_labels == wc_val] = mapped
    for dw_val, mapped in dw_map.items():
        mask = (unified == -1) & (dw_labels == dw_val)
        unified[mask] = mapped

    mask = unified != -1
    features = bands[mask]
    labels = unified[mask]

    if features.shape[0] > MAX_SAMPLES_PER_FILE:
        idx = np.random.choice(features.shape[0], MAX_SAMPLES_PER_FILE, replace=False)
        features = features[idx]
        labels = labels[idx]

    X.append(features)
    y.append(labels)
    print(f"✅ Processed {f}: {features.shape[0]} samples")

X = np.vstack(X)
y = np.hstack(y)

np.save("X_dataset_small1.npy", X)
np.save("y_dataset_small1.npy", y)
print("✅ Dataset saved to disk:", X.shape, y.shape)
