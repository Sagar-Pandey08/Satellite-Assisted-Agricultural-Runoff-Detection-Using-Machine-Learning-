# =====================================
# 05_refined_ndvi_ndwi.py
# =====================================
import numpy as np
import matplotlib.pyplot as plt
import joblib

def classify_npy_refined_in_batches(npy_file, model_path="rfmodel1.pkl", batch_size=25000):
    rf = joblib.load(model_path)
    bands = np.load(npy_file)
    H, W, _ = bands.shape
    flat = bands.reshape(-1, 4)
    preds = np.zeros(flat.shape[0], dtype=np.int32)

    for start in range(0, flat.shape[0], batch_size):
        end = min(start + batch_size, flat.shape[0])
        preds[start:end] = rf.predict(flat[start:end])
        print(f"Processed {end}/{flat.shape[0]} pixels...")

    preds = preds.reshape(H, W)

    # NDVI and NDWI
    B2, B3, B4, B8 = [bands[:,:,i].astype(float) for i in range(4)]
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    ndwi = (B3 - B8) / (B3 + B8 + 1e-6)

    cropland_mask = preds == 1
    water_mask = preds == 0

    ndvi_img = np.zeros((H, W, 3))
    ndvi_img[(cropland_mask) & (ndvi >= 0.30)] = [0, 1, 0]
    ndvi_img[(cropland_mask) & (ndvi < 0.30)] = [1, 1, 0]

    ndwi_img = np.zeros((H, W, 3))
    ndwi_img[(water_mask) & (ndwi >= 0.1)] = [0, 0, 0.5]
    ndwi_img[(water_mask) & (ndwi > 0) & (ndwi < 0.1)] = [0.68, 0.85, 0.9]

    return preds, bands, ndvi_img, ndwi_img
