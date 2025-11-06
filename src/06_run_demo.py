# =====================================
# 06_run_demo.py
# =====================================
import matplotlib.pyplot as plt
from 05_refined_ndvi_ndwi import classify_npy_refined_in_batches

TEST_FILE = "/kaggle/input/testbands/bands_2023-11-19_0_0.2.npy"
MODEL_PATH = "rfmodel1.pkl"

labels, bands, ndvi_img, ndwi_img = classify_npy_refined_in_batches(TEST_FILE, model_path=MODEL_PATH, batch_size=50000)

# Plot results
plt.figure(figsize=(18,6))
rgb = (bands[:,:,:3] - bands[:,:,:3].min()) / (bands[:,:,:3].max() - bands[:,:,:3].min())

plt.subplot(1,3,1)
plt.imshow(rgb)
plt.title("Original RGB Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(ndvi_img)
plt.title("NDVI (Green=Healthy, Yellow=Stressed)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(ndwi_img)
plt.title("NDWI (Dark Blue=Water, Light Blue=Runoff)")
plt.axis("off")

plt.show()

print("✅ Batch-safe NDVI & NDWI refinement complete!")
