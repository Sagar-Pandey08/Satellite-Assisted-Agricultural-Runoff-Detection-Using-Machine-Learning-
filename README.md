# Heading 1
Satellite-Assisted Agricultural Runoff Detection Using Machine Learning
## Heading 2
Project Overview

Agricultural runoff — the flow of excess fertilizers, pesticides, and sediments from farmlands — is one of the leading causes of water pollution and soil degradation.
This project leverages satellite remote sensing, GIS-based data preprocessing, and machine learning to detect and analyze runoff zones in agricultural regions.

By using multi-spectral satellite bands and land cover classification models, the project identifies areas with high runoff risk based on vegetation, soil, and water reflectance patterns.

## Heading 2
Objectives

🌍 Detect agricultural runoff zones using multi-spectral satellite imagery.

📡 Integrate Remote Sensing (Sentinel-2 / Landsat) and GIS data layers.

🤖 Train a Random Forest model for land cover classification.

💧 Generate NDVI and NDWI maps to assess crop stress and water overflow.

🛰️ Support sustainable Integrated Water Resource Management (IWRM) practices.

Folder Structure 
Satellite-Assisted-Agricultural-Runoff-Detection-Using-Machine-Learning/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── 01_dataset_builder.py
│   ├── 02_train_random_forest.py
│   ├── 03_evaluate_model.py
│   ├── 04_predict_batches.py
│   ├── 05_refined_ndvi_ndwi.py
│   └── 06_run_demo.py
│
├── data/
│   ├── bands/        ← satellite band arrays (.npy)
│   ├── labels/       ← land cover label maps (.tif)
│   └── testbands/    ← test sample for inference
│
├── models/
│   └── rfmodel1.pkl  ← trained Random Forest model
│
└── outputs/
    ├── X_dataset_small1.npy
    ├── y_dataset_small1.npy
    ├── confusion_matrix.png
    └── runoff_risk_map.png

Requirements 

numpy
pandas
rasterio
matplotlib
seaborn
scikit-learn
joblib

💧 Runoff Detection & Analysis

The model uses spectral indices to detect potential runoff-prone zones:

NDVI (Normalized Difference Vegetation Index)
→ Detects vegetation stress using NIR and Red bands.

NDVI < 0.3 → weak vegetation → potential runoff area.

NDWI (Normalized Difference Water Index)
→ Identifies surface water and flooded zones using Green and NIR bands.

NDWI > 0.2 → excess surface water → high runoff probability.

🖼️ Visualization Results

👨‍💻 Author

Sagar Pandey
🎓 B.Tech (AI & ML)

]
