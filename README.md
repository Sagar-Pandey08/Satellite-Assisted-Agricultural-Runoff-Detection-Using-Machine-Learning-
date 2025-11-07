
# Satellite-Assisted Agricultural Runoff Detection Using Machine Learning

## Project Overview

Agricultural runoff means the flow of excess fertilizers, pesticides, and sediments from farmlands , is one of the leading causes of water pollution and soil degradation.
This project leverages satellite remote sensing, GIS-based data preprocessing, and machine learning to detect and analyze runoff zones in agricultural regions.
By using multi-spectral satellite bands and land cover classification models, the project identifies areas with high runoff risk based on vegetation, soil, and water reflectance patterns.


## Objectives

🌍 Detect agricultural runoff zones using multi-spectral satellite imagery.

📡 Integrate Remote Sensing (Sentinel-2 / Landsat) and GIS data layers.

🤖 Train a Random Forest model for land cover classification.

💧 Generate NDVI and NDWI maps to assess crop stress and water overflow.

🛰️ Support sustainable Integrated Water Resource Management (IWRM) practices.


## Requirements 
-numpy

-pandas

-rasterio

-matplotlib

-seaborn

-scikit-learn

-joblib

## Runoff Detection & Analysis

The model uses spectral indices to detect potential runoff-prone zones:

-NDVI (Normalized Difference Vegetation Index)
→ Detects vegetation stress using NIR and Red bands.

-NDVI < 0.3 → weak vegetation → potential runoff area.

-NDWI (Normalized Difference Water Index)
→ Identifies surface water and flooded zones using Green and NIR bands.

-NDWI > 0.2 → excess surface water → high runoff probability.

## Visualization Results

<img width="944" height="350" alt="Screenshot 2025-11-07 072616" src="https://github.com/user-attachments/assets/ca3f70df-2250-4de9-acfc-dea79301b541" />
<img width="1263" height="350" alt="Screenshot 2025-11-07 072629" src="https://github.com/user-attachments/assets/2e542218-8aa0-4ba5-807e-782ee26a21ef" />



## Author

Sagar Pandey
🎓 B.Tech (AI & ML)


