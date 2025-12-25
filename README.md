# Urban Green Space Monitoring 🌳🏙️

## 📌 Overview
**Urban Green Space Monitoring** is a geospatial analysis project that leverages **remote sensing** and **machine learning** to quantify and visualize land use and land cover (LULC) changes in **Gurgaon, India** from **2015 to 2023**.  
Using **Landsat 8** and **Sentinel-2** satellite imagery, we classify urban, vegetation, water, and barren land categories and analyze urban expansion patterns, vegetation resilience, and water body reduction over time.

---

## 🎯 Objectives
- Quantify and analyze **urban expansion patterns** in Gurgaon between 2015, 2020, and 2023.
- Detect changes in **green space distribution** and density.
- Perform **LULC classification** using Random Forest.
- Conduct **change detection** and identify **urban-green transition zones**.
- Provide an **open-source, reproducible workflow** for similar studies in other cities.

---

## 🛰️ Data Sources
- **Landsat 8** (2015) – [USGS Earth Explorer](https://earthexplorer.usgs.gov/)  
- **Sentinel-2** (2020, 2023) – [Copernicus Open Access Hub](https://scihub.copernicus.eu/)  
- Pre-monsoon (March) imagery with **<5% cloud cover**.

---

## ⚙️ Methodology
1. **Data Acquisition & Preprocessing**
   - Download pre-monsoon surface reflectance imagery.
   - Clip to Gurgaon boundaries and reproject to EPSG:4326.

2. **Band & Index Selection**
   - Blue, Green, Red, NIR, SWIR1.
   - NDVI (Vegetation), NDBI (Built-up), NDWI (Water).

3. **Training Data Collection**
   - Digitized training points in Google Earth Engine.
   - Extracted spectral values for each LULC class.

4. **Supervised Classification**
   - Random Forest with stratified 5-fold cross-validation.
   - Performance metrics: Accuracy, Cohen’s Kappa, AUC.

5. **Change Detection**
   - Transition matrices for 2015–2020, 2020–2023, 2015–2023.
   - Urban expansion quantified from Vegetation → Urban conversions.

6. **Ambiguity Analysis**
   - **T-I-F Analysis** for mixed urban-vegetation zones.
   - **Texture Analysis** for spatial heterogeneity detection.

---

## 📊 Key Results
- **Urban area** increased by **40%** between 2015–2023.
- **Vegetation** showed a **3.6% net increase**, though recent decline observed.
- **Water bodies** decreased by **26%**.
- **Barren land** reduced by **66%**, largely converted to urban areas.
- Model achieved **94% overall accuracy** and **0.87 Cohen’s Kappa**.

---

## 🛠️ Tech Stack
- **Programming:** Python, JavaScript (Google Earth Engine)
- **Libraries:** `Rasterio`, `Scikit-learn`, `Seaborn`, `NumPy`, `Pandas`
- **Platform:** Google Earth Engine, Jupyter Notebook
- **Visualization:** Matplotlib, Seaborn, QGIS

---

## 🚀 How to Run
1. **Clone the Repository**

- git clone https://github.com/Piyush94G/urban-green-space-monitoring.git
- cd urban-green-space-monitoring

2. **Install Dependencies**

- pip install -r requirements.txt

3. **Run Notebooks/Scripts**

- Use provided Jupyter notebooks or Python scripts to replicate preprocessing, classification, and analysis.

---

## 📌 Applications
- Urban planning and policy-making.
- Environmental impact assessment.
- Academic research in remote sensing and GIS.
- Monitoring sustainable development goals (SDG 11: Sustainable Cities and Communities).

## 🔗 Data Access

- Due to large size, datasets are hosted externally:
 https://drive.google.com/drive/folders/1bjK9qirlfrMchFjoC6EEnVWbwDSWP3u4
