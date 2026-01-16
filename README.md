# 🩺 Breast Cancer Classification using Machine Learning

## 📌 Project Overview
This project implements a **Breast Cancer Classification system** using **Logistic Regression** to predict whether a breast tumor is **Benign** or **Malignant**.  
The model is trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains features extracted from digitized images of **Fine Needle Aspirate (FNA)** of breast masses.

---

## 🧠 Machine Learning Model
- **Algorithm:** Logistic Regression  
- **Problem Type:** Binary Classification  
- **Target Labels:**
  - `0` → Malignant  
  - `1` → Benign  

---

## 📂 Dataset Information

### 🔗 Dataset Source
- **UCI Machine Learning Repository:**  
  https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29  


### 📊 Dataset Description
The dataset contains features computed from digitized images of breast mass cell nuclei obtained via Fine Needle Aspirate (FNA).  
These features describe characteristics such as shape, texture, and structure of the cell nuclei.

The dataset is based on research by:  
**K. P. Bennett and O. L. Mangasarian (1992)**  
*"Robust Linear Programming Discrimination of Two Linearly Inseparable Sets"*

---

## 🧾 Attribute Information

### Columns
1. **ID Number**  
2. **Diagnosis**
 - `M` = Malignant  
 - `B` = Benign
**30 real-valued features**

### 🧬 Feature Description
For each cell nucleus, **10 base features** are computed:

- Radius  
- Texture  
- Perimeter  
- Area  
- Smoothness  
- Compactness  
- Concavity  
- Concave Points  
- Symmetry  
- Fractal Dimension  

For each feature, the following statistics are calculated:
- **Mean**
- **Standard Error**
- **Worst (largest values)**  

➡️ Resulting in **30 total numerical features per sample**.

---

## 📈 Dataset Statistics
- **Total Samples:** 569  
- **Benign:** 357  
- **Malignant:** 212  
- **Missing Values:** None  

---

## 🛠️ Libraries Used
```python
import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## ⚙️ Project Workflow

1. Load the dataset  
2. Data preprocessing  
3. Train-test split  
4. Train Logistic Regression model  
5. Evaluate model performance  
6. Make predictions on new data  

---

## ✅ Model Performance

### 📊 Accuracy Results
- **Accuracy on Training Data:** `94.94%`  
- **Accuracy on Test Data:** `92.98%`

## 🔍 Sample Prediction

### Input Data
```python
input_data = (
  20.29,14.34,135.1,1297,0.1003,0.1328,0.198,0.1043,0.1809,0.05883,
  0.7572,0.7813,5.438,94.44,0.01149,0.02461,0.05688,0.01885,0.01756,0.005115,
  22.54,16.67,152.2,1575,0.1374,0.205,0.4,0.1625,0.2364,0.07678
)
```

### Prediction Output
```
The Breast cancer is Malignant
```

## 🚀 Future Improvements

- Implement advanced classifiers (SVM, Random Forest, XGBoost)

- Perform feature scaling and normalization

- Use cross-validation for better evaluation

- Deploy the model using Flask or Streamlit

## 📜 License

This project is intended for educational purpose using publicly available data.
