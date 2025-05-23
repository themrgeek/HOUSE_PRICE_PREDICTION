# 🏠 House Price Prediction in Ontario, Canada

**A final project for CPSC_5616EL_02 (Machine Learning / Deep Learning) at Laurentian University - Winter 2025**

---

## 📊 Project Overview

This project aims to predict house prices across Ontario, Canada, using various Machine Learning and Deep Learning models. The objective is to analyze, model, and evaluate the performance of these models to offer accurate and interpretable predictions valuable to real estate professionals, policymakers, and investors.

### 🔍 Models Implemented
- Support Vector Regression (SVR)
- Decision Tree Regressor (DT)
- Random Forest Regressor (RF)
- Artificial Neural Network (ANN)
- *(LSTM model support was added but not used due to dataset constraints)*

---

## 🗂️ Directory Structure

<img width="651" alt="Screenshot 2025-05-23 at 3 13 17 PM" src="https://github.com/user-attachments/assets/eee56a3b-311c-458e-8e28-c76fe0483113" />

---

## 📁 Dataset

- **Source**: Originally from Kaggle (now private)
- **Scraped from**: [Zoocasa Toronto Real Estate](https://www.zoocasa.com/toronto-on-real-estate)
- **Backup**: [GitHub Mirror](https://github.com/slavaspirin/Toronto-housing-price-prediction/blob/master/data.xlsx)

### 📌 Features
- **Numerical**: Bedrooms, Bathrooms, Square Footage, List Price
- **Categorical**: Property Type, Address, Description (One-hot encoded)

**Note**: Missing values were handled using imputation techniques, and numerical features were normalized for consistency.

---

## 🧠 Techniques

- Regression modeling on structured real estate data
- One-hot encoding for categorical variables
- Data normalization and cleaning
- Evaluation metrics:
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - R² (Coefficient of Determination)
  - Accuracy thresholds at ±5% and ±10%
- Visualization tools:
  - Prediction scatter plots
  - Error distribution boxplots
  - Feature importance heatmaps

---

## ✅ Results Summary

<img width="885" alt="Screenshot 2025-05-23 at 3 17 20 PM" src="https://github.com/user-attachments/assets/14839015-b886-497d-9de3-c56b0288d28b" />


> **Conclusion**: SVM delivered the best balance of accuracy and practicality. RF offered interpretability and decent error margins. ANN struggled with R², and DT suffered from overfitting.

---

## 🧪 How to Run

1. Install **MATLAB** with the **Deep Learning Toolbox**.
2. Place your dataset file `data.xlsx` in the root project directory.
3. Open MATLAB and run the following in the command window:

```matlab
