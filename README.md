# 🧠 Machine Learning Analysis on Credit Default Dataset

A comprehensive machine learning project applying regression, clustering, and classification techniques to analyze and predict credit card default behavior using a real-world dataset of 30,000+ clients.

---

## 📁 Project Overview

This project uses the "Default of Credit Card Clients" dataset to demonstrate the application of multiple machine learning algorithms across three major categories:

- **Regression**: Predicting payment amounts using Random Forest Regressor  
- **Clustering**: Discovering customer segments using Mean Shift and BIRCH  
- **Classification**: Predicting credit default using Naive Bayes and Support Vector Machine (SVM)

The pipeline includes data preprocessing, feature engineering, model training, evaluation, and visualizations.

---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository  
- **Records**: 30,000 credit card clients  
- **Features**: Demographics, bill amounts, payment history, credit limits, etc.  
- **Target Variables**:  
  - `default payment next month` (classification)  
  - `PAY_AMT1` (regression target example)

---

## 🛠️ Technologies Used

- **Language**: Python 3  
- **Libraries**:  
  - `pandas`, `numpy` for data manipulation  
  - `scikit-learn` for ML algorithms  
  - `matplotlib`, `seaborn` for data visualization  
  - `PCA` for dimensionality reduction (visual clustering)

---

## 📌 Machine Learning Models

### ✅ Regression
- **Random Forest Regressor**  
  - Predicts payment amounts
  - Evaluated using RMSE

### ✅ Clustering
- **Mean Shift**
- **BIRCH**  
  - Visualized using PCA (2D projection)

### ✅ Classification
- **Naive Bayes**
- **Support Vector Machine (SVM)**  
  - Evaluated using accuracy, precision, recall, F1-score

---

## 📈 Visualizations

- PCA 2D plots for clustering
- Confusion matrix & classification reports
- RMSE value for regression model
