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

## 📊 Algorithm Comparison

This project applied five machine learning algorithms spanning **regression**, **classification**, and **unsupervised clustering** to a real-world credit card default dataset with over 30,000 records.

| Algorithm               | Type          | Metric Used        | Strengths                                        | Limitations                                          |
|------------------------|---------------|--------------------|--------------------------------------------------|------------------------------------------------------|
| Random Forest Regressor| Regression     | RMSE               | Handles nonlinear relationships; good accuracy   | Harder to compare RMSE to classification scores      |
| Naive Bayes            | Classification | F1 Score           | Fast, handles high-dimensional data              | Assumes feature independence                         |
| Support Vector Machine | Classification | F1 Score           | High performance, effective in high dimensions   | Requires tuning; slower on large data               |
| Mean Shift             | Clustering     | Silhouette Score   | No need to predefine clusters                    | Computationally expensive                           |
| BIRCH                  | Clustering     | Silhouette Score   | Scalable for large data; hierarchical            | Performance varies with threshold/branching factor  |

### 📈 Unified Performance Comparison Chart

Each algorithm was evaluated with an appropriate metric:

- **Regression**: Root Mean Square Error (lower is better)
- **Classification**: F1 Score (higher is better)
- **Clustering**: Silhouette Score (closer to 1 is better)

This chart helps visually assess the relative performance of each model under the selected metric. It enables quick insight into:
![image](https://github.com/user-attachments/assets/05f1fc87-e7c4-4c3e-875b-cf0d492a68a1)

- How well classification models (Naive Bayes, SVM) balance precision and recall.
- How effective unsupervised models (Mean Shift, BIRCH) are at natural grouping.
- The regression error magnitude of Random Forest on a payment prediction task.


### Output
[Random Forest Regression] RMSE: 6948.752645188508
![ML1](https://github.com/user-attachments/assets/7bb52287-adde-457a-bdf7-5c04cc2a1d26)
![ML2](https://github.com/user-attachments/assets/a0ff6d3e-c4f8-4185-9afa-95e21cca3830)
![ML3](https://github.com/user-attachments/assets/166cbf3a-9da2-4aef-9ed9-592bd4d02e30)
![ML4](https://github.com/user-attachments/assets/fde39ac8-c45e-43e0-80a8-3ab9d9dd6ef6)
![ML5](https://github.com/user-attachments/assets/9673a91c-60b2-43dc-9a71-38c48cccb2e5)
![image](https://github.com/user-attachments/assets/05f1fc87-e7c4-4c3e-875b-cf0d492a68a1)





