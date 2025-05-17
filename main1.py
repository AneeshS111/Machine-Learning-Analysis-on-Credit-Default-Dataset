import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import MeanShift, Birch
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("Copy of default of credit card clients.csv", header=1)
df.drop(columns=['ID'], inplace=True)
df.rename(columns={'default payment next month': 'target'}, inplace=True)
# ----------------------------- REGRESSION -----------------------------
# Random Forest Regression
# Choose 'PAY_AMT1' as target
X_reg = df.drop(columns=['PAY_AMT1'])
y_reg = df['PAY_AMT1']

# Standardize the features
X_reg = pd.get_dummies(X_reg)
scaler = StandardScaler()
X_reg = scaler.fit_transform(X_reg)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_reg.predict(X_test_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print("\n[Random Forest Regression] RMSE:", rmse)

# ----------------------------- CLUSTERING -----------------------------
# Mean Shift and BIRCH Clustering

# Use PCA for dimensionality reduction
X_cluster = df.drop(columns=['target'])
X_cluster = pd.get_dummies(X_cluster)
X_cluster = StandardScaler().fit_transform(X_cluster)
X_pca = PCA(n_components=2).fit_transform(X_cluster)

# Mean Shift Clustering
mean_shift = MeanShift()
ms_labels = mean_shift.fit_predict(X_cluster)

# BIRCH Clustering
birch = Birch()
birch_labels = birch.fit_predict(X_cluster)

# Plotting Clustering Results
plt.figure(figsize=(12, 5))

# Mean Shift Plot
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=ms_labels, cmap='viridis')
plt.title('Mean Shift Clustering')

# BIRCH Plot
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=birch_labels, cmap='plasma')
plt.title('BIRCH Clustering')

plt.show()

# ----------------------------- CLASSIFICATION -----------------------------
# Naive Bayes and SVM Classification

X_cls = df.drop(columns=['target'])
y_cls = df['target']
X_cls = pd.get_dummies(X_cls)
X_cls = StandardScaler().fit_transform(X_cls)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_cls, y_train_cls)
y_pred_nb = nb.predict(X_test_cls)
print("Naive Bayes Classification Report:\n", classification_report(y_test_cls, y_pred_nb))

# Support Vector Machine
svm = SVC()
svm.fit(X_train_cls, y_train_cls)
y_pred_svm = svm.predict(X_test_cls)
print("SVM Classification Report:\n", classification_report(y_test_cls, y_pred_svm))

# ----------------------------- VISUALIZATIONS -----------------------------
# Confusion Matrix for SVM
plt.figure(figsize=(6, 5))
cm_svm = confusion_matrix(y_test_cls, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion Matrix for Naive Bayes
plt.figure(figsize=(6, 5))
cm_nb = confusion_matrix(y_test_cls, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Bar plot of feature importances from Random Forest (for Regression)
feature_importances = rf_reg.feature_importances_
plt.figure(figsize=(10, 5))
plt.barh(df.drop(columns=['target']).columns, feature_importances)
plt.title('Feature Importance - Random Forest Regression')
plt.show()
