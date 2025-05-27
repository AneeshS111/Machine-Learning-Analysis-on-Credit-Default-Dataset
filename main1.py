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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score

# Load dataset
df = pd.read_csv("Default of credit card clients.csv", header=1)

# Drop the ID column if it exists
df.drop(columns=['ID'], inplace=True)

# Rename target column for ease
df.rename(columns={'default payment next month': 'target'}, inplace=True)

# ----------------------------- REGRESSION -----------------------------
# Random Forest Regression
X_reg = df.drop(columns=['PAY_AMT1'])
y_reg = df['PAY_AMT1']
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
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=ms_labels, cmap='viridis')
plt.title('Mean Shift Clustering')
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=birch_labels, cmap='plasma')
plt.title('BIRCH Clustering')
plt.show()

# ----------------------------- CLASSIFICATION -----------------------------
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

# SVM
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

# Feature Importance - Random Forest Regression
feature_importances = rf_reg.feature_importances_
plt.figure(figsize=(10, 5))
plt.barh(df.drop(columns=['target']).columns, feature_importances)
plt.title('Feature Importance - Random Forest Regression')
plt.show()

# ----------------------------- MODEL COMPARISON VISUALIZATION -----------------------------
# Classification metrics
nb_precision = precision_score(y_test_cls, y_pred_nb)
nb_recall = recall_score(y_test_cls, y_pred_nb)
nb_f1 = f1_score(y_test_cls, y_pred_nb)

svm_precision = precision_score(y_test_cls, y_pred_svm)
svm_recall = recall_score(y_test_cls, y_pred_svm)
svm_f1 = f1_score(y_test_cls, y_pred_svm)

# Clustering metrics
silhouette_ms = silhouette_score(X_cluster, ms_labels)
silhouette_birch = silhouette_score(X_cluster, birch_labels)

# Normalize RMSE (for visual comparison)
max_rmse = 10000  # Adjust depending on expected max RMSE
norm_rmse_score = 1 - min(rmse / max_rmse, 1)

# Compile into DataFrame for comparison
comparison_data = {
    'Model': [
        'Random Forest', 'Naive Bayes', 'Naive Bayes', 'Naive Bayes',
        'SVM', 'SVM', 'SVM',
        'Mean Shift', 'BIRCH'
    ],
    'Metric': [
        'Normalized RMSE', 'Precision', 'Recall', 'F1 Score',
        'Precision', 'Recall', 'F1 Score',
        'Silhouette Score', 'Silhouette Score'
    ],
    'Score': [
        norm_rmse_score,
        nb_precision, nb_recall, nb_f1,
        svm_precision, svm_recall, svm_f1,
        silhouette_ms, silhouette_birch
    ]
}

comparison_df = pd.DataFrame(comparison_data)

# Plotting grouped bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=comparison_df, x='Model', y='Score', hue='Metric', palette='Set2')
plt.title("Comparison of All Algorithms Across Evaluation Metrics")
plt.ylabel("Normalized Score")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
