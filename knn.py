import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)
df['species'] = y

# Data Analysis
print(df.head())
print(df.describe())
print(df['species'].value_counts())

# Data Visualization
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Training and Testing Split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalization of the dataset
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Initialize the k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model to the training data
knn.fit(train_x, train_y)

# Predict the output for the test data
pred_y = knn.predict(test_x)

# Print the test and predicted values
print("Test Values: ", test_y)
print("Predicted Values: ", pred_y)

# Print the accuracy score
print("Accuracy Score: {:.2f}%".format(accuracy_score(test_y, pred_y) * 100))

# Print the classification report
print("\nClassification Report:\n", classification_report(test_y, pred_y, target_names=iris.target_names))

# Print the confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(test_y, pred_y))