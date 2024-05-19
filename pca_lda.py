import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load the Iris dataset
iris = load_iris()
x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)
df['species'] = y

# Standardize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Perform PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=x_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['species'] = y

# Visualize PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='species', data=pca_df, palette='viridis', s=100)
plt.title('PCA of Iris Dataset')
plt.show()

# Perform LDA
lda = LDA(n_components=2)
x_lda = lda.fit_transform(x_scaled, y)

# Create a DataFrame for LDA results
lda_df = pd.DataFrame(data=x_lda, columns=['LD1', 'LD2'])
lda_df['species'] = y

# Visualize LDA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='LD1', y='LD2', hue='species', data=lda_df, palette='viridis', s=100)
plt.title('LDA of Iris Dataset')
plt.show()

# Print explained variance ratios
print("Explained variance ratio (PCA): ", pca.explained_variance_ratio_)