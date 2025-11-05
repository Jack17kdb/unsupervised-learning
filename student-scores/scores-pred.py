import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("student_scores.csv")
print(df.head(), "\n")

x = df[['Math', 'Physics', 'Chemistry', 'Biology', 'English']]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(x_scaled)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
final_df = pd.concat([df, pca_df], axis=1)

print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(8,6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', s=60)
plt.title('PCA Projection of Student Scores')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
