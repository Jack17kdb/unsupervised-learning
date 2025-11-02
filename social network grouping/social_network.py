import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("social_network.csv")
print(df.head(), "\n")

x = df[["Age", "FriendCount"]]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

linked = linkage(x_scaled, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram (Ward's Method)")
plt.xlabel("Data Points (Users)")
plt.ylabel("Euclidean Distance")
plt.show()

labels = fcluster(linked, 3, criterion='maxclust')
df["Cluster"] = labels

plt.figure(figsize=(8,6))
for cluster in np.unique(labels):
    cluster_points = df[df["Cluster"] == cluster]
    plt.scatter(cluster_points["Age"], cluster_points["FriendCount"], label=f"Cluster {cluster}")

plt.title("Social Network Groups (Hierarchical Clustering)")
plt.xlabel("Age")
plt.ylabel("Friend Count")
plt.legend()
plt.show()

print(df[["UserID", "Age", "FriendCount", "Cluster"]])
