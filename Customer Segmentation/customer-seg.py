import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("customers.csv")
print(df.head(), "\n")

x = df[["AnnualIncome", "SpendingScore"]]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

inertia = []
for k in range(1, 11):
	kmeans = KMeans(n_clusters=k, random_state=42)
	kmeans.fit(x_scaled)
	inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(x_scaled)

plt.figure(figsize=(8,6))
for cluster in range(4):
    cluster_points = df[df["Cluster"] == cluster]
    plt.scatter(cluster_points["AnnualIncome"], cluster_points["SpendingScore"], label=f"Cluster {cluster}")

plt.title("Customer Segments (K-Means Clustering)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.legend()
plt.show()

centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster Centers (Approximate):")
print(pd.DataFrame(centers, columns=["AnnualIncome", "SpendingScore"]))
