import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

df = pd.read_csv("customer_behavior.csv")
print(df.head(), "\n")

x = df[['Age', 'Annual_Income', 'Spending_Score']]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

dbscan = DBSCAN(eps=0.8, min_samples=3)
labels = dbscan.fit_predict(x_scaled)

df['Cluster'] = labels

print(df, "\n")

plt.figure(figsize=(8, 6))
plt.scatter(df['Annual_Income'], df['Spending_Score'], c=labels, cmap='rainbow', s=60)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("DBSCAN Clustering (Anomalies = -1)")
plt.show()
