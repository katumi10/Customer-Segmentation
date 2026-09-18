
# Task 2: Customer Segmentation
# Elevvo Machine Learning Internship

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# 2. Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

# 3. Select features for clustering
# Annual Income and Spending Score are the features required for this task
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Visualize the original data
plt.figure(figsize=(8, 6))
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"]
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Annual Income vs Spending Score")
plt.show()

# 5. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Find the optimal number of clusters using the Elbow Method
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.show()

# 7. Train K-Means with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 8. Display customers with their cluster
print("\nCustomers with cluster assignments:")
print(df.head())

# 9. Visualize the clusters
plt.figure(figsize=(10, 7))

for cluster in sorted(df["Cluster"].unique()):
    cluster_data = df[df["Cluster"] == cluster]

    plt.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        label=f"Cluster {cluster}"
    )

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation Using K-Means")
plt.legend()
plt.show()

# 10. Analyze average spending and income per cluster
cluster_summary = df.groupby("Cluster")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print("\nAverage income and spending score by cluster:")
print(cluster_summary)

# 11. Number of customers in each cluster
cluster_counts = df["Cluster"].value_counts().sort_index()

print("\nNumber of customers in each cluster:")
print(cluster_counts)

