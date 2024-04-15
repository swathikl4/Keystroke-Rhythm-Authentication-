import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('DSL-StrongPasswordData.csv')

# Data exploratory analysis
print("Data Head:")
print(data.head())

print("\nData Tail:")
print(data.tail())

print("\nNumber of Rows and Columns:")
print(data.shape)

print("\nData Types of Each Column:")
print(data.dtypes)

print("\nNull Values in Each Column:")
print(data.isnull().sum())

# Categorical graphs for each column
for column in data.columns:
    if data[column].dtype == 'object':
        plt.figure(figsize=(10, 6))
        sns.countplot(data[column])
        plt.title(f"{column} Count Plot")
        plt.show()

# Box plot for each column
for column in data.columns:
    if data[column].dtype == 'float64':
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=data[column])
        plt.title(f"{column} Box Plot")
        plt.show()

# Calculate average latency for each user and each column
average_latencies = {}
for user in data['subject'].unique():
    user_data = data[data['subject'] == user]
    user_avg_latencies = user_data.groupby('subject').mean().squeeze()
    average_latencies[user] = user_avg_latencies

# Plot average latency for each user and each column
plt.figure(figsize=(12, 6))
for i, (user, avg_latency) in enumerate(average_latencies.items()):
    if i % 10 == 0 and i != 0:
        plt.legend(loc='upper right')
        plt.xlabel('Column')
        plt.ylabel('Average Latency')
        plt.title(f'Average Latency for 10 Users (Subjects)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(12, 6))
    plt.plot(avg_latency.index, avg_latency.values, marker='o', label=f'User {user}')
plt.legend(loc='upper right')
plt.xlabel('Column')
plt.ylabel('Average Latency')
plt.title(f'Average Latency for 10 Users (Subjects)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Display heatmap of the data
plt.figure(figsize=(20, 16))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'size': 8})
plt.title('Correlation Heatmap')
plt.show()


# Standardize the data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(['subject'], axis=1))

# Perform t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
data_tsne = tsne.fit_transform(data_scaled)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Visualize clustered data with t-SNE
plt.figure(figsize=(12, 8))
sns.scatterplot(x=data_tsne[:, 0], y=data_tsne[:, 1], hue=clusters, palette='viridis')
plt.title('t-SNE Visualization of Clusters')
plt.show()