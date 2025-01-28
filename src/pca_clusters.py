import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.cluster import KMeans

# Load preprocessed data
clustering_df = pd.read_csv('data/processed/clustering_data_original.csv')

# Function to handle missing values and convert categorical features
def preprocess_for_pca(df):
    df = df.fillna('missing')
    for col in df.columns:
        df[col] = df[col].astype('category')
    df_encoded = df.apply(lambda x: x.cat.codes if x.dtype.name == 'category' else x)
    return df_encoded

# Preprocess data for PCA
clustering_df_encoded = preprocess_for_pca(clustering_df)

# Determine the optimal number of clusters using the Elbow method
wcss = []
max_clusters = 15  # Set the maximum number of clusters to test

for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(clustering_df_encoded)
    wcss.append(kmeans.inertia_)

# Plot the WCSS against the number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Determining Optimal Number of Clusters')
plt.grid()
plt.show()

# Perform PCA with cumulative variance plot
def perform_pca(df_encoded, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(df_encoded)
    pca_scores_df = pd.DataFrame(pca_result, columns=[f'pca_{i+1}' for i in range(n_components)])
    return pca_scores_df, pca

# Plot cumulative variance
def plot_cumulative_variance(pca):
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by Number of Principal Components')
    plt.grid()
    plt.show()

# Plot PCA Visualizations
def plot_pca(df_embedding, clusters_predict):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_embedding)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=pca_result[:, 0], y=pca_result[:, 1],
        hue=clusters_predict,
        palette=sns.color_palette('hsv', len(np.unique(clusters_predict))),
        alpha=0.6
    )
    plt.title('PCA of Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Use only a subset of the data to speed up processing and maintain consistency with clustering
    subset_size = 100000  # Adjust the subset size as needed
    if len(clustering_df) > subset_size:
        clustering_df = clustering_df.sample(n=subset_size, random_state=42)

    # Preprocess data for PCA
    clustering_df_encoded = preprocess_for_pca(clustering_df)

    # Perform PCA and plot cumulative variance
    pca_scores_df, pca = perform_pca(clustering_df_encoded)
    plot_cumulative_variance(pca)

    # Add PCA results to DataFrame
    clustering_df['pca_one'] = pca_scores_df['pca_1']
    clustering_df['pca_two'] = pca_scores_df['pca_2']

    # Determine the optimal number of clusters using the Elbow method
    wcss = []
    max_clusters = 15  # Set the maximum number of clusters to test

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(clustering_df_encoded)
        wcss.append(kmeans.inertia_)

    # Visualize clusters using PCA
    optimal_clusters = 2  # Set the optimal number of clusters based on the Elbow method
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clustering_df['cluster'] = kmeans.fit_predict(clustering_df_encoded)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='pca_one', y='pca_two',
        hue='cluster',
        palette=sns.color_palette('hsv', optimal_clusters),
        data=clustering_df,
        legend='full',
        alpha=0.6
    )
    plt.title('PCA of Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid()
    plt.show()