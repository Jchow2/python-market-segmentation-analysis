import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import plotly.express as px # type: ignore

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

# Perform PCA with cumulative variance plot
def perform_pca(df_encoded, n_components=2):
    pca = PCA(n_components=n_components, random_state=42) # type: ignore
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

# Perform PCA and plot cumulative variance
pca_scores_df, pca = perform_pca(clustering_df_encoded, n_components=10)  # Adjust n_components as needed
plot_cumulative_variance(pca) # type: ignore

# Perform t-SNE on PCA-reduced data
def perform_tsne(pca_scores_df, perplexity=50, learning_rate='auto'):
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    tsne_result = tsne.fit_transform(pca_scores_df)
    tsne_df = pd.DataFrame(tsne_result, columns=['tsne_one', 'tsne_two'])
    return tsne_df

# Visualize t-SNE
def visualize_tsne(tsne_df, clusters_predict):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='tsne_one', y='tsne_two',
        hue=clusters_predict,
        palette=sns.color_palette('hsv', len(np.unique(clusters_predict))),
        data=tsne_df,
        legend='full',
        alpha=0.6
    )
    plt.title('t-SNE Visualization of Data (after PCA)')
    plt.show()

# Interactive t-SNE visualization
def interactive_tsne(tsne_df, clusters_predict):
    tsne_df['cluster'] = clusters_predict
    fig = px.scatter(
        tsne_df, x='tsne_one', y='tsne_two',
        color='cluster', title='t-SNE of Clusters',
        labels={'tsne_one': 't-SNE One', 'tsne_two': 't-SNE Two'}
    )
    fig.show()

if __name__ == "__main__":
    # Load cluster predictions to determine the maximum subset size
    clusters_predict = np.load('data/processed/clusters_predict.npy')
    max_subset_size = len(clusters_predict)

    # **1. Subset clustering_df before PCA and t-SNE**
    subset_size = min(100000, max_subset_size)
    clustering_df_subset = clustering_df.sample(n=subset_size, random_state=42)

    # Preprocess data for PCA (using the subset)
    clustering_df_encoded_subset = preprocess_for_pca(clustering_df_subset)

    # **2. Perform PCA on the subset**
    pca_scores_df_subset, pca = perform_pca(clustering_df_encoded_subset, n_components=10)
    plot_cumulative_variance(pca) # type: ignore

    # **3. Subset the cluster predictions to match the subset size**
    clusters_predict_subset = clusters_predict[:subset_size] #Adjust as needed

    # Perform t-SNE on the subset
    tsne_df = perform_tsne(pca_scores_df_subset, perplexity=50, learning_rate='auto')

    # Visualize t-SNE using the subset data and predictions
    visualize_tsne(tsne_df, clusters_predict_subset)
    interactive_tsne(tsne_df, clusters_predict_subset)