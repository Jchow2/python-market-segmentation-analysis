from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, accuracy_score
from yellowbrick.cluster import KElbowVisualizer # type: ignore
import lightgbm as lgb # type: ignore
import logging

## Define Text Embedding
def make_text(x):
    text =  f"""age: {x['age']},
                border: {x['border']},
                occupation: {x['occupation']},
                gender: {x['gender']},
                education: {x['education']},
                crossingfreq: {x['crossingfreq']},
                produce: {x['produce']},
                commodityproduct: {x['commodityproduct']},
                commoditymarket: {x['commoditymarket']},
                language: {x['language']},
                proceduredest: {x['proceduredest']},
                Country_code: {x['Country_code']}
            """
    return text

def get_embeddings_with_timing(texts, batch_size):
    import time
    start_time = time.time()
    output = model.encode(sentences=texts,
                          show_progress_bar=True,
                          normalize_embeddings=True,
                          batch_size=batch_size)
    end_time = time.time()
    print(f"Batch size: {batch_size}, Time taken: {end_time - start_time} seconds", flush=True)
    return output

def perform_clustering(df_embedding):
    # Instantiate the clustering model and visualizer
    km = KMeans(init="k-means++", random_state=0, n_init="auto")
    visualizer = KElbowVisualizer(km, k=(2, min(20, len(df_embedding))), locate_elbow=False)

    # Fit the data to the visualizer
    visualizer.fit(df_embedding)
    visualizer.show()

    # Define the number of clusters based on the elbow method
    n_clusters = 8  # Adjust based on elbow method results

    # Define KMeans clustering model
    km = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
    clusters = km.fit(df_embedding)

    # Print clustering details
    print(km.cluster_centers_.shape)
    print(km.labels_.shape)
    print(km.predict(df_embedding).shape)
    print("Compactness (Inertia):", clusters.inertia_)

    # Predict cluster memberships
    clusters_predict = clusters.predict(df_embedding)

    return km, clusters_predict

    # Sample a subset for the silhouette score calculation
    # sample_size = min(100000, len(df_embedding))  # Adjust the sample size as needed
    # if len(df_embedding) > sample_size:
        # sample_indices = np.random.choice(len(df_embedding), sample_size, replace=False)
        # df_embedding_sample = df_embedding.iloc[sample_indices]
        # clusters_predict_sample = clusters_predict[sample_indices]
    # else:
        # df_embedding_sample = df_embedding
        # clusters_predict_sample = clusters_predict

    # Calculate the silhouette score
    # silhouette_avg = silhouette_score(df_embedding_sample, clusters_predict_sample)
    # print(f"Silhouette Score: {silhouette_avg}")
    # print("Cluster memberships:\n{}".format(km.labels_))
    # return km, clusters_predict

def explain_clusters_with_lightgbm(clustering_df, clusters_predict):
    # Convert columns to categorical data type
    for col in ['age', 'Country_code', 'border', 'occupation', 'gender', 'education', 'crossingfreq', 'produce', 'commodityproduct', 'commoditymarket', 'language', 'proceduredest']:
        clustering_df[col] = clustering_df[col].astype('category')

    # Train the LightGBM model with config parameters to suppress warnings
    lgb_km = lgb.LGBMClassifier()
    lgb_km.set_params(verbosity=-1)  # Set verbosity to -1 to suppress warnings and info messages
    lgb_km.fit(X=clustering_df, y=clusters_predict)

    # Make predictions
    y_pred = lgb_km.predict(clustering_df)
    accuracy = accuracy_score(y_pred, clusters_predict)
    print('Training-set accuracy score: {0:0.4f}'.format(accuracy))
    print(classification_report(clusters_predict, y_pred))

if __name__ == "__main__":
    # Load the preprocessed data from feature_engineering.py
    clustering_df = pd.read_csv('data/processed/clustering_data_original.csv')

    # Use only a subset of the clustering data
    subset_size = 100000 # Adjust the subset size as needed
    if len(clustering_df) > subset_size:
        clustering_df = clustering_df.sample(n=subset_size, random_state=42)

    # Create text embeddings
    text = clustering_df.apply(lambda x: make_text(x), axis=1).tolist()

    # Load a smaller and faster model
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

    # Choose the optimal batch size based on timing results
    optimal_batch_size = 64  # Replace with the batch size that gave the best performance

    # Get the embeddings with the optimal batch size
    output = model.encode(sentences=text,
                          show_progress_bar=True,
                          normalize_embeddings=True,
                          batch_size=optimal_batch_size)

    # Convert embeddings to a DataFrame
    df_embedding = pd.DataFrame(output)

    # Perform clustering analysis
    km, clusters_predict = perform_clustering(df_embedding)

    # Explain clusters using LightGBM
    explain_clusters_with_lightgbm(clustering_df, clusters_predict)

    # Save embeddings and cluster predictions for visualization
    df_embedding.to_csv('data/processed/df_embedding.csv', index=False)
    np.save('data/processed/clusters_predict.npy', clusters_predict)