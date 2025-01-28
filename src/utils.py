# Utility library imports
import os
import json
from datetime import datetime

# Process site-packages imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import lightgbm as lgb

# Utility functions for file operations
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def save_results(results, output_file):
    with open(output_file, 'w') as file:
        json.dump(results, file)

def plot_results(data, title, xlabel, ylabel):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def read_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def save_csv(file_path, df):
    df.to_csv(file_path, index=False)

# Utility functions for data preparation
def load_data_from_drive():
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / '../../project-root/data/raw').resolve()
    data_dir_str = str(data_dir).replace('\\', '/')
    os.chdir(data_dir_str)
    databank = pd.read_csv('platform_databank.csv', low_memory=False)
    sessions = pd.read_csv('platform_sessions.csv', low_memory=False)
    replies = pd.read_csv('platform_replies.csv', low_memory=False)
    screen_dict = pd.read_csv('platform_screen_dict.csv', low_memory=False)
    requests = pd.read_csv('platform_requests.csv', low_memory=False)
    return databank, sessions, replies, screen_dict, requests

def prepare_databank_data(databank):
    databank['Country_code'] = databank['cell_num_id'].astype(str).str[:3]
    databank['Country_code'] = databank['Country_code'].astype('int')
    databank['key_name'] = databank['key_name'].str.lower()
    return databank

def prepare_interaction_data(databank, sessions, replies, screen_dict, requests):
    interaction_data = pd.merge(requests, replies, on='request_id', how="inner", validate="many_to_many")
    interaction_data.drop(['udate_y', 'sess_id_y'], axis=1, inplace=True)
    interaction_data.rename(columns={'udate_x': 'udate', 'sess_id_x': 'sess_id'}, inplace=True)
    interaction_data1 = pd.merge(interaction_data, screen_dict, on='response_id', how="inner", validate="many_to_many")
    session_data = pd.merge(sessions, interaction_data1, on='sess_id', how="inner", validate="many_to_many")
    session_data.drop(['platform_id_y'], axis=1, inplace=True)
    session_data.rename(columns={'platform_id_x': 'platform_id'}, inplace=True)
    demographics = databank.pivot_table(index=['sess_id', 'cell_num_id'], columns='key_name', values='value_name', aggfunc='first')
    demographics_reset = demographics.reset_index()
    demographics_unique = demographics_reset.drop_duplicates(subset='cell_num_id')
    demographic_data = pd.merge(session_data, demographics_unique, on='cell_num_id', how="inner", validate="many_to_many")
    demographic_filtered = demographic_data[demographic_data['response_theme'] != 'Main Screen']
    demographic_filtered.drop(['udate_y', 'sess_id_y'], axis=1, inplace=True)
    demographic_filtered.rename(columns={'udate_x': 'udate', 'sess_id_x': 'sess_id'}, inplace=True)
    return demographic_filtered

# Utility functions for feature engineering
def preprocess_data(demographic_filtered):
    clustering_keys = ['age', 'border', 'occupation', 'gender', 'education', 'crossingfreq', 'produce', 'commodityproduct', 'commoditymarket', 'language', 'proceduredest']
    clustering_df = demographic_filtered[['sess_id', 'cell_num_id'] + clustering_keys]
    clustering_df = clustering_df.drop_duplicates(subset=['sess_id', 'cell_num_id'])
    clustering_df['Country_code'] = clustering_df['cell_num_id'].astype(str).str[:3]
    clustering_df['Country_code'] = clustering_df['Country_code'].astype('int')
    return clustering_df

def preprocess_data_original(databank):
    clustering_keys = ['age', 'Country_code', 'border', 'occupation', 'gender', 'education', 'crossingfreq', 'produce', 'commodityproduct', 'commoditymarket', 'language', 'proceduredest']
    clustering_df = databank.pivot_table(index=['sess_id', 'cell_num_id'], columns='key_name', values='value_name', aggfunc='first')
    clustering_df = clustering_df.reset_index()
    clustering_df['Country_code'] = clustering_df['cell_num_id'].astype(str).str[:3]
    clustering_df['Country_code'] = clustering_df['Country_code'].astype('int')
    clustering_df = clustering_df[clustering_keys]
    return clustering_df

# Utility functions for clustering analysis
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

def get_embeddings_with_timing(texts, batch_size, model):
    import time
    start_time = time.time()
    output = model.encode(sentences=texts, show_progress_bar=True, normalize_embeddings=True, batch_size=batch_size)
    end_time = time.time()
    print(f"Batch size: {batch_size}, Time taken: {end_time - start_time} seconds", flush=True)
    return output

def perform_clustering(df_embedding):
    if len(df_embedding) < 3:
        raise ValueError("The number of samples should be at least 3 for clustering.")
    km = KMeans(init="k-means++", random_state=0, n_init="auto")
    visualizer = KElbowVisualizer(km, k=(2, min(20, len(df_embedding))), locate_elbow=False)
    visualizer.fit(df_embedding)
    visualizer.show()
    n_clusters = 8
    km = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
    clusters = km.fit(df_embedding)
    clusters_predict = clusters.predict(df_embedding)
    sample_size = min(300000, len(df_embedding))
    if len(df_embedding) > sample_size:
        sample_indices = np.random.choice(len(df_embedding), sample_size, replace=False)
        df_embedding_sample = df_embedding.iloc[sample_indices]
        clusters_predict_sample = clusters_predict[sample_indices]
    else:
        df_embedding_sample = df_embedding
        clusters_predict_sample = clusters_predict
    silhouette_avg = silhouette_score(df_embedding_sample, clusters_predict_sample)
    print(f"Silhouette Score: {silhouette_avg}")
    return km, clusters_predict

def explain_clusters_with_lightgbm(clustering_df, clusters_predict):
    for col in ['age', 'Country_code', 'border', 'occupation', 'gender', 'education', 'crossingfreq', 'produce', 'commodityproduct', 'commoditymarket', 'language', 'proceduredest']:
        clustering_df[col] = clustering_df[col].astype('category')
    lgb_km = lgb.LGBMClassifier()
    lgb_km.set_params(verbosity=-1)
    lgb_km.fit(X=clustering_df, y=clusters_predict)
    y_pred = lgb_km.predict(clustering_df)
    accuracy = accuracy_score(y_pred, clusters_predict)
    print('Training-set accuracy score: {0:0.4f}'.format(accuracy))
    print(classification_report(clusters_predict, y_pred))

# Utility functions for classification report
def load_preprocessed_data():
    demographic_data = pd.read_csv('data/processed/demographic_filtered.csv')
    return demographic_data

def sample_data(demographic_data, sample_size=5000):
    if demographic_data.shape[0] > sample_size:
        demographic_data_sampled = demographic_data.sample(n=sample_size, random_state=42)
    else:
        demographic_data_sampled = demographic_data
    return demographic_data_sampled

def prepare_features_and_target(demographic_data_sampled):
    X = demographic_data_sampled.drop(['response_theme', 'reply_id', 'request_id', 'response_id', 'parent', 'created_date', 'notes'], axis=1)
    y = demographic_data_sampled['response_theme']
    return X, y

def preprocess_datetime_columns(X):
    X['udate'] = X['udate'].apply(lambda d: datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))
    return X

def create_preprocessing_pipelines():
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ], memory=None)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ], memory=None)
    return numerical_transformer, categorical_transformer

def create_preprocessor(numerical_transformer, categorical_transformer, X):
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    for col in categorical_columns:
        X[col] = X[col].astype(str)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])
    return preprocessor

def build_and_train_model(preprocessor, X_train, y_train):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LinearSVC(random_state=42))
    ], memory=None)
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))