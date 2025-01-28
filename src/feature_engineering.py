import pandas as pd
import os

def preprocess_data(demographic_filtered):
    # Define clustering keys
    clustering_keys = ['age', 'border', 'occupation', 'gender', 'education', 'crossingfreq', 'produce', 'commodityproduct', 'commoditymarket', 'language', 'proceduredest']
    # Removed 'Country_code' because it's added later

    # Extract relevant data for clustering
    clustering_df = demographic_filtered[['sess_id', 'cell_num_id'] + clustering_keys]
    clustering_df = clustering_df.drop_duplicates(subset=['sess_id', 'cell_num_id']) # keep only unique session and cell_num_id

    # Calculate Country_code (if not already present) - This will add it
    clustering_df['Country_code'] = clustering_df['cell_num_id'].astype(str).str[:3]
    clustering_df['Country_code'] = clustering_df['Country_code'].astype('int')

    return clustering_df

def preprocess_data_original(databank):
    # Define clustering keys
    clustering_keys = ['age', 'Country_code', 'border', 'occupation', 'gender', 'education', 'crossingfreq', 'produce', 'commodityproduct', 'commoditymarket', 'language', 'proceduredest']

    # Split Key/Value Pairs and convert text to numerical vectors
    clustering_df = databank.pivot_table(index=['sess_id', 'cell_num_id'], columns='key_name', values='value_name', aggfunc='first')
    clustering_df = clustering_df.reset_index()
    clustering_df['Country_code'] = clustering_df['cell_num_id'].astype(str).str[:3]
    clustering_df['Country_code'] = clustering_df['Country_code'].astype('int')
    clustering_df = clustering_df[clustering_keys]

    return clustering_df

if __name__ == "__main__":
    # Ensure 'data/processed' directory exists
    processed_data_dir = os.path.join(os.path.dirname(__file__), '../../project-root/data/processed')
    os.makedirs(processed_data_dir, exist_ok=True)

    # Change the working directory to the processed data directory
    os.chdir(processed_data_dir)
    
    # Load the cleaned and merged data
    demographic_filtered_path = os.path.join(processed_data_dir, 'demographic_filtered.csv')
    demographic_filtered = pd.read_csv(demographic_filtered_path, low_memory=False)
    print("Loaded demographic_filtered.csv successfully.")

    # Preprocess the data for clustering
    clustering_df = preprocess_data(demographic_filtered)

    # Save the preprocessed data
    clustering_data_path = os.path.join(processed_data_dir, 'clustering_data.csv')
    clustering_df.to_csv(clustering_data_path, index=False)
    print(f"Clustering data saved to: {clustering_data_path}")

    # Load the cleaned data prepared by data_preparation.py
    databank_path = os.path.join(processed_data_dir, 'databank_cleaned.csv')
    databank = pd.read_csv(databank_path, low_memory=False)
    print("Loaded databank_cleaned.csv successfully.")

    # Preprocess the databank DataFrame
    clustering_df_original = preprocess_data_original(databank)

    # Save the preprocessed data to the processed data directory
    clustering_data_original_path = os.path.join(processed_data_dir, 'clustering_data_original.csv')
    clustering_df_original.to_csv(clustering_data_original_path, index=False)
    print(f"Original Clustering data saved to: {clustering_data_original_path}")