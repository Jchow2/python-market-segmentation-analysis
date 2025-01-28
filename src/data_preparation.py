import pandas as pd
import os

from sklearn.preprocessing import StandardScaler

def load_data_from_drive():
    from pathlib import Path

    # Set the working directory to the data/raw folder relative to the script location
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / '../../project-root/data/raw').resolve()

    # Convert the path to a string and replace backslashes with forward slashes
    data_dir_str = str(data_dir).replace('\\', '/')
    os.chdir(data_dir_str)

    # Load individual DataFrames
    databank = pd.read_csv('platform_databank.csv')
    sessions = pd.read_csv('platform_sessions.csv')
    replies = pd.read_csv('platform_replies.csv')
    screen_dict = pd.read_csv('platform_screen_dict.csv')
    requests = pd.read_csv('platform_requests.csv')

    return databank, sessions, replies, screen_dict, requests

def prepare_databank_data(databank):
    # Create a new column 'Country_code' and convert data types
    databank['Country_code'] = databank['cell_num_id'].astype(str).str[:3]
    databank['Country_code'] = databank['Country_code'].astype('int')

    # Standardize key names
    databank['key_name'] = databank['key_name'].str.lower()

    return databank

def prepare_interaction_data(databank, sessions, replies, screen_dict, requests):
    # Merge interaction data
    interaction_data = pd.merge(requests, replies, on='request_id', how="inner", validate="many_to_many")
    interaction_data.drop(['udate_y', 'sess_id_y'], axis=1, inplace=True)
    interaction_data.rename(columns={'udate_x': 'udate', 'sess_id_x': 'sess_id'}, inplace=True)

    interaction_data1 = pd.merge(interaction_data, screen_dict, on='response_id', how="inner", validate="many_to_many")

    # Merge session data
    session_data = pd.merge(sessions, interaction_data1, on='sess_id', how="inner", validate="many_to_many")
    session_data.drop(['platform_id_y'], axis=1, inplace=True)
    session_data.rename(columns={'platform_id_x': 'platform_id'}, inplace=True)

    # Expanding the databank table
    demographics = databank.pivot_table(index=['sess_id', 'cell_num_id'], columns='key_name', values='value_name', aggfunc='first')
    demographics_reset = demographics.reset_index()
    demographics_unique = demographics_reset.drop_duplicates(subset='cell_num_id')

    demographic_data = pd.merge(session_data, demographics_unique, on='cell_num_id', how="inner", validate="many_to_many")
    demographic_filtered = demographic_data[demographic_data['response_theme'] != 'Main Screen']
    demographic_filtered.drop(['udate_y', 'sess_id_y'], axis=1, inplace=True)
    demographic_filtered.rename(columns={'udate_x': 'udate', 'sess_id_x': 'sess_id'}, inplace=True)

    return demographic_filtered

def preprocess_data_original(databank):
    # Define clustering keys
    clustering_keys = ['age', 'Country_code', 'border', 'occupation', 'gender', 'education', 'crossingfreq', 'produce', 'commodityproduct', 'commoditymarket', 'language', 'proceduredest']

    # Split Key/Value Pairs and convert text to numerical vectors
    clustering_df = databank.pivot_table(index=['sess_id', 'cell_num_id'], columns='key_name', values='value_name', aggfunc='first')
    clustering_df = clustering_df.reset_index()
    clustering_df['Country_code'] = clustering_df['cell_num_id'].astype(str).str[:3]
    clustering_df['Country_code'] = clustering_df['Country_code'].astype('int')
    clustering_df = clustering_df[clustering_keys]

    # Handle missing values
    clustering_df = clustering_df.fillna('missing')

    # Convert categorical features to 'category' data type
    for col in clustering_df.columns:
        clustering_df[col] = clustering_df[col].astype('category')

    # Convert categorical features to numerical codes
    clustering_df_encoded = clustering_df.apply(lambda x: x.cat.codes if x.dtype.name == 'category' else x)

    return clustering_df_encoded

if __name__ == "__main__":
    # Load data from Google Drive
    databank, sessions, replies, screen_dict, requests = load_data_from_drive()

    # Prepare the databank DataFrame
    databank = prepare_databank_data(databank)

    # Ensure 'data/processed' directory exists
    processed_data_dir = os.path.join(os.path.dirname(__file__), '../../project-root/data/processed')
    os.makedirs(processed_data_dir, exist_ok=True)

    # Change the working directory to the processed data directory
    os.chdir(processed_data_dir)

    # Save the prepared databank DataFrame
    databank.to_csv('databank_cleaned.csv', index=False)
    print(f"Databank saved to: databank_cleaned.csv")

    # Prepare interaction data
    demographic_filtered = prepare_interaction_data(databank, sessions, replies, screen_dict, requests)

    # Save the cleaned and merged data
    demographic_filtered.to_csv('demographic_filtered.csv', index=False)
    print(f"Demographic data saved to: demographic_filtered.csv")

    # Preprocess the databank DataFrame for original clustering
    clustering_df_original = preprocess_data_original(databank)

    # Save the preprocessed data to the processed data directory
    clustering_df_original.to_csv('clustering_data_original.csv', index=False)
    print(f"Original Clustering data saved to: clustering_data_original.csv")