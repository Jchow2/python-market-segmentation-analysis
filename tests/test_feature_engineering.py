import unittest
import sys
from pathlib import Path
import pandas as pd

# Add the directory containing feature_engineering.py to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from feature_engineering import preprocess_data, preprocess_data_original

class TestFeatureEngineering(unittest.TestCase):

    def test_preprocess_data(self):
        # Mock the DataFrame for preprocess_data
        mock_df = pd.DataFrame({
            'sess_id': [1, 2],
            'cell_num_id': [123, 456],
            'age': [25, 30],
            'border': ['A', 'B'],
            'occupation': ['Farmer', 'Trader'],
            'gender': ['M', 'F'],
            'education': ['High School', 'College'],
            'crossingfreq': [5, 10],
            'produce': ['Maize', 'Beans'],
            'commodityproduct': ['Product1', 'Product2'],
            'commoditymarket': ['Market1', 'Market2'],
            'language': ['English', 'Swahili'],
            'proceduredest': ['Destination1', 'Destination2']
        })

        # Call the function
        clustering_df = preprocess_data(mock_df)

        # Verify the output DataFrame
        self.assertIn('Country_code', clustering_df.columns)
        self.assertEqual(clustering_df['Country_code'].iloc[0], 123)

    def test_preprocess_data_original(self):
        # Mock the DataFrame for preprocess_data_original
        mock_df = pd.DataFrame({
            'sess_id': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            'cell_num_id': [123, 456, 123, 456, 123, 456, 123, 456, 123, 456, 123, 456, 123, 456, 123, 456, 123, 456, 123, 456, 123, 456],
            'key_name': ['age', 'age', 'border', 'border', 'occupation', 'occupation', 'gender', 'gender', 'education', 'education', 'crossingfreq', 'crossingfreq', 'produce', 'produce', 'commodityproduct', 'commodityproduct', 'commoditymarket', 'commoditymarket', 'language', 'language', 'proceduredest', 'proceduredest'],
            'value_name': [25, 30, 'A', 'B', 'Farmer', 'Trader', 'M', 'F', 'High School', 'College', 5, 10, 'Maize', 'Beans', 'Product1', 'Product2', 'Market1', 'Market2', 'English', 'Swahili', 'Destination1', 'Destination2']
        })

        # Call the function
        clustering_df = preprocess_data_original(mock_df)

        # Verify the output DataFrame
        self.assertIn('Country_code', clustering_df.columns)

if __name__ == '__main__':
    unittest.main()