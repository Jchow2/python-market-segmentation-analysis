import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import pandas as pd

# Add the directory containing data_preparation.py to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from data_preparation import load_data_from_drive, prepare_databank_data, prepare_interaction_data

class TestDataPreparation(unittest.TestCase):
    
    @patch('data_preparation.os.chdir')
    @patch('data_preparation.pd.read_csv')
    @patch('pathlib.Path')
    def test_load_data_from_drive(self, mock_path, mock_read_csv, mock_chdir):
        # Mock the Path object and its methods
        mock_script_dir = MagicMock()
        mock_data_dir = MagicMock()
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.resolve.return_value.parent = mock_script_dir
        mock_script_dir.__truediv__.return_value.resolve.return_value = mock_data_dir
        mock_data_dir.__str__.return_value = '/mocked/path'
        
        # Mock the pandas read_csv function
        mock_df = MagicMock()
        mock_read_csv.return_value = mock_df
        
        # Call the function
        databank, sessions, replies, screen_dict, requests = load_data_from_drive()
        
        # Assertions to verify behavior
        mock_path.assert_called_once()
        mock_script_dir.__truediv__.assert_called_once_with('../../project-root/data/raw')
        mock_chdir.assert_called_once_with('/mocked/path')
        mock_read_csv.assert_any_call('platform_databank.csv')
        mock_read_csv.assert_any_call('platform_sessions.csv')
        mock_read_csv.assert_any_call('platform_replies.csv')
        mock_read_csv.assert_any_call('platform_screen_dict.csv')
        mock_read_csv.assert_any_call('platform_requests.csv')
        self.assertEqual(databank, mock_df)
        self.assertEqual(sessions, mock_df)
        self.assertEqual(replies, mock_df)
        self.assertEqual(screen_dict, mock_df)
        self.assertEqual(requests, mock_df)

    def test_prepare_databank_data(self):
        # Create a mock DataFrame
        mock_df = pd.DataFrame({
            'cell_num_id': ['1234567890', '0987654321'],
            'key_name': ['KEY1', 'KEY2']
        })
        
        # Call the function
        result = prepare_databank_data(mock_df)
        
        # Assertions to verify behavior
        self.assertIn('Country_code', result.columns)
        self.assertEqual(result['Country_code'].iloc[0], 123)
        self.assertEqual(result['Country_code'].iloc[1], 98)
        self.assertEqual(result['key_name'].iloc[0], 'key1')
        self.assertEqual(result['key_name'].iloc[1], 'key2')

    def test_prepare_interaction_data(self):
        # Create mock DataFrames
        mock_databank = pd.DataFrame({
            'sess_id': [1, 2],
            'cell_num_id': ['1234567890', '0987654321'],
            'key_name': ['age', 'gender'],
            'value_name': [25, 'M']
        })
        mock_requests = pd.DataFrame({'request_id': [1, 2], 'other_col': ['A', 'B']})
        mock_replies = pd.DataFrame({'request_id': [1, 2], 'response_id': [1, 2], 'udate_y': ['2021-01-01', '2021-01-02'], 'sess_id_y': [1, 2], 'sess_id': [1, 2]})
        mock_screen_dict = pd.DataFrame({'response_id': [1, 2], 'screen_col': ['X', 'Y']})
        mock_sessions = pd.DataFrame({'sess_id': [1, 2], 'cell_num_id': ['1234567890', '0987654321'], 'platform_id_y': ['P1', 'P2'], 'response_theme': ['Theme1', 'Theme2'], 'udate_y': ['2021-01-01', '2021-01-02'], 'sess_id_y': [1, 2]})
        
        # Call the function
        result = prepare_interaction_data(mock_databank, mock_sessions, mock_replies, mock_screen_dict, mock_requests)
        
        # Assertions to verify behavior
        self.assertIn('request_id', result.columns)
        self.assertNotIn('udate_y', result.columns)
        self.assertNotIn('sess_id_y', result.columns)
        self.assertNotIn('platform_id_y', result.columns)

if __name__ == '__main__':
    unittest.main()