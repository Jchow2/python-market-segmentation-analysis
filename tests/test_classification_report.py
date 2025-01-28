import unittest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Add the directory containing classification_report.py to the system path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from classification_report import load_preprocessed_data, sample_data, prepare_features_and_target, preprocess_datetime_columns, create_preprocessing_pipelines, create_preprocessor, build_and_train_model, evaluate_model

class TestClassificationReport(unittest.TestCase):

    @patch('classification_report.pd.read_csv')
    def test_load_preprocessed_data(self, mock_read_csv):
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_read_csv.return_value = mock_df
        df = load_preprocessed_data()
        mock_read_csv.assert_called_once_with('data/processed/demographic_filtered.csv')
        pd.testing.assert_frame_equal(df, mock_df)

    def test_sample_data(self):
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        df = sample_data(mock_df, 2)
        self.assertEqual(len(df), 2)

    def test_prepare_features_and_target(self):
        df = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4],
            'response_theme': [0, 1],
            'reply_id': [1, 2],
            'request_id': [1, 2],
            'response_id': [1, 2],
            'parent': [1, 2],
            'created_date': ['2021-01-01', '2021-01-02'],
            'notes': ['note1', 'note2']
        })
        X, y = prepare_features_and_target(df)
        pd.testing.assert_frame_equal(X, df[['feature1', 'feature2']])
        pd.testing.assert_series_equal(y, df['response_theme'])

    def test_preprocess_datetime_columns(self):
        df = pd.DataFrame({'udate': ['2021-01-01 00:00:00', '2021-01-02 00:00:00']})
        df = preprocess_datetime_columns(df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['udate']))

    def test_create_preprocessing_pipelines(self):
        numerical_transformer, categorical_transformer = create_preprocessing_pipelines()
        self.assertIsInstance(numerical_transformer, Pipeline)
        self.assertIsInstance(categorical_transformer, Pipeline)

    def test_create_preprocessor(self):
        numerical_transformer, categorical_transformer = create_preprocessing_pipelines()
        X = pd.DataFrame({'num_col': [1, 2], 'cat_col': ['a', 'b']})
        preprocessor = create_preprocessor(numerical_transformer, categorical_transformer, X)
        self.assertIsInstance(preprocessor, ColumnTransformer)

    @patch('classification_report.Pipeline.fit')
    def test_build_and_train_model(self, mock_fit):
        numerical_transformer, categorical_transformer = create_preprocessing_pipelines()
        X = pd.DataFrame({'num_col': [1, 2], 'cat_col': ['a', 'b']})
        preprocessor = create_preprocessor(numerical_transformer, categorical_transformer, X)
        y = pd.Series([0, 1])
        model = build_and_train_model(preprocessor, X, y)
        mock_fit.assert_called_once_with(X, y)

    @patch('classification_report.accuracy_score')
    @patch('classification_report.classification_report')
    def test_evaluate_model(self, mock_classification_report, mock_accuracy_score):
        mock_accuracy_score.return_value = 1.0
        mock_classification_report.return_value = 'classification report'
        model = MagicMock()
        model.predict.return_value = [0, 1]  # Mock model predictions
        X = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        y = pd.Series([0, 1])
        accuracy_report = evaluate_model(model, X, y)
        print("Returned accuracy_report:", accuracy_report)
        self.assertIsNotNone(accuracy_report, "evaluate_model returned None")
        accuracy, report = accuracy_report
        mock_accuracy_score.assert_called_once_with(ANY, [0, 1])
        mock_classification_report.assert_called_once_with(ANY, [0, 1])
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(report, 'classification report')

if __name__ == '__main__':
    unittest.main()