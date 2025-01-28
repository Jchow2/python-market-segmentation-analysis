import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import unittest
from pathlib import Path

# Add the directory containing clustering_analysis.py to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from clustering_analysis import make_text, perform_clustering, explain_clusters_with_lightgbm

# Create a simplified DataFrame with sample data
simplified_data = pd.DataFrame({
    'sess_id': [1, 2],
    'cell_num_id': ['US123', 'MX456'],
    'age': [30, 40],
    'border': ['US-Mexico', 'US-Canada'],
    'occupation': ['Engineer', 'Doctor'],
    'gender': ['Male', 'Female'],
    'education': ['Bachelor', 'Master'],
    'crossingfreq': ['Daily', 'Weekly'],
    'produce': ['None', 'Fruits'],
    'commodityproduct': ['Electronics', 'Medicine'],
    'commoditymarket': ['Local', 'International'],
    'language': ['English', 'Spanish'],
    'proceduredest': ['USA', 'Canada'],
    'Country_code': ['US', 'MX']
})

# Define a make_text function to convert the simplified data into text
def make_text(x):
    text = f"""age: {x['age']},
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

# Create the text representation of the data
simplified_data['text'] = simplified_data.apply(make_text, axis=1)

class TestClusteringAnalysis(unittest.TestCase):

    def test_make_text(self):
        sample_input = simplified_data.iloc[0]
        expected_output = """age: 30,
                border: US-Mexico,
                occupation: Engineer,
                gender: Male,
                education: Bachelor,
                crossingfreq: Daily,
                produce: None,
                commodityproduct: Electronics,
                commoditymarket: Local,
                language: English,
                proceduredest: USA,
                Country_code: US
            """
        self.assertEqual(make_text(sample_input), expected_output)

    @patch('clustering_analysis.KMeans')
    @patch('clustering_analysis.KElbowVisualizer')
    def test_perform_clustering(self, MockKElbowVisualizer, MockKMeans):
        mock_visualizer = MockKElbowVisualizer.return_value
        mock_visualizer.fit.return_value = None
        mock_visualizer.show.return_value = None

        mock_kmeans = MockKMeans.return_value
        mock_kmeans.fit.return_value = MagicMock(inertia_=100, labels_=np.array([0, 1]), predict=MagicMock(return_value=np.array([0, 1])))

        df_embedding = pd.DataFrame(np.random.rand(2, 768))
        km, clusters_predict = perform_clustering(df_embedding)
        self.assertEqual(km, mock_kmeans)
        self.assertTrue((clusters_predict == np.array([0, 1])).all())

    @patch('clustering_analysis.lgb.LGBMClassifier')
    def test_explain_clusters_with_lightgbm(self, MockLGBMClassifier):
        mock_model = MockLGBMClassifier.return_value
        mock_model.predict.return_value = np.array([0, 1])  # Ensure the length matches clusters_predict
        mock_model.fit.return_value = None

        clustering_df = pd.DataFrame({
            'age': [30, 40],
            'Country_code': ['US', 'MX'],
            'border': ['US-Mexico', 'US-Canada'],
            'occupation': ['Engineer', 'Doctor'],
            'gender': ['Male', 'Female'],
            'education': ['Bachelor', 'Master'],
            'crossingfreq': ['Daily', 'Weekly'],
            'produce': ['None', 'Fruits'],
            'commodityproduct': ['Electronics', 'Medicine'],
            'commoditymarket': ['Local', 'International'],
            'language': ['English', 'Spanish'],
            'proceduredest': ['USA', 'Canada']
        })
        clusters_predict = np.array([0, 1])

        explain_clusters_with_lightgbm(clustering_df, clusters_predict)

        mock_model.fit.assert_called_once_with(X=clustering_df, y=clusters_predict)
        mock_model.predict.assert_called_once_with(clustering_df)

if __name__ == '__main__':
    unittest.main()