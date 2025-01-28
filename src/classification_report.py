import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

def load_preprocessed_data():
    # Load preprocessed data
    demographic_data = pd.read_csv('data/processed/demographic_filtered.csv')
    return demographic_data

def sample_data(demographic_data, sample_size=100000):
    # Sample a smaller subset of the data to reduce computation time
    if demographic_data.shape[0] > sample_size:
        demographic_data_sampled = demographic_data.sample(n=sample_size, random_state=42)
    else:
        demographic_data_sampled = demographic_data
    return demographic_data_sampled

def prepare_features_and_target(demographic_data_sampled):
    # Define your feature DataFrame and target variable
    X = demographic_data_sampled.drop(['response_theme', 'reply_id', 'request_id', 'response_id', 'parent', 'created_date', 'notes'], axis=1)
    y = demographic_data_sampled['response_theme']
    return X, y

def preprocess_datetime_columns(X):
    # Convert datetime columns to datetime objects
    X['udate'] = X['udate'].apply(lambda d: datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))
    return X

def create_preprocessing_pipelines():
    # Create the preprocessing pipelines for both numerical and categorical data
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
    # Define categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

    # Convert mixed-type columns to string type
    for col in categorical_columns:
        X[col] = X[col].astype(str)  # Convert all values in categorical columns to strings

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    return preprocessor

def build_and_train_model(preprocessor, X_train, y_train):
    # Build the LinearSVC model using the preprocessor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LinearSVC(random_state=42))
    ], memory=None)

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    # Predict on the test data
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Report:\n", report)
    print("Accuracy Score:", accuracy)
    return accuracy, report

if __name__ == "__main__":
    # Load and sample preprocessed data
    demographic_data = load_preprocessed_data()
    demographic_data_sampled = sample_data(demographic_data)

    # Prepare features and target
    X, y = prepare_features_and_target(demographic_data_sampled)

    # Preprocess datetime columns
    X = preprocess_datetime_columns(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Create preprocessing pipelines
    numerical_transformer, categorical_transformer = create_preprocessing_pipelines()

    # Create preprocessor
    preprocessor = create_preprocessor(numerical_transformer, categorical_transformer, X_train)

    # Build and train model
    pipeline = build_and_train_model(preprocessor, X_train, y_train)

    # Evaluate the model
    evaluate_model(pipeline, X_test, y_test)