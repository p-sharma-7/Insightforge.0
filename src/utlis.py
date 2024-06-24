import dill # type: ignore
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException

def save_object(obj, file_path):
    """
    This function is responsible for saving the object in the file path
    :param obj: object to be saved
    :param file_path: file path where the object is to be saved
    :return: None
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train, y_train)  # Train model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test) 
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report


    except Exception as e:
        raise CustomException(e, sys)
    
def preprocess_data(data):
    # Check for non-numeric values in numeric columns and handle them
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        if data[column].apply(lambda x: isinstance(x, str)).any():
            # Handle non-numeric values here (e.g., convert to NaN, remove, etc.)
            # Example: Convert to NaN and then fill with median or remove
            data[column] = pd.to_numeric(data[column], errors='coerce')
            data[column].fillna(data[column].median(), inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    return data, label_encoders
    

def load_object(file_path):
    """
        This function is responsible for loading the object from the file path
        :param file_path: file path from where the object is to be loaded
        :return: object
        """
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e,sys)
    
def process_data(file_path):
    df = pd.read_csv(file_path)
    # Efficiently manipulate data using pandas
    df['processed'] = df['column'].apply(lambda x: x * 2)
    return df