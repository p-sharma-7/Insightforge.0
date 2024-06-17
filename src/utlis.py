import dill # type: ignore
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import GridSearchCV


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
    
def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            '''para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)'''

            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
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