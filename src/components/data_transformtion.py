import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import category_encoders as ce
import os
from src.utlis import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self, target_column_name):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            data= pd.read_csv('artifacts/data.csv')
            X= data.drop(columns=[target_column_name],axis=1)

            categorical_columns = X.select_dtypes(include=['object', 'category']).columns
            numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.drop(target_column_name)

#categorical_column
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            #if data.nunique().sum() < 50:
            cat_pipeline=Pipeline(steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore")),
                    ("scaler",StandardScaler(with_mean=False))
                ])

            '''
                elif data.nunique().sum() > 50:
                cat_pipeline=Pipeline(steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("Target encoding", ce.TargetEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ])'''

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path, target_column_name):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object(target_column_name)


            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.concatenate((input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)), axis=1)

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)