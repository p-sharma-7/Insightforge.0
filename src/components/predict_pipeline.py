import sys
import pandas as pd
from src.exception import CustomException
from src.utlis import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path="artifacts/model.pkl"
            preprocess_path="artifacts/preprocess.pkl"
            model=load_object(model_path)
            preprocess=load_object(preprocess_path)
            data_scaled=preprocess.transform(features)
            pred =model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)


        