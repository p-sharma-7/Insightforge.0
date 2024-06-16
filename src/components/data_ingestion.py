import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformtion import DataTransformation
from src.components.data_transformtion import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifacts','train.csv')
    test_data_path: str= os.path.join('artifacts','test.csv')
    raw_data_path:str= os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self, data):
        logging.info("Entered the data ingestion method or components")
        try:
            df=pd.DataFrame(data)
            logging.info("read the dataset as dataframe")


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df = pd.DataFrame(data)  # Define the variable "df" by reading the CSV file using the "data" variable

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            train_set,test_set= train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        
        except Exception as e:
            logging.error(f"Error reading data: {e}")
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.train_model(train_arr,test_arr))