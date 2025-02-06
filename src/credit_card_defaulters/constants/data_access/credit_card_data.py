from src.credit_card_defaulters.config.mongodb_connection import MongoDBClient
from src.credit_card_defaulters.constants.training_pipeline import DATABASE_NAME
from src.credit_card_defaulters.logger import logging
from typing import Optional
import pandas as pd
from json import loads, dumps
import numpy as np

class CreditData:
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
            logging.info("Created connection with the Mongo db client")
        except Exception as e:
            logging.error(f"Error happened in the CreditData class - {e}")
            raise e
    
    def save_csv_file(self, file_path, collection_name:str, database_name:Optional[str] = None):
        """
            This function is used to send the data from csv file into Mongodb database
        """
        try:
            df = pd.read_excel(file_path)
            records = loads(df.to_json(orient='records'))
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            collection.insert_many(records)
        except Exception as e:
            logging.error(f"Error happened in the save_csv_file method of the CreditData class- {e}")
            raise e
    
    def export_collection_as_dataframe(self, collection_name:str, database_name:Optional[str] = None) -> pd.DataFrame:
        """
            This function is used to fetch the data from mongo db and store it as a dataframe
        """
        try:
            logging.info(f"Enterd the export_collection_as_dataframe method of the CreditData class")
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Dataframe created in export_collection_as_dataframe method in CreditData class is {df.shape} shape")

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis = 1)
            df.replace({"na":np.nan}, inplace = True)
            return df
        except Exception as e:
            logging.error(f"Error happened in the export_collection_as_dataframe in the Creditdata class - {e}")