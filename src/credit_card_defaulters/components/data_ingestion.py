from src.credit_card_defaulters.entity.config_entity import DataIngestionConfig
from src.credit_card_defaulters.logger import logging
import pandas as pd
from src.credit_card_defaulters.constants.data_access.credit_card_data import CreditData
import os
from sklearn.model_selection import train_test_split
from src.credit_card_defaulters.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config = DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            logging.info(f"Error happened in the init method of DataIngestion class - {e}")
    
    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
            Export mongodb collection record as dataframe in the feature store folder
        """
        try:
            logging.info("Entered the export_data_into_feature_store method")
            credit_data = CreditData()

            df = credit_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Dataframe as received from the export_data_into_feature_store method in DataIngestionClass is of shape {df.shape}")

            # Get the file path to store the dataframe
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            
            # creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            df.to_csv(feature_store_file_path, index = False, header = True)
            return df
        except Exception as e:
            logging.error(f"Error happened in the export_data_into_feature_store method of DataIngestion class - {e} ")
            raise e
    
    def split_data_as_train_and_test(self, df:pd.DataFrame) -> None:
        """
            This method will split the data into train ans test set
        """
        try:
            logging.info("Entered the split_data_as_train_and_test method of DataIngestion class")
            train_set, test_set = train_test_split(df,
                                                   train_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Train test split complete")

            # creating directory to store the train and test set
            dir_name = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_name)

            # Store the data in respective files
            train_set.to_csv(self.data_ingestion_config.training_file_path, index = False, header = True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index =False, header = True)
            logging.info("Train and test files stored")
        except Exception as e:
            logging.error("Exception happened in split_data_as_train_and_test method of DataIngestion class")
            raise e
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Initiating data ingestion")
            df = self.export_data_into_feature_store()
            self.split_data_as_train_and_test(df)
            data_ingestion_artifact= DataIngestionArtifact(
                                trained_file_path=self.data_ingestion_config.training_file_path,
                                test_file_path=self.data_ingestion_config.test_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            logging.error("Error encountered in initate_data_ingestion method of the DataIngestion class")
            raise e





