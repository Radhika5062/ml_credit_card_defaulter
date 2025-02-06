from src.credit_card_defaulters.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.credit_card_defaulters.entity.config_entity import DataValidationConfig
import pandas as pd
from src.credit_card_defaulters.logger import logging
from src.credit_card_defaulters.utils.common import read_yaml_file
from src.credit_card_defaulters.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import os 
from src.credit_card_defaulters.utils.common import write_yaml_file
from src.credit_card_defaulters.constants.training_pipeline import TRAIN_FILE_NAME, TEST_FILE_NAME

class DataValidation:
    def __init__(self,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            logging.info("Entered the init method of the DataValidation class")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info(f"Schema file path = {SCHEMA_FILE_PATH}")
            logging.info(f"Current working directory is = {os.getcwd()}")
        except Exception as e:
            logging.error(f"Error happened in the init method of the DataValidation class")
    
    def validate_number_of_columns(self, df:pd.DataFrame)->bool:
        """
            This method validated the number of columns present in dataframe with the number of columns mentioned in schema file
        """
        try:
            logging.info("Entered the validate_number_of_columns method of the DataValidation class")
            total_number_of_columns_in_df = len(df.columns)
            total_number_of_columns_in_schema_file = len(self._schema_config["column_names"])
            logging.info(f"Number of columns in dataframe = {total_number_of_columns_in_df}")
            logging.info(f"Number of columns mentioned in schema file - {total_number_of_columns_in_schema_file}")

            if total_number_of_columns_in_schema_file == total_number_of_columns_in_df:
                logging.info(f"Number of columns match")
                return True
            logging.info("Number of columns do not match")
            return False
        except Exception as e:
            logging.error(f"Error happened in validate_number_of_columns method of the DataValidation class")
    
    def validate_name_of_columns(self, df:pd.DataFrame) -> bool:
        """
            This function validates the name of the columns present in the dataframe with the ones present in the schema file
        """
        try:
            logging.info("Entered the validate_name_of_columns method of the DataValidation class")
            dataframe_columns = df.columns
            schema_file_columns = self._schema_config["column_names"]
            missing_columns_list = []

            status = True
            for col in dataframe_columns:
                if col not in schema_file_columns:
                    status = False
                    missing_columns_list.append(col)
            logging.info(f"Missing columns are = {missing_columns_list}")
            return status
        except Exception as e:
            logging.error(f"Error happened in the validate_name_of_columns method of the DataValidation class - {e}")
            raise e
    
    @staticmethod
    def read_csv(file_name)->pd.DataFrame:
        """
            This is a static method that is used to read a csv file via pandas library and get the data in the form of a dataframe
        """
        try:
            logging.info(f"Reading file = {file_name}")
            return pd.read_csv(file_name)
        except Exception as e:
            logging.error(f"Error happened in the read_csv method of the DataValidation class")
            raise e
    
    def detect_dataset_drift(self, base_df, current_df, thresold = 0.05)-> bool:
        """
            This method is used to detect drift in the dataset
        """
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if thresold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({
                    column:{
                        "p_value":float(is_same_dist.pvalue),
                        "drift_status":is_found
                    }
                })
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            dir_name = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_name, exist_ok=True)
            write_yaml_file(data = report, file_path=drift_report_file_path)
            return status
        except Exception as e:
            logging.error(f"Error happened in the detect_dataset_drift method if the DataValidation class")
            raise e

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            logging.info("Entered the initiate_data_validation method of the DataValidation class")
            error_message = ""
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read the train and test files
            train_df = DataValidation.read_csv(train_file_path)
            test_df = DataValidation.read_csv(test_file_path)

            # Start validation one by one

            status = self.validate_number_of_columns(df=train_df)
            if not status:
                error_message = f"{error_message} Train dataframe does not contain all the columns."
            status = self.validate_number_of_columns(df=test_df)
            if not status:
                error_message = f"{error_message} Test dataframe does not contain all the columns"

            status = self.validate_name_of_columns(df = train_df)
            if not status:
                error_message = f"{error_message} Train dataframe column names are different"
            status = self.validate_name_of_columns(df=test_df)
            if not status:
                error_message = f"{error_message} Test dataframe column names are different"

            if len(error_message) > 0:
                status = False
                logging.error(f"VALIDATION FAILED - {error_message}")
                raise Exception(error_message)
            
            # Check for data drift
            status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            # Storing the valid and invalid data
            if status == False:
                invalid_train_file_path = self.data_validation_config.invalid_train_file
                invalid_test_file_path = self.data_validation_config.invalid_train_file
            else:
                invalid_train_file_path = None
                invalid_test_file_path = None

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=invalid_train_file_path,
                invalid_test_file_path=invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )         
            logging.info(f"Data validation artifact - {data_validation_artifact}")
            return data_validation_artifact 
        except Exception as e:
            logging.error(f"Error happened in initiate_data_validation method of the DataValidation class")
            raise e 
