from src.credit_card_defaulters.entity.config_entity import PredictionConfig
from src.credit_card_defaulters.logger import logging
from src.credit_card_defaulters.constants import training_pipeline
from src.credit_card_defaulters.utils.common import read_yaml_file
import pandas as pd
import numpy as np
from src.credit_card_defaulters.utils.common import load_object
from src.credit_card_defaulters.ml.model.estimator import ModelResolver
import os


class Prediction:
    def __init__(self, prediction_config:PredictionConfig, pred_file=''):
        try:
            logging.info("Entered the init method of the Prediction class")
            self.prediction_config = prediction_config
            self._schema_config = read_yaml_file(training_pipeline.SCHEMA_FILE_PATH)
            self.pred_file = pred_file
        except Exception as e:
            logging.error(f"Error occurred in the init method of the Prediction class")
    
    def validate_number_of_columns(self, df:pd.DataFrame)->bool:
        """
            This method validated the number of columns present in dataframe with the number of columns mentioned in schema file
        """
        try:
            logging.info("Entered the validate_number_of_columns method in PredictionPipeline class")
            total_number_of_columns_in_df = len(df.columns)
            total_number_of_columns_in_schema_file = len(self._schema_config["prediction_column_names"])
            logging.info(f"Number of columns in dataframe = {total_number_of_columns_in_df}")
            logging.info(f"Number of columns mentioned in schema file - {total_number_of_columns_in_schema_file}")

            if total_number_of_columns_in_schema_file == total_number_of_columns_in_df:
                logging.info(f"Number of columns match")
                return True
            logging.info("Number of columns do not match")
            return False
        except Exception as e:
            logging.error(f"Error happened in validate_number_of_columns method of the Prediction class")
    
    def validate_name_of_columns(self, df:pd.DataFrame) -> bool:
        """
            This function validates the name of the columns present in the dataframe with the ones present in the schema file
        """
        try:
            logging.info("Entered the validate_name_of_columns method of the Prediction class")
            dataframe_columns = df.columns
            schema_file_columns = self._schema_config["prediction_column_names"]
            missing_columns_list = []

            status = True
            for col in dataframe_columns:
                if col not in schema_file_columns:
                    status = False
                    missing_columns_list.append(col)
            logging.info(f"Missing columns are = {missing_columns_list}")
            return status
        except Exception as e:
            logging.error(f"Error happened in the validate_name_of_columns method of the Prediction class - {e}")
            raise e
    
    @staticmethod
    def read_csv(file_name)->pd.DataFrame:
        """
            This is a static method that is used to read a csv file via pandas library and get the data in the form of a dataframe
        """
        try:
            logging.info(f"Reading file = {file_name}")
            return pd.read_csv(file_name, header = 0)
        except Exception as e:
            logging.error(f"Error happened in the read_csv method of the Prediction class")
            raise e
    
    # def get_latest_artifact(self) ->str:
    #     try:
    #         logging.info("Entered the get_latest_artifact of the Prediction class")
    #         timestamps = list(map(int, os.listdir(self.prediction_config.artifact_dir)))
    #         latest_timestamp = max(timestamps)
    #         latest_path = os.path.join(self.prediction_config.artifact_dir, f"{latest_timestamp}")
    #         logging.info(f"latest_model_path = {latest_path}")
    #         return latest_path
    #     except Exception as e:
    #         logging.info(f"Error occured in the get_latest_artifact method of the Prediction class")
    #         raise e
        

    
    # def apply_data_transformation(self):
    #     try:
    #         logging.info("Entered the apply_data_transformation method of the Prediction class")
    #         load_object(self.prediction_config.transformed_object_file_path)

    def get_best_model(self) ->str:
        try:
            logging.info("Entered the get_best_model of the Prediction class")
            timestamps = list(map(int, os.listdir(training_pipeline.SAVED_MODE_DIR)))
            latest_timestamp = max(timestamps)
            latest_model_path = os.path.join(training_pipeline.SAVED_MODE_DIR, f"{latest_timestamp}",training_pipeline.MODEL_FILE_NAME)
            logging.info(f"latest_model_path = {latest_model_path}")
            return latest_model_path
        except Exception as e:
            logging.info(f"Error occured in the get_best_model method of the Prediction class")
            raise e



    
    def predict_output(self):
        try:
            if self.pred_file == '':
                logging.info(f"pred file is not present. Use default")
                file_path = self.prediction_config.get_prediction_file_path
            else:
                logging.info(f"pred file is present. Use this")
                file_path = os.path.join(
                                         self.pred_file)
            logging.info(f"Prediction file path = {file_path}")

            df = Prediction.read_csv(file_path)
            df.replace({"na":"np.nan"}, inplace = True)
            logging.info(f"Final dataframe = {df}")
            logging.info(f"Validate csv file")
            self.validate_number_of_columns(df)
            self.validate_name_of_columns(df)

            latest_model_path = self.get_best_model()
            logging.info(f"Latest model path in Prediction class - {latest_model_path}")
            model = load_object(latest_model_path)
            logging.info(f"Model = {model}")
            if self.validate_name_of_columns and self.validate_number_of_columns:
                logging.info(f"Latest transformation object file = {self.prediction_config.transformed_object_file_path}")
                transformation_object = load_object(self.prediction_config.transformed_object_file_path)
                logging.info(f"transformation object = {transformation_object}")
                transformed_object = transformation_object.transform(df)
                logging.info(f"Transformed object - {transformed_object}")
                prediction = model.predict(transformed_object)
                logging.info(f"Prediction - {prediction}")
                return prediction
            logging.info("Validation error")
            return "Validation error. Check the file format"
        except Exception as e:
            logging.error(f"Error occurred in the predict_output of Predictiom class - {e}")



