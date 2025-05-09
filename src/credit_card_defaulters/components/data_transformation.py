from src.credit_card_defaulters.logger import logging
import numpy as np
import pandas as pd
from src.credit_card_defaulters.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.credit_card_defaulters.entity.config_entity import DataTransformationConfig
from src.credit_card_defaulters.constants.training_pipeline import TARGET_COLUMN_NAME, CLUSTER_LABEL_COLUMN_NAME
from src.credit_card_defaulters.constants.clustering import ClusteringModel
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from src.credit_card_defaulters.utils.common import save_numpy_array_data, save_object
import os

class DataTransformation:
    def __init__(self,
                 data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            logging.info(f"Entered the init method of the DataTransformation class")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.clustering = ClusteringModel(self.data_transformation_config)
        except Exception as e:
            logging.error(f"Error occured in the init method of the DataTransformation class - {e}")
            raise e
    
    def separate_features(self, data, label_column = [TARGET_COLUMN_NAME, CLUSTER_LABEL_COLUMN_NAME]):
        """
            This method is used to split the independent and dependent variables
        """
        try:
            logging.info(f"Entered the separate_label_features of the DataTransformation Class")
            X = data.drop(label_column, axis = 1)
            y = data[label_column[0]]
            return X, y
        except Exception as e:
            logging.error(f"Error occurred in the separate_label_features method of the DataTransformation class - {e}")
            raise e
    
    @staticmethod
    def read_data(file_name) ->pd.DataFrame:
        try:
            logging.info(f"Entered the read_data method of the DataTransformation class")
            return pd.read_csv(file_name)
        except Exception as e:
            logging.error(f"Error occurred in the read_data method of the DataTransformation class - {e}")
            raise e
    
    @classmethod
    def get_data_transformer_object(cls)-> Pipeline:
        try:
            logging.info(f"Entered the get_data_transformation_object of the DataTransformation class")
            robust_scaler = RobustScaler()
            preprocessor = Pipeline(
                steps = [
                    ("RobustScaler",robust_scaler)
                ]
            )
            return preprocessor
        except Exception as e:
            logging.error(f"Error occured in the get_data_transformer_object method of the DataTransformation class - {e}")
            raise e

    
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Entered the initiate_data_transformation method of the DataTransformation class")

            preprocessor = self.get_data_transformer_object()

            # Get the train and test datasets
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            train_df_X = train_df.drop(columns = [TARGET_COLUMN_NAME], axis = 1)
            train_df_y = train_df[TARGET_COLUMN_NAME]

            test_df_X = test_df.drop(columns=[TARGET_COLUMN_NAME], axis =1)
            test_df_y = test_df[TARGET_COLUMN_NAME]

            # Now we will do fit on the train dataset and transform on the test dataset
            preprocessor_object = preprocessor.fit(train_df_X)
            transformed_train_df_X = preprocessor_object.transform(train_df_X)
            transformed_test_df_X = preprocessor_object.transform(test_df_X)

            # Now we will handle the imbalanced dataset
            smt = SMOTETomek(sampling_strategy='minority')

            train_df_final_X, train_df_final_y = smt.fit_resample(
                transformed_train_df_X,
                train_df_y
            )
            test_df_final_X, test_df_final_y = smt.fit_resample(
                transformed_test_df_X,
                test_df_y
            )

            # converting data into array
            train_arr = np.c_[train_df_final_X, np.array(train_df_final_y)]
            test_arr = np.c_[test_df_final_X, np.array(test_df_final_y)]
            

            save_numpy_array_data(self.data_transformation_config.data_transformation_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.data_transformation_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_preprocessor_object_file_path,preprocessor_object )
                
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_preprocessor_object_file_path,
                transformed_train_file_path = self.data_transformation_config.data_transformation_train_file_path,
                transformed_test_file_path=self.data_transformation_config.data_transformation_test_file_path
            )
            logging.info(f"Data transformation artifact = {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Error occured in the initiate_data_transformation method of the DataTransformation Class - {e}")
            raise e


                

                








        

            

            


