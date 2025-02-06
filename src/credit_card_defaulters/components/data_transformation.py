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
            
            full_df = pd.concat([train_df, test_df])

            # Apply clustering
            number_of_clusters_training =  self.clustering.elbow_plot(full_df)
            logging.info(f"Number of clusters obtained = {number_of_clusters_training}")

            # Add a column to which shows the cluster number for train_df
            train_df = self.clustering.create_clusters(train_df, number_of_clusters_training)
            logging.info(f"Added cluster labels to the train_df dataframe")

            # Add a column which shows the cluster number for test_df
            test_df = self.clustering.create_clusters(test_df, number_of_clusters_training)
            logging.info(f"Added cluster labels to the test_df dataframe")

            # getting unique number of clusters from dataset
            list_of_clusters_train_df = train_df[CLUSTER_LABEL_COLUMN_NAME].unique()
            list_of_clusters_test_df = test_df[CLUSTER_LABEL_COLUMN_NAME].unique()
            logging.info(f"list_of_clusters_train_df = {list_of_clusters_train_df}")
            logging.info(f"list_of_clusters_test_df = {list_of_clusters_test_df}")

            # Parsing all clusters and looking for the best ML to fit the individual cluster
            for i in list_of_clusters_train_df:
                # Separating the inependent and dependent features for train data and test data
                if i in list_of_clusters_train_df:
                    cluster_data_train = train_df[train_df[CLUSTER_LABEL_COLUMN_NAME] == i]
                    train_df_X, train_df_y = self.separate_features(cluster_data_train)
                if i in list_of_clusters_test_df:
                    cluster_data_train = test_df[test_df[CLUSTER_LABEL_COLUMN_NAME] == i]
                    test_df_X, test_df_y = self.separate_features(cluster_data_train)

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

                # Saving numpy array for this cluster
                logging.info(f"Saving numpy array for cluster - {i}")
                save_numpy_array_data(self.data_transformation_config.data_transformation_train_file_path
                                      .replace('npy', f'{i}.npy'), train_arr)
                save_numpy_array_data(self.data_transformation_config.data_transformation_test_file_path
                                        .replace('npy',f'{i}.npy'), test_arr)
                save_object(self.data_transformation_config.transformed_preprocessor_object_file_path
                            .replace('pkl', f'{i}.pkl'),preprocessor_object )
                
            data_transformation_artifact = DataTransformationArtifact(
                transformed_clustering_object_file_path=self.data_transformation_config.     transformed_clustering_model_object_file_path,
                transformed_data_dir = self.data_transformation_config.data_transformation_transformed_dir
            )
            logging.info(f"Data transformation artifact = {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Error occured in the initiate_data_transformation method of the DataTransformation Class - {e}")
            raise e


                

                








        

            

            


