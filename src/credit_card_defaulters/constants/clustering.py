from src.credit_card_defaulters.logger import logging
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
import os
from src.credit_card_defaulters.entity.config_entity import DataTransformationConfig
from src.credit_card_defaulters.constants.training_pipeline import DATA_TRANSFORMATION_ELBOW_PLOT_FILE_NAME
from src.credit_card_defaulters.utils.common import save_object
import pandas as pd
from src.credit_card_defaulters.constants.training_pipeline import CLUSTER_LABEL_COLUMN_NAME


class ClusteringModel:
    def __init__(self, data_transformation_config:DataTransformationConfig):
        self.data_transformation_config = data_transformation_config

    def elbow_plot(self, data):
        logging.info(f"Entered the elbow_plot method of the ClusteringModel class")
        wcss = [] #initialising am empty list
        try:
            for i in range(1, 11):
                # Initializing Kmeans object
                kmeans = KMeans(n_clusters=i, init = 'k-means++', random_state=42)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            # Creating graph between WCSS and number of clusters
            plt.plot(range(1, 11), wcss)
            plt.title("The elbow method")
            plt.xlabel("Number of clusters")
            plt.ylabel("WCSS")
            # Saving the elbow plot
            os.makedirs(os.path.join(
                            self.data_transformation_config.data_transformation_dir), exist_ok=True)
            plt.savefig(os.path.join(
                            self.data_transformation_config.data_transformation_dir,
                            DATA_TRANSFORMATION_ELBOW_PLOT_FILE_NAME
                                    ))
            self.kn = KneeLocator(range(1,11), wcss, curve='convex', direction = 'decreasing')
            logging.info(f"The ideal number of clusters in = {self.kn.knee}")
            return self.kn.knee
        except Exception as e:
            logging.info(f"Error encountered in the elbow_plot method of the clustering model class - {e}")
            raise e
    
    def create_clusters(self, data, number_of_clusters):
        try:
            self.data = data
            logging.info(f"Entered the create_clsuters method of the ClusteringModel class")
            logging.info(f"Datatype of self.data = {self.data.shape}")
            self.kmeans = KMeans(n_clusters = number_of_clusters, init = 'k-means++', random_state =42)
            self.y_predict = self.kmeans.fit_predict(self.data)
            # os.makedirs(self.data_transformation_config.transformed_clustering_model_object_file_path, exist_ok=True)
            self.save_model = save_object(self.data_transformation_config.transformed_clustering_model_object_file_path,
                                          self.kmeans)
            self.data[CLUSTER_LABEL_COLUMN_NAME] = self.y_predict
            return self.data
        except Exception as e:
            logging.info(f"Error occured in the create_clusters method of the ClusteringModel class")
            raise e
        