# All the values which are constant and are required for training will be kept here

import os
from credit_card_defaulters.constants.database import DATABASE_NAME, COLLECTION_NAME

TARGET_COLUMN_NAME = 'default payment next month'
ARTIFACT_DIR = "artifacts"
PIPELINE_NAME = 'credit_card'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
FILE_NAME = 'credit_card.csv'
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
SAVED_MODE_DIR = os.path.join("saved_model_dir")
PREPROCESSING_OBJECT_FILE_NAME = 'preprocessing.pkl'
CLUSTERING_FILE_NAME = 'clustering.pkl'
MODEL_FILE_NAME = "model.pkl"
CLUSTER_LABEL_COLUMN_NAME = "Cluster_labels"

# data ingestion related config
DATA_INGESTION_COLLECTION_NAME:str = COLLECTION_NAME
DATA_INGESTION_DIR_NAME:str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str = "feature_store"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:str = 0.20

# data validation related constants
DATA_VALIDATION_DIR_NAME:str = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "validated"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"
 
# data transformation related constamts
DATA_TRANSFORMATION_DIR_NAME:str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str = "transformd_object"
DATA_TRANSFORMATION_ELBOW_PLOT_FILE_NAME:str = 'elbow_plot.png'

# model training related constants
MODEL_TRAINER_DIR_NAME:str = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR:str = 'trained_model'
MODEL_TRAINER_TRAINED_MODEL_NAME:str = 'model.pkl'
MODEL_TRAINER_EXPECTED_SCORE:float = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD:float = 0.05

# model evaluation related constants
MODEL_EVALUATION_DIR_NAME:str = 'model_evaluation'
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE:float = 0.2
MODEL_EVALUATION_REPORT_NAME:str = "report.yaml"

#Model pusher relared constants
MODEL_PUSHER_DIR_NAME:str = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = SAVED_MODE_DIR

# prediction related constants
PREDICTION_DIR_NAME = "prediction"
PREDICTION_CSV_FILE_PATH = 'prediction_dataset.csv'