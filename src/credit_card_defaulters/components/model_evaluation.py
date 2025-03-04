from src.credit_card_defaulters.logger import logging
from src.credit_card_defaulters.entity.config_entity import ModelEvaluationConfig
from src.credit_card_defaulters.entity.artifact_entity import ModelTrainerArtifact, DataValidationArtifact, ModelEvaluationArtifact
import os
import datetime
from pathlib import Path
from src.credit_card_defaulters.utils.common import get_latest_folder
import pandas as pd
from src.credit_card_defaulters.constants.training_pipeline import TARGET_COLUMN_NAME
import pandas as pd
from src.credit_card_defaulters.ml.model.estimator import ModelResolver
from src.credit_card_defaulters.utils.common import load_object
from src.credit_card_defaulters.constants.ml.metric.classification_metric import get_classification_score
from src.credit_card_defaulters.utils.common import write_yaml_file
import yaml
import numpy as np

class ModelEvaluation:
    def __init__(self,
                model_evaluation_config:ModelEvaluationConfig,
                data_validation_artifact:DataValidationArtifact,
                model_trainer_artifact:ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            logging.error(f"Error occured in the init method of the ModelEvaluation class - {e}")

    
    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            logging.info("Entered the initiate_model_evaluation method of ModelEvaluation Class")
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            df = pd.concat([train_df, train_df])

            y_true = df[TARGET_COLUMN_NAME]
            
            df.drop(TARGET_COLUMN_NAME, axis=1, inplace=True)

            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()

            is_model_accepted = True
            
            if not model_resolver.does_model_exists():
                logging.info("In the if block of not model_resolver.does_model_exists()")
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=None,
                    best_model_path=None,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None
                )
                logging.info(f"Model evaluation artifact = {model_evaluation_artifact}")
                return model_evaluation_artifact
            
            logging.info("Working with latest model")
            latest_model_path = model_resolver.get_best_model()

            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)

            y_trained_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            trained_metric = get_classification_score(y_true=y_true, y_pred=y_trained_pred)
            latest_metric = get_classification_score(y_true=y_true, y_pred=y_latest_pred)

            improved_accuracy = float(trained_metric.f1_score - latest_metric.f1_score)
            logging.info(f"Improved Accuracy = {improved_accuracy}")

            if self.model_evaluation_config.change_threshold < improved_accuracy:
                is_model_accepted = True
            else:
                is_model_accepted = False
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                trained_model_path=train_model_file_path,
                train_model_metric_artifact=trained_metric.__dict__,
                best_model_metric_artifact=latest_metric.__dict__            
                )
            
            model_eval_report = model_evaluation_artifact.__dict__
            
           
            write_yaml_file(model_eval_report, self.model_evaluation_config.report_file_path)
            logging.info(f"Model evaluation artifact - {model_evaluation_artifact}")
            return model_evaluation_artifact
        
        except Exception as e:
            logging.error(f"Error occured in the initate_model_Evaluation method of the ModelEvaluation class - {e}")


