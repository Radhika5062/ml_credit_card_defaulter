from src.credit_card_defaulters.logger import logging
from src.credit_card_defaulters.entity.config_entity import PredictionConfig
import os
from src.credit_card_defaulters.constants.training_pipeline import ARTIFACT_DIR
from src.credit_card_defaulters.components.prediction import Prediction
class PredictionPipeline:
    def __init__(self, pred_file=''):
        self.pred_file = pred_file

    def get_latest_artifact(self) ->str:
        try:
            logging.info("Entered the get_latest_artifact of the Prediction class")
            timestamps = list(map(int, os.listdir(ARTIFACT_DIR)))
            latest_timestamp = max(timestamps)
            latest_path = os.path.join(ARTIFACT_DIR, f"{latest_timestamp}")
            logging.info(f"latest_model_path = {latest_path}")
            return latest_path
        except Exception as e:
            logging.info(f"Error occured in the get_latest_artifact method of the PredictionPipeline class - {e}")
            raise e

    
    def start_prediction_pipeline(self):
        try:
            latest_artifact_dir = self.get_latest_artifact()
            self.prediction_pipeline_config = PredictionConfig(latest_artifact_dir)
            prediction_obj = Prediction(self.prediction_pipeline_config, self.pred_file)
            prediction = prediction_obj.predict_output()
            logging.info(f"Prediction in the PredictionPipeline = {prediction}")
            return prediction
        except Exception as e:
            logging.info(f"Error occured in the start_prediction_pipeline method of PredictionPipeline class - {e}")