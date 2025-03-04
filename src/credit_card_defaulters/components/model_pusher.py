from src.credit_card_defaulters.logger import logging
from src.credit_card_defaulters.entity.config_entity import ModelPusherConfig
from src.credit_card_defaulters.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
import os
import shutil

class ModelPusher:
    def __init__(self, model_pusher_config:ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            logging.error(f"Error occured in the init method of the ModelPusher class - {e}")
    
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Entered the initiate_model_pusher method of the ModelPusher class")
            trained_model_path = self.model_evaluation_artifact.trained_model_path

            # creating model pusher dir to save model
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path), exist_ok= True)

            shutil.copy(src=trained_model_path, dst = model_file_path)

            saved_model_path= self.model_pusher_config.saved_model_path

            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)

            shutil.copy(src=trained_model_path, dst = saved_model_path)

            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path, model_file_path=model_file_path)
            return model_pusher_artifact
        except Exception as e:
            logging.error(f"Error occured in the intitate_model_pusher method of the ModelPusher class")