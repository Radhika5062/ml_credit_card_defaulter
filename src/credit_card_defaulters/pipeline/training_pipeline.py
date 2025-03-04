from src.credit_card_defaulters.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from src.credit_card_defaulters.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact, ModelEvaluationArtifact
from src.credit_card_defaulters.logger import logging
from src.credit_card_defaulters.components.data_ingestion import DataIngestion
from src.credit_card_defaulters.components.data_validation import DataValidation
from src.credit_card_defaulters.components.data_transformation import DataTransformation
from src.credit_card_defaulters.components.model_trainer import ModelTrainer
from src.credit_card_defaulters.components.model_evaluation import ModelEvaluation
from src.credit_card_defaulters.components.model_pusher import ModelPusher
from src.credit_card_defaulters.cloud_storage.s3_syncer import s3Sync
from src.credit_card_defaulters.constants.s3_bucket import TRAINING_BUCKET_NAME
from src.credit_card_defaulters.constants.training_pipeline import SAVED_MODE_DIR


class TrainingPipeline:
    is_pipeline_running = False

    s3_sync = s3Sync()

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Entered the start_data_ingestion method of the TrainingPipeline class")
            self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            logging.error(f"Error happened in the start_data_ingestion method of the TrainingPipeline - {e}")
    
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Entered the start_data_validation method of the TrainingPipeline class")
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, 
                                             data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            logging.error(f"Error happened in the start_data_validation method of the TrainingPipeline class - {e}")
            raise e
    
    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact):
        try:
            logging.info("Entered the start_data_transformation method of the TrainingPipeline class")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Error occured in the start_data_transformation method  of the TrainingPipeline class - {e}")
            raise e
    
    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info("Entered the start_model_trainer method of the TrainingPipeline class")
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error happened in the start_model_trainer method of the TrainingPipeline class - {e}")
    
    def start_model_evaluation(self, 
                               model_trainer_artifact:ModelTrainerArtifact,
                               data_validation_artifact:DataValidationArtifact
                               ):
        try:
            logging.info("Entered the start_model_evaluation method of the TrainingPipeline class")
            model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=self.training_pipeline_config)
            model_evaluation = ModelEvaluation(model_evaluation_config=model_evaluation_config, 
                                               data_validation_artifact = data_validation_artifact,
                                               model_trainer_artifact=model_trainer_artifact)
            
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            logging.error(f"Error occured in the start_model_evaluation method of the TrainingPipleine class")
    
    def start_model_pusher(self, model_eval_artifact:ModelEvaluationArtifact):
        try:
            logging.info(f"Entered the start_model_pusher method of the TrainingPipeline class")
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config=model_pusher_config, 
                                       model_evaluation_artifact=model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            logging.error(f"Error occurred in the start_model_pusher class of the Training Pipeline - {e}")
    
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir, aws_bucket_url = aws_bucket_url)
        except Exception as e:
            logging.error(f"Error occured in the sync_artifact_dir_to_s3 method of the TrainingPipeline class - {e}")
    
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODE_DIR}"
            self.s3_sync.sync_folder_to_s3(folder = SAVED_MODE_DIR, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            logging.error(f"Error occured in the sync_saved_model_dir_to_s3 method of TrainingPipeline class - {e}")
        

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact,
                                                                    data_validation_artifact=data_validation_artifact
                                                                    )
            logging.info(f"Model Evaluation artifact is model accepted = {model_evaluation_artifact.is_model_accepted}")
            if model_evaluation_artifact.is_model_accepted:
                logging.error(f"Trained model is better than the best model")
            model_pusher_artifact = self.start_model_pusher(model_eval_artifact=model_evaluation_artifact)

            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            TrainingPipeline.is_pipeline_running = False
        except Exception as e:
            self.sync_artifact_dir_to_s3()
            TrainingPipeline.is_pipeline_running = False
            logging.error(f"Error happened in the run_pipeline method of the TrainingPipeline class - {e}")
            