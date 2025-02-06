from src.credit_card_defaulters.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from src.credit_card_defaulters.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.credit_card_defaulters.logger import logging
from src.credit_card_defaulters.components.data_ingestion import DataIngestion
from src.credit_card_defaulters.components.data_validation import DataValidation
from src.credit_card_defaulters.components.data_transformation import DataTransformation
class TrainingPipeline:
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


    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
        except Exception as e:
            logging.info(f"Error happened in the run_pipeline method of the TrainingPipeline class - {e}")
            