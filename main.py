from src.credit_card_defaulters.logger import logging
from utils import dump_csv_file_to_mongodb_collection
from credit_card_defaulters.constants.database import DATABASE_NAME, COLLECTION_NAME
from src.credit_card_defaulters.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    # file_name = 'data.xls'
    # database_name = DATABASE_NAME
    # collection_name = COLLECTION_NAME
    # dump_csv_file_to_mongodb_collection(file_path=file_name, database_name=database_name, collection_name=collection_name)
    training_pipeline = TrainingPipeline()
    training_pipeline.run_pipeline()