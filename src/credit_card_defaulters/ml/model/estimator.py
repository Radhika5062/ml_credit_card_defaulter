from src.credit_card_defaulters.logger import logging
from src.credit_card_defaulters.constants.training_pipeline import SAVED_MODE_DIR
import os
from src.credit_card_defaulters.constants.training_pipeline import MODEL_FILE_NAME

class CreditModel:
    def __init__(self, preprocessor, model):
        try:
            logging.info("Entered the init method of the CreditModel class")
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            logging.info(f"Error happened in the init method of the CreditModel class - {e}")
            raise e
    
    def predit(self, x):
        try:
            logging.info(f"Entered the predict method of the CreditModel class")
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            logging.info(f"Entered the predict method of the CreditModel class - {3}")
            raise e
    
class ModelResolver:
    def __init__(self, model_dir = SAVED_MODE_DIR):
        try:
            logging.info("Entered the init method of the ModelResolver class")
            self.model_dir = model_dir
        except Exception as e:
            logging.info("Error occurred in the init method of the ModelResolver class")
            raise e
        
    def does_model_exists(self)->bool:
        try:
            logging.info(f"Entered the does_model_exists method of the ModelResolver class")
            if not os.path.exists(self.model_dir):
                return False
            timestamps = os.listdir(self.model_dir)
            if len(timestamps) ==0:
                return False
            latest_model_path = self.get_best_model()

            if not os.path.exists(latest_model_path):
                return False
            
            return True
        except Exception as e:
            logging.info(f"Error occured in the does_model_exists method of the ModelResolver class")
            raise e
    
    def get_best_model(self) ->str:
        try:
            logging.info("Entered the get_best_model of the ModelResolver class")
            timestamps = list(map(int, os.listdir(self.model_dir)))
            latest_timestamp = max(timestamps)
            latest_model_path = os.path.join(self.model_dir, f"{latest_timestamp}",MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            logging.info(f"Error occured in the get_best_model method of the ModelResolver class")
            raise e
        

        
