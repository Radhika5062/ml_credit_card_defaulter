from src.credit_card_defaulters.logger import logging
from src.credit_card_defaulters.entity.config_entity import ModelTrainerConfig
from src.credit_card_defaulters.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
import os
from src.credit_card_defaulters.utils.common import load_numpy_array_data
from src.credit_card_defaulters.constants.ml.metric.classification_metric import get_classification_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from src.credit_card_defaulters.utils.common import save_object, load_object
from src.credit_card_defaulters.ml.model.estimator import CreditModel


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info("Entered the init method of the ModelTrainer class")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            logging.error(f"Error occurred in the init method of the ModelTrainer")
    
    def get_best_params_for_xgboost(self, train_x, train_y):
        logging.info("Entered the get_best_params_for_xgboost method in the ModelTrainer class")
        try:
            self.param_grid_xgboost = {
                'max_depth': [6, 8, 10],
                'learning_rate': [0.1, 0.001],
                'gamma':[0, 10, 20, 30]
            }
            self.grid_xgb = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic'),
                                     param_grid=self.param_grid_xgboost, 
                                     verbose = 3, 
                                     cv =5, 
                                     n_jobs = -1)
            self.grid_xgb.fit(train_x, train_y)
            logging.info(f"Got the grid fit done in xgboost")
            #extracting best features
            self.max_depth = self.grid_xgb.best_params_['max_depth']
            self.learning_rate = self.grid_xgb.best_params_['learning_rate']
            self.gamma = self.grid_xgb.best_params_['gamma']
            # self.sampling_method = self.grid_xgb.best_params_['sampling_method']
            logging.info(f"Xgboost best params are - {self.grid_xgb.best_params_}")
            # creating a new model with best params
            self.xgb = XGBClassifier(max_depth = self.max_depth
                                     ,learning_rate = self.learning_rate
                                     #,sampling_method = self.sampling_method
                                     ,objective='binary:logistic'
                                     ,gamma = self.gamma
                                     )
            logging.info(f"XGB classifier created")
            self.xgb.fit(train_x, train_y)
            logging.info(f"Xgboost best params are - {self.grid_xgb.best_params_}")
            return self.xgb
        except Exception as e:
            logging.error(f"Error occured in the get_best_params_for_xgboost class - {e}")
    
    def get_best_model_gradient_boost(self, train_x, train_y):
        logging.info("Entered the get_best_model_gradient_boost method in the ModelTrainer class")
        try:
            self.param_grid_gradient_boost = {
                'learning_rate':[0.1, 0.01],
                'n_estimators':[100, 150]
            }
            self.grid_gb = GridSearchCV(estimator = GradientBoostingClassifier(), 
                                     param_grid=self.param_grid_gradient_boost,
                                     verbose = 3,
                                     cv = 2,
                                     n_jobs = -1)
            self.grid_gb.fit(train_x, train_y)
            self.learning_rate = self.grid_gb.best_params_['learning_rate']
            self.n_estimators = self.grid_gb.best_params_['n_estimators']
            
            self.gb = GradientBoostingClassifier(learning_rate=self.learning_rate,
                                                 n_estimators=self.n_estimators
                                                 )
            self.gb.fit(train_x, train_y)
            logging.info(f"Gradient Boost best params = {self.grid_gb.best_params_}")
            return self.gb
        except Exception as e:
            logging.error(f"Error occured in the get_best_model_gradient_boost of ModelTrainer class - {e}")
    
    def get_best_model(self, train_x, train_y, test_x, test_y):
        logging.info("Entered the get_best_model method of the ModelTrainer")
        try:
            selected_model = ''
            # xgboost model
            self.xgboost = self.get_best_params_for_xgboost(train_x=train_x, train_y=train_y)
            self.prediction_xgboost_testing = self.xgboost.predict(test_x)
            self.prediction_xgboost_training = self.xgboost.predict(train_x)
            
            self.xgboost_classification_metric_testing = get_classification_score(y_true=test_y, y_pred=self.prediction_xgboost_testing)
            self.xgboost_classification_metric_training = get_classification_score(y_true=train_y, y_pred=self.prediction_xgboost_training)
            logging.info(f"Classification metric for xgboost testing= {self.xgboost_classification_metric_testing}")
            logging.info(f"Classification metric for xgboost training= {self.xgboost_classification_metric_training}")

            # gradient boost model
            self.gradient_boost = self.get_best_model_gradient_boost(train_x=train_x, train_y=train_y)
            self.prediction_gradient_boost_testing = self.gradient_boost.predict(test_x)
            self.prediction_gradient_boost_training = self.gradient_boost.predict(train_x)

            self.gradient_classification_metric_testing = get_classification_score(y_pred=self.prediction_gradient_boost_testing, y_true=test_y)
            self.gradient_classification_metric_training = get_classification_score(y_pred=self.prediction_gradient_boost_training, y_true=train_y)
            logging.info(f"Classification metric for gradient boost testing= {self.gradient_classification_metric_testing}")
            logging.info(f"Classification metric for gradient boost training= {self.gradient_classification_metric_training}")

            logging.info("Comparing the two models")
            logging.info(f"self.xgboost_classification_metric_training.recall_score = {self.xgboost_classification_metric_training.recall_score}")
            logging.info(f"self.gradient_classification_metric_training.recall_score = {self.gradient_classification_metric_training.recall_score}")
            if self.xgboost_classification_metric_testing.recall_score < self.gradient_classification_metric_testing.recall_score:
                selected_model = self.gradient_boost 
            else:
                selected_model = self.xgboost
            
            self.classification_metric_training = selected_model.predict(train_x)
            logging.info(f"Selected model - {selected_model}")
            return selected_model
        except Exception as e:
            logging.error(f"Error occured in the get_best_model method of the ModelTrainer Class - {e}")


    def overfitting_underfitting_check(self, training_score, testing_score):
        try:
            logging.info("Entered the overfitting_underfitting_check")
            if training_score < self.model_trainer_config.expected_accuracy:
                logging.error(f"Model has not provided expected accuracy. Training_score = {training_score} and expected accuracy = {self.model_trainer_config.expected_accuracy}")
            diff = abs(training_score - testing_score)
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.error(f"Model is overfitting or underfitting")
        except Exception as e:
            logging.error(f"Error occurred in the overfitting_underfitting_check method of the ModelTrainer class - {e}")

    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"Entered the intiate_model_trainer method of the ModelTrainer class")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            model = self.get_best_model(x_train, y_train, x_test, y_test)

            y_train_pred = model.predict(x_train)

            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            y_test_pred = model.predict(x_test)

            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            self.overfitting_underfitting_check(training_score=classification_train_metric.f1_score,
                                                testing_score=classification_test_metric.f1_score)
            
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trainer_model_path)
            os.makedirs(model_dir_path, exist_ok=True)
            credit_model = CreditModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trainer_model_path, obj=credit_model)

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trainer_model_path,
                                                          train_metric_artifact=classification_train_metric,
                                                          test_metric_artifact=classification_test_metric)
            logging.info(f"Model trainer artifact = {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error occured in the intiate_model_trainer method of the ModelTrainer class - {e}")

                