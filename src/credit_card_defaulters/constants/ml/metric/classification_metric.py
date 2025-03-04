from src.credit_card_defaulters.logger import logging
from src.credit_card_defaulters.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def get_classification_score(y_true, y_pred)->ClassificationMetricArtifact:
    try:
        logging.info(f"Entered the get_classification_score method")
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_roc_auc_score = roc_auc_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            roc_auc_score=model_roc_auc_score
        )
        return classification_metric
    except Exception as e:
        logging.info(f"Error occured in get_classification_score method - {e}")