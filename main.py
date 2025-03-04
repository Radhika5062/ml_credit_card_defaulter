from src.credit_card_defaulters.logger import logging
from utils import dump_csv_file_to_mongodb_collection
from credit_card_defaulters.constants.database import DATABASE_NAME, COLLECTION_NAME
from src.credit_card_defaulters.pipeline.training_pipeline import TrainingPipeline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from src.credit_card_defaulters.pipeline.prediction_pipeline import PredictionPipeline
from src.credit_card_defaulters.constants.training_pipeline.application import APP_HOST, APP_PORT
from uvicorn import run as app_run
import pandas as pd
from pydantic import BaseModel
from datetime import datetime
import os
app = FastAPI()

origins = ["*"]

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]

)


app.get("/", tags = ["authentication"])
async def index():
    return RedirectResponse(url = '/docs')

@app.get("/train")
async def train():
    try:
        training_pipeline = TrainingPipeline()
        if training_pipeline.is_pipeline_running:
            return Response("Training Pipeline is already running")
        
        training_pipeline.run_pipeline()
        return Response("Training successfully completed")
    except Exception as e:
        logging.error(f"Error occurred in the async train function of main.py - {e}")


@app.get("/predict_bulk")
async def predict_bulk():
    try:
        logging.info("Entered the predict_bulk function in main.py")
        prediction_pipeline = PredictionPipeline()
        logging.info("Running prediction pipeline in main.py")
        prediction = prediction_pipeline.start_prediction_pipeline()
       
        return Response(f"Output is: {prediction}")
    except  Exception as e:
        logging.error(f"Error occured in the predict bulk route - {e}")

class Item(BaseModel):
    LIMIT_BAL:int
    SEX:int
    EDUCATION:int
    MARRIAGE:int
    AGE:int
    PAY_0:int
    PAY_2:int
    PAY_3:int
    PAY_4:int
    PAY_5:int
    PAY_6:int
    BILL_AMT1:int
    BILL_AMT2:int
    BILL_AMT3:int
    BILL_AMT4:int
    BILL_AMT5:int
    BILL_AMT6:int
    PAY_AMT1:int
    PAY_AMT2:int
    PAY_AMT3:int
    PAY_AMT4:int
    PAY_AMT5:int
    PAY_AMT6:int


@app.post('/predict_single')
async def predict_single(features:Item):
    try:
        logging.info("Entered the predict_single function of main.py")
        prediction_pipeline = PredictionPipeline()
        logging.info("Running prediction pipeline in main.py")
        dict_form_data = features.model_dump()
        logging.info(dict_form_data)
        logging.info(f"Items = {dict_form_data.items()}")
        df = pd.DataFrame([dict_form_data])
        logging.info(f"Dataframe - {df.head()}")
        timestamp = str(round(datetime.now().timestamp()))
        file_name = "prediction/prediction_" + timestamp + '.csv'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        df.to_csv(file_name, header = True,  index=False)
        prediction_pipeline = PredictionPipeline(file_name)
        logging.info("Running prediction pipeline in main.py")
        prediction = prediction_pipeline.start_prediction_pipeline()
        return Response(f"Output: {prediction.astype(int)}")
    except  Exception as e:
        logging.error(f"Error occurred in the predict_single function of main.py - {e} ")







def main():
    try:
        training_pipeline = TrainingPipeline()
        if training_pipeline.is_pipeline_running:
            return Response("Training Pipeline is already running")
        
        training_pipeline.run_pipeline()
        return Response("Training successfully completed")
    except Exception as e:
        logging.error(f"Error occurred in the async train function of main.py - {e}")







if __name__ == '__main__':
    app_run(app, host = APP_HOST, port = APP_PORT)


# if __name__ == "__main__":
#     # file_name = 'data.xls'
#     # database_name = DATABASE_NAME
#     # collection_name = COLLECTION_NAME
#     # dump_csv_file_to_mongodb_collection(file_path=file_name, database_name=database_name, collection_name=collection_name)
#     training_pipeline = TrainingPipeline()
#     training_pipeline.run_pipeline()
#     prediction_pipeline = PredictionPipeline()
#     prediction_pipeline.start_prediction_pipeline()