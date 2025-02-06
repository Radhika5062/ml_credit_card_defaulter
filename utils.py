import pandas as pd
from src.credit_card_defaulters.logger import logging
from json import loads, dumps
from src.credit_card_defaulters.config_db import mongo_client

def dump_csv_file_to_mongodb_collection(
        file_path:str,
        database_name:str,
        collection_name:str
) -> None:
    """
        This function first reads the data, then converts the data into json format acceptable via mongo db and then sends the data to Mongo db database
    """
    try:
        # Read the data from excel file
        logging.info(f"Entered the dump_csv_file_to_mongodb_collection method")
        df = pd.read_excel("data.xls")
        logging.info(f"Reading the data completed")
        documents = loads(df.to_json(orient = 'records'))
        logging.info(f"Converting data to json completed")
        mongo_client[database_name][collection_name].insert_many(documents=documents)
        logging.info(f"Loaded the data into the mongo db database")
    except Exception as e:
        logging.error(f"Exception occured {e}")
        raise e
    

