from dotenv import load_dotenv
from src.credit_card_defaulters.constants.database import DATABASE_NAME
load_dotenv
from src.credit_card_defaulters.logger import logging
import os
from src.credit_card_defaulters.constants.env_variables import MONGO_DB_URL_KEY
import pymongo
import certifi
ca = certifi.where()

class MongoDBClient:
    client = None 

    def __init__(self, database_name = DATABASE_NAME) -> None:
        """"
            This will be used to make a connection with the mongodb database and then retrieve the data from the mongodb database
        """
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGO_DB_URL_KEY)
                logging.info(f"Retrieved the mongo db url from the environment varianble")
                # if "localhost" in mongo_db_url:
                #     MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                # else:
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            logging.error(f"There is an exception in the MongoDBClient class - {e}")
            raise e