from dataclasses import dataclass
import os
import pymongo

@dataclass
class EnvironmentVariable:
    # This is the constant that we have defined in the .env file
    mongo_db_url = os.getenv("MONGO_DB_URL")

env_var = EnvironmentVariable()

# Now we are just making the connection to the mongo db cluster so that we can use it later. 
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)

