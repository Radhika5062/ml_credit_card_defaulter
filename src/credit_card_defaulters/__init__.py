# Reading the .env file
from src.credit_card_defaulters.logger import logging
from dotenv import load_dotenv
load_dotenv()
logging.info("Reading the environment variables")