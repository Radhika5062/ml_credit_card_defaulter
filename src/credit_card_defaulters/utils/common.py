import yaml
from src.credit_card_defaulters.logger import logging
import os
import dill
import numpy as np
import pickle
import datetime

def read_yaml_file(file_path):
    """
        This function is used to read the yaml file
    """
    try:
        logging.info(f"Read {file_path} yaml file")
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error happened in read_yaml_file function - {e}")
        
def write_yaml_file(data, file_path):
    """
        This function is used to write the yaml file
    """
    try:
        logging.info(f"Write data in {file_path} yaml file")
        logging.info(f"Data content = {data}")
        logging.info(f"Yaml content = {yaml.dump(data)}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            print(f"Inside write - {yaml.dump(data=data)}")
            yaml.dump(data, f)
    except Exception as e:
        logging.error(f"Error happened in the write_yaml_file - {e}")
        raise e
    
def save_object(file_path:str, obj:object) ->None:
    try:
        logging.info(f"Entered the save_object method")
        logging.info(f"Filename is - {file_path}")
        logging.info(f"Dirname = {os.path.dirname(file_path)}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logging.info(f"Directory created for save_object")
        with open(file_path,"wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Object file created")
    except Exception as e:
        logging.error(f"Error occurred in the save_object method - {e}")
        raise e

def load_object(file_path:str) -> object:
    try:
        logging.info("Entered the load_object method")
        if not os.path.exists(file_path):
            logging.info(f"file path {file_path} does not exists to load it using the load_object method")
        with open(file_path,"rb") as f:
            return dill.load(f)
    except Exception as e:
        logging.error(f"Error occurred in the load_object method - {e}")
        raise e

def save_numpy_array_data(file_path:str, arr:np.array) ->None:
    try:
        logging.info(f"Entered the save_numpy_array_data")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            np.save(f, arr)
    except Exception as e:
        logging.error(f"Error occurred in the save_numoy_array_data method - {e}")
        raise e

def load_numpy_array_data(file_path:str) -> np.array:
    try:
        logging.info("Entered the load_numpy_array_data")
        with open(file_path, "rb") as f:
            return np.load(f)
    except Exception as e:
        logging.error(f"Error occured in the load_numpy_data method - {e}")

def parse_timestamp(folder_name)->str:
    try:
        logging.info("Entered the parse_timestamp method")
        parts = folder_name.split("_")
        data = [int(i) for i in parts if i != 'M']
        month, day, year, hour, minute = data
        return datetime.datetime(year, month, day, hour, minute)
    except Exception as e:
        return None
        logging.error(f"Error occured in the parse_timestamp method - {e}")

def get_latest_folder(file_path:str) -> str:
    try:
        logging.info("Entered the get_latest_folder method")
        logging.info(f"File path = {file_path}")
        dir_name = os.path.dirname(file_path).split('\\')[0]
        logging.info(f"dirname = {dir_name}")
        folders = [f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, f))]
        logging.info(f"folders = {folders}")
        folder_dates = {f:parse_timestamp(f) for f in folders if parse_timestamp(f)}
        logging.info(f"folders = {folder_dates}")
        latest_folder = max(folder_dates, key = folder_dates.get, default = None)
        logging.info(f"latest folder is {latest_folder}")
        return latest_folder, dir_name
    except Exception as e:
        logging.info(f"Error occured in the get_latest_folder method - {e}")

