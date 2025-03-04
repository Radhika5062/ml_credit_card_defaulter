import os
from src.credit_card_defaulters.logger import logging

class s3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        command = f"aws s3 sync {folder} {aws_bucket_url}"
        os.system(command)
    
    def sync_folder_from_s3(self, folder, aws_bucket_url):
        command = f'aws s3 sycn {aws_bucket_url} {folder}'
        os.system(command)