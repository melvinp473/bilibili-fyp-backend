import numpy as np
import pandas as pd
import csv

from src.db import mongo_db_function

db = mongo_db_function.get_database('FIT4701')
collection = mongo_db_function.get_collection(db, "Data")
data_list = mongo_db_function.get_by_query(collection, {"DATASET_ID": '6566a3c64472e1daa25d592b'}, "DATASET_ID")

fields = data_list[0].keys()
for i in data_list:
    i.pop('_id')
    i.pop('DATASET_ID')
file_name = 'Australia processed.csv'

with open(file_name, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(data_list)
