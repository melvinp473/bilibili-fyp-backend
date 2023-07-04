import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import KBinsDiscretizer

from src.db import mongo_db_function

db = mongo_db_function.get_database('FIT4701')
collection = mongo_db_function.get_collection(db, "Data")
store = mongo_db_function.get_by_query(collection, {"DATASET_ID": '64a13340a42d65e5e5b31d26'}, "DATASET_ID")
path = mongo_db_function.list_to_csv(store)

df = pd.read_csv(path)

df['STROKE'] = pd.to_numeric(df['STROKE'], errors='coerce')

# Create a new array with the converted values
df['risk_levels'] = np.where(df['STROKE'] < 0.5, 'low risk', 'high risk')

my_list = df.values.tolist()
column_names = df.columns.tolist()

# Include column names in the list
my_list_with_headers = [column_names] + my_list

print(my_list_with_headers)

mongo_db_function.remove_file(path)

filename = 'Default MLDATA CLS (normalized).csv'

# Write the data to the CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(my_list_with_headers)