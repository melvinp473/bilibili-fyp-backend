import numpy as np
import pandas as pd
import csv

from src.db import mongo_db_function

db = mongo_db_function.get_database('FIT4701')
collection = mongo_db_function.get_collection(db, "Data")
store = mongo_db_function.get_by_query(collection, {"DATASET_ID": '65263581db6e3c189c0d6522'}, "DATASET_ID")
path = mongo_db_function.list_to_csv(store)

df = pd.read_csv(path)

df['stroke reported'] = pd.to_numeric(df['stroke reported'], errors='coerce')

# Create a new array with the converted values
df['STROKE_RISK_LVL'] = np.where(df['stroke reported'] < 0.5, 0, 1)
df.drop(columns=['stroke reported'], inplace=True)

my_list = df.values.tolist()
column_names = df.columns.tolist()

# Include column names in the list
my_list_with_headers = [column_names] + my_list

print(my_list_with_headers)

mongo_db_function.remove_file(path)

output_filename = 'Australia (CLS).csv'

# Write the data to the CSV file
with open(output_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(my_list_with_headers)