from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from src.db import mongo_db_function

db = mongo_db_function.get_database('FIT4701')
collection = mongo_db_function.get_collection(db, "Data")
mongo_db_function.delete_dataset(collection, "000")

def missing_values(dataset_id):
    # Preprocessing with string "n/a"
    # X = [["n/a", 3, 5], [2, "n/a", 9], [8, 7, "n/a"]]
    # imp_mean = SimpleImputer(missing_values="n/a", strategy='mean')
    # X_new = imp_mean.fit_transform(X)
    # print(X_new)

    # Preprocessing with numeric data type
    # X = [[0, 3, 5], [2, 0, 9], [8, 7, 0]]
    #
    # imp_mean = SimpleImputer(missing_values="NaN", strategy='mean')
    # X_new = imp_mean.fit_transform(X)
    # print("Expected Output:")
    # print([5, 3, 5], [2, 5, 9], [8, 7, 7])
    # print("Actual Output:")
    # print(X_new)

    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")

    # store = mongo_db_function.get_by_query(collection, dataset_id, "DATASET_ID")



    # # print(store[0].values())
    # db_data = []
    # keys = []
    # for item in store:
    #     item.pop('DATASET_ID')
    #     item.pop('_id')
    #     values = list(item.values())
    #     db_data.append(values)
    #
    # # print(db_data)
    #
    # df = pd.DataFrame(data=db_data)
    # df = df.replace("n/a", np.nan)
    #
    # arr = df.values
    # # print(arr)
    # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    # arr_new = imp_mean.fit_transform(arr)
    # # print(type(arr_new[0][0]))
    # # print(arr_new)


# missing_values(dataset_id={"DATASET_ID": "6480818a2b02836f0686e027"})
