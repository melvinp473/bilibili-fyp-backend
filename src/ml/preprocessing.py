from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from src.db import mongo_db_function


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
    store = mongo_db_function.get_by_query(collection, dataset_id, "DATASET_ID")

    # print(store[0].values())
    db_data = []
    keys = list(store[0].keys())
    keys.pop(0)
    keys.remove("DATASET_ID")
    # print(keys)

    dataset_id_val = store[0].get('DATASET_ID')
    # print(dataset_id_val)
    for item in store:
        item.pop('DATASET_ID')
        item.pop('_id')
        values = list(item.values())
        db_data.append(values)

    # print(db_data)

    df = pd.DataFrame(data=db_data)
    df = df.replace("n/a", np.nan)

    arr = df.values
    # print(arr)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    arr_new = imp_mean.fit_transform(arr)
    # print(type(arr_new[0][0]))
    # print(arr_new)
    # print(len(keys))
    # print(len((arr_new[0])))
    documents = []
    for element in arr_new:
        temp_dict = {}
        for i in range(len(keys)):
            temp_dict[keys[i]] = element[i]
        temp_dict['DATASET_ID'] = dataset_id_val
        documents.append(temp_dict)
    mongo_db_function.delete_dataset(collection, dataset_id_val)
    mongo_db_function.insert_dataset(collection, documents)
    # print(documents)


missing_values(dataset_id={"DATASET_ID": "6480818a2b02836f0686e027"})
