import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, r_regression, f_regression

from src.db import mongo_db_function


def imputation(dataset_id, strategy_type):
    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")
    store = mongo_db_function.get_by_query(collection, dataset_id, "DATASET_ID")

    db_data = []
    keys = list(store[0].keys())
    keys.pop(0)
    keys.remove("DATASET_ID")

    dataset_id_val = store[0].get('DATASET_ID')
    # for item in store:
    #     item.pop('DATASET_ID')
    #     item.pop('_id')
    #     values = list(item.values())
    #     db_data.append(values)

    df = pd.DataFrame(data=store)
    df = df.drop('DATASET_ID', axis=1)
    df = df.drop('_id', axis=1)

    # df = pd.DataFrame(data=db_data)
    df = df.replace("n/a", np.nan)

    arr = df.values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy=strategy_type)
    arr_new = imp_mean.fit_transform(arr)
    documents = []
    for element in arr_new:
        temp_dict = {}
        for i in range(len(keys)):
            temp_dict[keys[i]] = element[i]
        temp_dict['DATASET_ID'] = dataset_id_val
        documents.append(temp_dict)
    mongo_db_function.delete_dataset(collection, dataset_id_val)
    print(documents[0])
    mongo_db_function.insert_dataset(collection, documents)


def standardization(dataset_id):
    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")
    store = mongo_db_function.get_by_query(collection, dataset_id, "DATASET_ID")
    db_data = []
    keys = list(store[0].keys())
    keys.pop(0)
    keys.remove("DATASET_ID")
    keys.remove('CODE')
    dataset_id_val = store[0].get('DATASET_ID')
    for item in store:
        item.pop('DATASET_ID')
        item.pop('_id')
        item.pop('CODE')
        values = list(item.values())
        db_data.append(values)
    df = pd.DataFrame(data=db_data)
    arr = df.values

    scaler = preprocessing.StandardScaler().fit(arr)
    arr_new = scaler.transform(arr)

    documents = []
    code = 1
    for element in arr_new:
        temp_dict = {}
        for i in range(len(keys)):
            temp_dict[keys[i]] = element[i]
        temp_dict['DATASET_ID'] = dataset_id_val
        temp_dict['CODE'] = code
        code = code + 1
        documents.append(temp_dict)
    mongo_db_function.delete_dataset(collection, dataset_id_val)
    mongo_db_function.insert_dataset(collection, documents)

def normalization(dataset_id):
    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")
    store = mongo_db_function.get_by_query(collection, dataset_id, "DATASET_ID")
    db_data = []
    keys = list(store[0].keys())
    keys.pop(0)
    keys.remove("DATASET_ID")
    keys.remove('CODE')
    dataset_id_val = store[0].get('DATASET_ID')
    for item in store:
        item.pop('DATASET_ID')
        item.pop('_id')
        item.pop('CODE')
        values = list(item.values())
        db_data.append(values)
    df = pd.DataFrame(data=db_data)
    x = df.values

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x)
    x_processed = scaler.transform(x)

    documents = []
    code = 1
    for element in x_processed:
        temp_dict = {}
        for i in range(len(keys)):
            temp_dict[keys[i]] = element[i]
        temp_dict['DATASET_ID'] = dataset_id_val
        temp_dict['CODE'] = code
        code = code + 1
        documents.append(temp_dict)
    mongo_db_function.delete_dataset(collection, dataset_id_val)
    mongo_db_function.insert_dataset(collection, documents)


def label(dataset_id):
    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")
    store = mongo_db_function.get_by_query(collection, dataset_id, "DATASET_ID")
    db_data = []
    keys = list(store[0].keys())
    keys.pop(0)
    keys.remove("DATASET_ID")
    dataset_id_val = store[0].get('DATASET_ID')
    for item in store:
        item.pop('DATASET_ID')
        item.pop('_id')
        values = list(item.values())
        db_data.append(values)
    df = pd.DataFrame(data=db_data)
    arr = df.values
    label = arr[0]
    i_store = []
    for i in range(len(label)):
        try:
            float(label[i])
        except ValueError:
            i_store.append(i)
    documents = []
    label_encoder = preprocessing.LabelEncoder()
    for column in i_store:
        arr[:, column] = label_encoder.fit_transform(arr[:, column])
    for element in arr:
        temp_dict = {}
        for i in range(len(keys)):
            temp_dict[keys[i]] = element[i]
        temp_dict['DATASET_ID'] = dataset_id_val

        documents.append(temp_dict)
    mongo_db_function.delete_dataset(collection, dataset_id_val)
    mongo_db_function.insert_dataset(collection, documents)


def k_selection(dataset_id, k, regression_type, target_attribute):
    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")
    input_ = {"DATASET_ID": dataset_id}
    store = mongo_db_function.get_by_query(collection, input_, "DATASET_ID")

    df = pd.DataFrame(data=store)
    df = df.drop('DATASET_ID', axis=1)
    df = df.drop('_id', axis=1)
    # print(df)
    x = df.drop(target_attribute, axis=1)
    y = df[target_attribute]

    regr = ""
    ret_arr = []
    if regression_type == "mutual_info_regression":
        regr = mutual_info_regression

    elif regression_type == "r_regression":
        regr = r_regression

    elif regression_type == "f_regression":
        regr = f_regression

    selector = SelectKBest(regr, k=k)
    selector.fit_transform(x, y)
    scores = selector.scores_[selector.get_support()]
    selection = x.columns[selector.get_support()]

    for i in range(len(scores)):
        ret_arr.append((selection[i], scores[i]))

    return ret_arr

# label({"DATASET_ID": "6489def06240641623711ca0"})
# print(k_selection("6491a29f8ec5697220711e44", 5, "f_regression", "STROKE"))