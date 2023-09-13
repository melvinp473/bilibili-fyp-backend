import math

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, r_regression, f_regression, mutual_info_classif, chi2, \
    f_classif
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
# from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from src.db import mongo_db_function

from datetime import datetime


def imputation(dataset_id, strategy_type, variables):
    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")
    store = mongo_db_function.get_by_query(collection, dataset_id, "DATASET_ID")

    db_data = []
    keys = list(store[0].keys())
    keys.pop(0)
    keys.remove("DATASET_ID")

    dataset_id_val = store[0].get('DATASET_ID')

    df = pd.DataFrame(data=store)
    df = df.drop('DATASET_ID', axis=1)
    df = df.drop('_id', axis=1)
    df_new = df

    # df = pd.DataFrame(data=db_data)
    # df = df.replace(None, np.nan)

    # arr = df.values
    df_temp = pd.DataFrame()
    for variable in variables:
        df_temp[variable] = df[variable]

    imp_mean = SimpleImputer(missing_values=np.nan, strategy=strategy_type)
    df_temp = pd.DataFrame(data=imp_mean.fit_transform(df_temp), columns=variables)

    # for variable in variables:
    #     df_new[variable] = df_temp[variable]

    for variable in variables:
        df[variable] = df_temp[variable]
    arr_new = df.values
    documents = []
    # arr_new = df_new.values
    for element in arr_new:
        temp_dict = {}
        for i in range(len(keys)):
            if math.isnan(element[i]):
                temp_dict[keys[i]] = None
            else:
                temp_dict[keys[i]] = element[i]
        temp_dict['DATASET_ID'] = dataset_id_val
        documents.append(temp_dict)

    mongo_db_function.delete_dataset(collection, dataset_id_val)
    print(documents[0])
    mongo_db_function.insert_dataset(collection, documents)


def standardization(dataset_id, variables):
    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")
    store = mongo_db_function.get_by_query(collection, dataset_id, "DATASET_ID")
    db_data = []
    keys = list(store[0].keys())
    keys.pop(0)
    keys.remove("DATASET_ID")
    # keys.remove('CODE')
    dataset_id_val = store[0].get('DATASET_ID')
    # for item in store:
    #     item.pop('DATASET_ID')
    #     item.pop('_id')
    #     item.pop('CODE')
    #     values = list(item.values())
    #     db_data.append(values)
    # df = pd.DataFrame(data=db_data)

    df = pd.DataFrame(data=store)
    df = df.drop('DATASET_ID', axis=1)
    df = df.drop('_id', axis=1)
    # arr = df.values
    df_temp = pd.DataFrame()
    for variable in variables:
        df_temp[variable] = df[variable]

    df_temp = pd.DataFrame(data=preprocessing.StandardScaler().fit_transform(df_temp), columns=variables)
    for variable in variables:
        df[variable] = df_temp[variable]
    arr_new = df.values
    documents = []
    # code = 1
    for element in arr_new:
        temp_dict = {}
        for i in range(len(keys)):
            temp_dict[keys[i]] = element[i]
        temp_dict['DATASET_ID'] = dataset_id_val
        # temp_dict['CODE'] = code
        # code = code + 1
        documents.append(temp_dict)
    mongo_db_function.delete_dataset(collection, dataset_id_val)
    mongo_db_function.insert_dataset(collection, documents)


def normalization(dataset_id, variables):
    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")
    store = mongo_db_function.get_by_query(collection, dataset_id, "DATASET_ID")
    db_data = []
    keys = list(store[0].keys())
    keys.pop(0)
    keys.remove("DATASET_ID")
    # keys.remove('CODE')
    dataset_id_val = store[0].get('DATASET_ID')
    # for item in store:
    #     item.pop('DATASET_ID')
    #     item.pop('_id')
    #     item.pop('CODE')
    #     values = list(item.values())
    #     db_data.append(values)
    # df = pd.DataFrame(data=db_data)
    # x = df.values
    #
    # scaler = preprocessing.MinMaxScaler()
    # scaler.fit(x)
    # x_processed = scaler.transform(x)
    df = pd.DataFrame(data=store)
    df = df.drop('DATASET_ID', axis=1)
    df = df.drop('_id', axis=1)
    # arr = df.values
    df_temp = pd.DataFrame()
    for variable in variables:
        df_temp[variable] = df[variable]

    df_temp = pd.DataFrame(data=preprocessing.MinMaxScaler().fit_transform(df_temp), columns=variables)
    for variable in variables:
        df[variable] = df_temp[variable]
    arr_new = df.values

    documents = []
    # code = 1
    for element in arr_new:
        temp_dict = {}
        for i in range(len(keys)):
            temp_dict[keys[i]] = element[i]
        temp_dict['DATASET_ID'] = dataset_id_val
        # temp_dict['CODE'] = code
        # code = code + 1
        documents.append(temp_dict)
    mongo_db_function.delete_dataset(collection, dataset_id_val)
    mongo_db_function.insert_dataset(collection, documents)


def outliers_removal(dataset_id, variables):
    # id = '64a12d201367499af006379f'
    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")
    # input_ = {"DATASET_ID": dataset_id}
    store = mongo_db_function.get_by_query(collection, dataset_id, "DATASET_ID")

    keys = list(store[0].keys())
    keys.pop(0)
    keys.remove("DATASET_ID")
    dataset_id_val = store[0].get('DATASET_ID')

    # Create the dataframe
    df = pd.DataFrame(data=store)
    df = df.drop('DATASET_ID', axis=1)
    df = df.drop('_id', axis=1)

    for outlier in variables:
        # outlier = 'HCC_HOLDER'

        # sns.boxplot(df['HCC_HOLDER'])
        # plt.show()

        # print(np.where(df['HCC_HOLDER']>2.5))

        # z = np.abs(stats.zscore(df[outlier]))
        # print(z)

        Q1 = np.percentile(df[outlier], 25, method='midpoint')
        Q3 = np.percentile(df[outlier], 75, method='midpoint')
        IQR = Q3 - Q1
        # print(IQR)

        # Above Upper bound
        upper = Q3 + 1.5 * IQR
        upper_array = np.array(df[outlier] >= upper)
        print("Upper Bound:", upper)
        print(upper_array.sum())

        # Below Lower bound
        lower = Q1 - 1.5 * IQR
        lower_array = np.array(df[outlier] <= lower)
        print("Lower Bound:", lower)
        print(lower_array.sum())

        # print(df[upper_array])
        df.loc[upper_array, outlier] = None
        df.loc[lower_array, outlier] = None
        # print(df[upper_array][outlier])
        # print(df[lower_array][outlier])

    arr_new = df.values
    documents = []

    for element in arr_new:
        temp_dict = {}
        for i in range(len(keys)):
            if math.isnan(element[i]):
                temp_dict[keys[i]] = None
            else:
                temp_dict[keys[i]] = element[i]
        temp_dict['DATASET_ID'] = dataset_id_val
        documents.append(temp_dict)

    mongo_db_function.delete_dataset(collection, dataset_id_val)
    mongo_db_function.insert_dataset(collection, documents)


def label(dataset_id, variables):
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
    # df = pd.DataFrame(data=db_data)
    df = pd.DataFrame(data=store)
    df = df.drop('DATASET_ID', axis=1)
    df = df.drop('_id', axis=1)

    df_temp = pd.DataFrame()
    for variable in variables:
        df_temp[variable] = df[variable]
    arr = df_temp.values
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

    df_temp = pd.DataFrame(data=arr, columns=variables)
    for variable in variables:
        df[variable] = df_temp[variable]
    arr_new = df.values

    for element in arr_new:
        temp_dict = {}
        for i in range(len(keys)):
            temp_dict[keys[i]] = element[i]
        temp_dict['DATASET_ID'] = dataset_id_val

        documents.append(temp_dict)
    mongo_db_function.delete_dataset(collection, dataset_id_val)
    mongo_db_function.insert_dataset(collection, documents)


def k_selection(dataset_id, k, selection_type, target_attribute):
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
    if selection_type == "mutual_info_regression":
        regr = mutual_info_regression

    elif selection_type == "r_regression":
        regr = r_regression

    elif selection_type == "f_regression":
        regr = f_regression

    elif selection_type == "chi2":
        regr = chi2

    elif selection_type == "mutual_info_classif":
        regr = mutual_info_classif

    elif selection_type == "f_classif":
        regr = f_classif

    selector = SelectKBest(regr, k=k)
    selector.fit_transform(x, y)
    scores = selector.scores_[selector.get_support()]
    selection = x.columns[selector.get_support()]

    for i in range(len(scores)):
        ret_arr.append((selection[i], scores[i]))

    return ret_arr


def wrapper_selection(dataset_id, k, selection_type, target_attribute, estimator_type, estimator_params, model):
    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")
    input_ = {"DATASET_ID": dataset_id}
    store = mongo_db_function.get_by_query(collection, input_, "DATASET_ID")

    df = pd.DataFrame(data=store)
    df = df.drop('DATASET_ID', axis=1)
    df = df.drop('_id', axis=1)

    x = df.drop(target_attribute, axis=1)
    y = df[target_attribute]

    score_type = "r2"

    if estimator_type == "lin_regr":
        estimator = LinearRegression()

    elif estimator_type == "knn":
        if model == "classification":
            score_type = "accuracy"
            estimator = neighbors.KNeighborsClassifier(**estimator_params)
        elif model == "regression":
            estimator = neighbors.KNeighborsRegressor(**estimator_params)

    # if selection_type == "sfs":
    #     sfs = SequentialFeatureSelector(estimator, n_features_to_select=k, scoring=score_type)
    #     sfs.fit_transform(x, y)
    #     attributes = x.columns[sfs.get_support()]
    if selection_type == "sfs":
        sfs = SequentialFeatureSelector(estimator, k_features=k, scoring=score_type)
        sfs.fit_transform(x, y)
        attributes = list(sfs.k_feature_names_)
        attributes.append(sfs.k_score_)

        return attributes

    elif selection_type == "rfe":
        rfe = RFE(estimator, n_features_to_select=k)
        rfe.fit_transform(x, y)
        attributes = x.columns[rfe.get_support()]

        attributes = list(attributes)
        attributes.append(rfe.score(x, y))

        return attributes

def split_dataset(dataset_id, target_attribute):

    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Data")
    store = mongo_db_function.get_by_query(collection, dataset_id, "DATASET_ID")
    split_dict = {}
    for document in store:
        document.pop("_id")
        split_value = document[target_attribute]
        if split_value in split_dict:
            split_dict[split_value].append(document)
        else:
            split_dict[split_value] = [document]
    split_datasets = [split_dict[key] for key in split_dict]

    for doc in split_datasets:
        dataset_collection = mongo_db_function.get_collection(db, "Dataset")
        dataset_var = mongo_db_function.get_by_id(dataset_collection,doc[0]["DATASET_ID"])

        name = doc[0]["State"]
        id = dataset_var["user_id"]

        keys_list = list(doc[0].keys())
        keys_list.remove('DATASET_ID')
        data = {
            "name": name,
            "user_id": id,
            "status": "ACTIVE",
            "create_date": datetime.now(),
            "update_date": datetime.now(),
            "attributes": keys_list
        }

        dataset_doc_insert_result = dataset_collection.insert_one(data)
        dataset_doc_id = str(dataset_doc_insert_result.inserted_id)

        for each in doc:
            each["DATASET_ID"] = dataset_doc_id
        mongo_db_function.insert_dataset(collection, doc)


# label({"DATASET_ID": "6489def06240641623711ca0"})
# print(k_selection("6491a29f8ec5697220711e44", 5, "f_regression", "STROKE"))
