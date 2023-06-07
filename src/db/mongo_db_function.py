import pymongo
import csv
from pymongo.database import Database
from pymongo.collection import Collection
from src.db import mongo_db
from src.db import mongodb_connection
from bson.objectid import ObjectId
import os
import pandas as pd

def get_database(name: str) -> Database:
    """
    Get databases object from mongo db cluster
    :param name: db name which is tenant code
    :return: database object
    """
    return mongodb_connection[name]

def get_collection(db: Database, name: str) -> Collection:
    """
    Get collection object
    :param db: mongo db object
    :param name: collection name
    :return:
    """
    return db[name]

def get_by_id(collection: Collection, id: str) :
    doc_id = ObjectId(id)
    doc = collection.find_one({'_id': doc_id})
    return doc

def get_by_query(collection: Collection, dict: dict, key: str):
    doc = collection.find({key:dict[key]})
    data_list = []
    count = 0
    for i in doc:
        id = i.get('_id')
        new_id = str(id)
        i['_id'] = new_id
        data_list.append(i)
        count = count + 1
    return data_list


def upsert_document(collection: Collection, id, doc_dict: dict):

    doc_id = ObjectId(id)
    collection.replace_one({'_id': doc_id}, doc_dict, upsert=True)
    return doc_id

def create_document(collection: Collection, doc_dict: dict):

    collection.insert_one(document=doc_dict)
    for key, value in doc_dict.items():
        print(key, ' : ', value)
    return doc_dict

def delete_document(collection: Collection, id):
    doc_id = ObjectId(id)
    collection.delete_one({'_id': doc_id})
    return doc_id

def delete_dataset(collection: Collection, dataset_id):
    filter = {'DATASET_ID': dataset_id}
    collection.delete_many(filter)
    return dataset_id

def insert_dataset(collection: Collection, insert_data):
    collection.insert_many(insert_data)
    return

def list_to_csv(list: list):

    fields = list[0].keys()
    for i in list:
        i.pop('_id')
        i.pop('DATASET_ID')
    file_name = 'list.csv'
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, file_name)
    with open('list.csv', mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(list)
    return file_path

def csv_to_arff(file_path):
    csv = pd.read_csv(file_path)
    current_dir = os.getcwd()
    file_name = 'list.arff'
    path = os.path.join(current_dir, file_name)
    with open('list.arff', 'w') as f:
        f.write('@relation MLDATA\n\n')
        for col in csv.columns:
            col_nume = col
            if ' ' in col:
                col_nume = "'" + col + "'"
                print(col_nume)
            f.write('@attribute {} {}\n'.format(col_nume, 'numeric' if csv[col].dtype == 'float64' or csv[col].dtype == 'int64' else 'STRING'))
        f.write('\n@data\n')
        for _, row in csv.iterrows():
            f.write(','.join(str(val) for val in row.values) + '\n')

    print(path)
    return path

def remove_file(file_path):
    try:
        os.remove(file_path)
        print("remove successful", file_path)
    except FileNotFoundError:
        print("file not find:", file_path)
    except Exception as e:
        print("remove fail:", file_path)
        print("error:", e)

