import pymongo
from pymongo.database import Database
from pymongo.collection import Collection
from src.db import mongo_db
from src.db import mongodb_connection
from bson.objectid import ObjectId

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


def get_by_id(collection: Collection, id: str) -> dict:
    doc_id = ObjectId(id)
    doc = collection.find_one({'_id': doc_id})
    return doc


def upsert_document(collection: Collection, id, doc_dict: dict):

    doc_id = ObjectId(id)

    collection.replace_one({'_id': doc_id}, doc_dict, upsert=True)
    return doc_id

def create_document(collection: Collection, doc_dict: dict) -> dict:

    collection.insert_one(document=doc_dict)
    for key, value in doc_dict.items():
        print(key, ' : ', value)
    return doc_dict

def delete_document(collection: Collection, id):
    doc_id = ObjectId(id)
    collection.delete_one({'_id': doc_id})
    return doc_id
