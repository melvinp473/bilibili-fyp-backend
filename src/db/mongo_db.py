import os

from flask import Blueprint
from pymongo import MongoClient
import pprint
import dotenv
from bson.objectid import ObjectId

class Database:

    # this should be put in application.py
    dotenv.load_dotenv()
    def __init__(self):

        self.MONGO_DB_CONNECTION = os.getenv('MONGO_DB_CLUSTER_ENDPOINT')
        print(self.MONGO_DB_CONNECTION)
        # connect to cluster
        self.client = MongoClient(self.MONGO_DB_CONNECTION)
        #
        # # select database, select collection
        # self.db = self.client['FIT4701']
        # self.data_collection = self.db['Data']
        # self.dataset_collection = self.db['Dataset']
        # self.user_collection = self.db['User']


    def get_client(self):
        return self.client

    def print_table(self):
        # print certain documents
        all_documents = self.data_collection.find({"CODE": {"$lt": 5}})
        for document in all_documents:
            pprint.pprint(document)

        # print document count
        print(self.data_collection.count_documents({}))




