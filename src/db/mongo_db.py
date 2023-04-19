import os

from flask import Blueprint
from pymongo import MongoClient
import pprint
import dotenv

class mongoDB:

    # this should be put in application.py
    dotenv.load_dotenv()
    def __init__(self):

        self.MONGO_DB_CONNECTION = os.getenv('MONGO_DB_CLUSTER_ENDPOINT')

        # connect to cluster
        self.client = MongoClient(self.MONGO_DB_CONNECTION)

        # select database, select collection
        self.db = self.client['FIT4701']
        self.collection = self.db['Data']

    def print_table(self):
        # print certain documents
        all_documents = self.collection.find({"CODE": {"$lt": 5}})
        for document in all_documents:
            pprint.pprint(document)

        # print document count
        print(self.collection.count_documents({}))
