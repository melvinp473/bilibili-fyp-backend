import os

from flask import Blueprint
from pymongo import MongoClient
import pprint
import dotenv

mongo_db_api = Blueprint('facebook_feed_api', __name__, url_prefix='/mongoDB/feed')

# this should be put in application.py
dotenv.load_dotenv()

MONGO_DB_CONNECTION = os.getenv('MONGO_DB_CLUSTER_ENDPOINT')


# connect to cluster
client = MongoClient(MONGO_DB_CONNECTION)

# select database, select collection
db = client['FIT4701']
collection = db['Data']

# print certain documents
all_documents = collection.find({"CODE": {"$lt": 5}})
for document in all_documents:
    pprint.pprint(document)

# print document count
print(collection.count_documents({}))
