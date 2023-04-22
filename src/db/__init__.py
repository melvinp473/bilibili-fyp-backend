from .mongo_db import Database as mongo_db
mongodb = mongo_db()
mongodb_connection = mongodb.get_client()
