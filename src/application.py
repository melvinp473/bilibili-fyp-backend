import argparse
import pprint

import dotenv

from src.db import mongo_db
from src.db import mongo_db_function

dotenv.load_dotenv()

from src.app import create_app

application = create_app(debug=True)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(
    #     description=__doc__,
    #     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('--host')
    # parser.add_argument('--port')
    #
    # args = parser.parse_args()
    # print('args', args)
    # host = None
    # port = None
    # if args.host:
    #     host = args.host
    # if args.port:
    #     port = args.port

    # test = mongo_db()
    # db = mongo_db_function.get_database('FIT4701')
    # collection = mongo_db_function.get_collection(db,"Data")
    # change = {"key1": "value1", "key2": "value4"}
    # mongo_db_function.upsert_document(collection,"64367660c584bb4958e67921",change)
    # mongo_db_function.create_document(collection,change)
    # mongo_db_function.delete_document(collection,"64440aa0ac2dbfd2ce049252")



    application.run()