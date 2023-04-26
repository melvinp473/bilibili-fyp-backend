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

    test = mongo_db()
    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db,"Data")
    find = {"DATASET_ID": "64355a8d78b04a2b1549c1c5"}
    store = mongo_db_function.get_by_query(collection,find,"DATASET_ID")
    path = mongo_db_function.list_to_csv(store)
    mongo_db_function.csv_to_arff(path)
    mongo_db_function.remove_file(path)






    application.run()