import argparse
import pprint

import dotenv

from src.db import mongo_db
from src.db import mongo_db_function
import weka.core.jvm as jvm
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.classifiers import PredictionOutput, KernelClassifier, Kernel, Evaluation

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
    # find = {"DATASET_ID": "6435538178b04a2b1549b45e"}
    # store = mongo_db_function.get_by_query(collection,find,"DATASET_ID")
    # path = mongo_db_function.list_to_csv(store)
    # arff_path = mongo_db_function.csv_to_arff(path)
    #
    #
    # jvm.start(system_cp=True, packages=True, max_heap_size="512m")
    # loader = Loader(classname="weka.core.converters.ArffLoader")
    # ml_data = loader.load_file(arff_path)
    # ml_data.class_is_last()
    # cls = KernelClassifier(classname="weka.classifiers.functions.SMOreg", options=["-N", "0"])
    # kernel = Kernel(classname="weka.classifiers.functions.supportVector.RBFKernel", options=["-G", "0.1"])
    # cls.kernel = kernel
    # pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
    # evl = Evaluation(ml_data)
    # evl.crossvalidate_model(cls, ml_data, 10, Random(1), pout)

    # print(evl.summary())
    # print(pout.buffer_content())
    #
    # jvm.stop()
    # mongo_db_function.remove_file(path)






    application.run()