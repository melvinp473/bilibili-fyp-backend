import json

from flask import Flask, Response, request, Blueprint, make_response, jsonify
from flask_cors import CORS
from ..db import mongo_db, mongo_db_function
import weka.core.jvm as jvm
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.classifiers import PredictionOutput, KernelClassifier, Kernel, Evaluation
import re

import os

def create_app(debug=False):

    # print(os.getenv('FLASK_ENV'))

    application = Flask(__name__)
    application.debug = debug
    CORS(application, origins='http://localhost:4200', headers=['Content-Type'], methods=['POST'])

    @application.route("/", methods=['GET'])
    def home():
        return "Hello, World!"


    @application.route('/connect', methods=['POST'])
    def receive():

        data = request.get_json()

        print(data)

        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")

        mongo_db_function.create_document(collection, data)

        data = {'message': 'Successful'}
        response = jsonify(data)
        return response

    @application.route('/getDataset', methods=['POST'])
    def get_dataset():
        data = request.get_json()
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Dataset")
        list = mongo_db_function.get_by_query(collection,data,"user_id")
        # new_list = json.dumps(list)
        r_data = {'message': list}
        print(r_data)
        response = jsonify(r_data)
        return response


    @application.route('/machineLearning', methods=['POST'])
    def run_machineLearning():
        data = request.get_json()
        id = data["dataset_id"]
        print(id)
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")
        find = {"DATASET_ID": id}
        store = mongo_db_function.get_by_query(collection, find, "DATASET_ID")
        path = mongo_db_function.list_to_csv(store)
        arff_path = mongo_db_function.csv_to_arff(path)
        jvm.start(system_cp=True, packages=True, max_heap_size="512m")
        loader = Loader(classname="weka.core.converters.ArffLoader")
        ml_data = loader.load_file(arff_path)
        ml_data.class_is_last()
        cls = KernelClassifier(classname="weka.classifiers.functions.SMOreg", options=["-N", "0"])
        kernel = Kernel(classname="weka.classifiers.functions.supportVector.RBFKernel", options=["-G", "0.1"])
        cls.kernel = kernel
        pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
        evl = Evaluation(ml_data)
        evl.crossvalidate_model(cls, ml_data, 10, Random(1), pout)
        result = re.split(r"\s{2,}|\n", evl.summary())
        r_result = result[1:-1]
        r_data = {'message': r_result}
        print(pout.buffer_content())
        r_pout = re.split(r"\s{2,}|\n",pout.buffer_content())
        print(r_pout)
        response = jsonify(r_data)
        jvm.stop()
        mongo_db_function.remove_file(path)

        return response



    return application


