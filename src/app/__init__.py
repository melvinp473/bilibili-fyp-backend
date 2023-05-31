import json

from flask import Flask, Response, request, Blueprint, make_response, jsonify
from flask_cors import CORS
from ..db import mongo_db, mongo_db_function
from ..ml import machine_learning
import csv
import pandas as pd
import sklearn.linear_model as linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

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

        request_json = request.get_json()

        print(request_json)

        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")

        mongo_db_function.create_document(collection, request_json)

        data = {'message': 'Successful'}
        response = jsonify(data)
        return response

    @application.route('/get-dataset', methods=['POST'])
    def get_dataset():
        request_json = request.get_json()
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Dataset")
        list = mongo_db_function.get_by_query(collection,request_json,"user_id")
        # new_list = json.dumps(list)
        r_data = {'data': list}
        print(r_data)
        response = jsonify(r_data)
        return response


    @application.route('/machine-learning', methods=['POST'])
    def run_machineLearning():
        request_json = request.get_json()
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")
        store = mongo_db_function.get_by_query(collection, request_json, "DATASET_ID")
        path = mongo_db_function.list_to_csv(store)
        selected_attributes = request_json["selected_attributes"]
        algo = request_json["algo_type"]



        # machine_learning.support_vector_machines(path,selected_attributes)
        machine_learning.kth_nearest_neighbors(path,selected_attributes)

        json_data = machine_learning.linear_regression(path,selected_attributes)
        response = json_data

        mongo_db_function.remove_file(path)

        return response

    @application.route('/get-data', methods=['POST'])
    def get_data():
        # by jiahao: not used for now, just thinking if we should show the data or not
        request_json = request.get_json()
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")
        list = mongo_db_function.get_by_query(collection, request_json, "DATASET_ID")
        r_data = {'data': list}
        print(r_data)
        response = jsonify(r_data)
        return response

    @application.route('/upload-dataset', methods=['POST'])
    def upload_dataset():
        f = request.files['dataset']
        f.save(f.filename)
        with open(f.filename, 'r') as csvfile:
            reader = csv.reader(csvfile)

            columns = next(reader)
            columns.pop()
            print(columns)
            for column in columns:
                print(column)

        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Dataset")

        data = {
            "name" : "demo",
            "user_id" : "test",
            "status" : "ACTIVE",
            "create_date" : "31/5/23",
            "update_date" : "31/5/23",
            "attribute" : columns
        }

        result = collection.insert_one(data)
        print("Inserted ID:", result.inserted_id)

        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")
        csvfile = open(f.filename, 'r')
        reader = csv.DictReader(csvfile)

        for each in reader:
            row = {}
            for field in columns:
                row[field] = each[field]
            id = str(result.inserted_id)
            row["DATASET_ID"] = id

            print(row)
            collection.insert_one(row)
        response = "received " + f.filename
        return response


    return application


