import json
from io import BytesIO

from bson import ObjectId
from flask import Flask, Response, request, Blueprint, make_response, jsonify
from flask_cors import CORS
from ..db import mongo_db, mongo_db_function
from ..ml import machine_learning, preprocessing

import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

import os


def create_app(debug=False):
    # print(os.getenv('FLASK_ENV'))

    application = Flask(__name__)
    application.debug = debug
    CORS(application, origins='http://localhost:4200', headers=['Content-Type'], methods=['POST', 'DELETE'])

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
        list = mongo_db_function.get_by_query(collection, request_json, "user_id")
        # new_list = json.dumps(list)
        r_data = {'data': list}
        print(r_data)
        response = jsonify(r_data)
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

    @application.route('/machine-learning', methods=['POST'])
    def run_machine_learning():
        request_json = request.get_json()
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")
        store = mongo_db_function.get_by_query(collection, request_json, "DATASET_ID")
        path = mongo_db_function.list_to_csv(store)

        algo = request_json["algo_type"]
        independent_variables = request_json["independent_variables"]
        target_variable = request_json["target_variable"]
        additional_params = request_json["additional_params"]

        return_dict = ""

        if algo == "knn":
            return_dict = machine_learning.kth_nearest_neighbors(path, target_variable, independent_variables, additional_params)
        elif algo == "decision trees":
            return_dict = machine_learning.decision_trees(path, target_variable, independent_variables, additional_params)
        elif algo == "svm":
            return_dict = machine_learning.support_vector_machines(path, target_variable, independent_variables)
        elif algo == "linear regression":
            return_dict = machine_learning.linear_regression(path, target_variable, independent_variables)
        elif algo == "ensemble":
            return_dict = machine_learning.voting_regressor(path, target_variable, independent_variables)

        metric = return_dict
        mongo_db_function.remove_file(path)

        log = mongo_db_function.get_collection(db, "Log")
        run_id = mongo_db_function.get_run_id(log)

        run_id += 1

        data = {
            "user_id": request_json["user_id"],
            "algo_type": algo,
            "run_name": request_json["result_logging"]["runName"],
            "run_id": run_id,
            "metrics": metric,
            "create_date": datetime.now(),
        }

        mongo_db_function.update_log(log,data)

        return_dict = {'data': return_dict}
        json_data = jsonify(return_dict)

        return json_data

    @application.route('/upload-dataset', methods=['POST'])
    def upload_dataset():
        user_id = request.form['user_id']
        f = request.files['dataset']
        f.save(f.filename)
        with open(f.filename, 'r') as csvfile:
            reader = csv.reader(csvfile)

            columns = next(reader)
            attr_col = columns
            print(columns)

            for column in columns:
                print(column)

        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Dataset")

        data = {
            "name": f.filename.split('.')[0],
            "user_id": user_id,
            "status": "ACTIVE",
            "create_date": datetime.now(),
            "update_date": datetime.now(),
            "attributes": attr_col
        }

        result = collection.insert_one(data)
        print("Inserted ID:", result.inserted_id)

        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")
        csvfile = open(f.filename, 'r')
        reader = csv.DictReader(csvfile)

        data_rows = []
        for each in reader:
            row = {}
            for field in columns:
                row[field] = each[field]
            id = str(result.inserted_id)
            row["DATASET_ID"] = id
            print(row)
            data_rows.append(row)

        insert_result = collection.insert_many(data_rows)
        if insert_result.acknowledged:
            response = "successfully uploaded " + f.filename
        else:
            response = "Error when inserting data to database"
        return response

    @application.route('/preprocessing', methods=['POST'])
    def do_preprocessing():
        request_json = request.get_json()
        dataset_id = request_json['DATASET_ID']
        preprocessing_code = request_json['preprocessing_code']
        print(dataset_id)
        input = {"DATASET_ID": dataset_id}

        response = "preprocessing done"
        if preprocessing_code == 'mean imputation':
            preprocessing.imputation(input, "mean")
        elif preprocessing_code == 'median imputation':
            preprocessing.imputation(input, "median")
        elif preprocessing_code == 'label encoding':
            preprocessing.label(input)
        elif preprocessing_code == 'standardization':
            try:
                preprocessing.standardization(input)
            except ValueError as e:
                print(e)
                response = e

        return response

    @application.route('/analysis', methods=['POST'])
    def run_analysis():
        request_json = request.get_json()
        dataset_id = request_json['DATASET_ID']

        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")
        data = mongo_db_function.get_by_query(collection, {'DATASET_ID': dataset_id}, 'DATASET_ID')
        # file_path = mongo_db_function.list_to_csv(data)

        # Create DataFrame
        df = pd.DataFrame(data)
        attributes = db["Dataset"].find_one({"_id": ObjectId(dataset_id)})["attributes"]
        x = df[attributes]

        # Create your plot using Matplotlib or Seaborn
        fig, ax = plt.subplots(figsize=(11, 11))  # Set the desired figure size

        # Plot the correlation matrix as a heatmap
        sns.heatmap(x.corr(), annot=True, fmt=".2f", cmap='coolwarm', )

        # Set the title
        plt.title('Correlation Matrix')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()

        # mongo_db_function.remove_file(file_path)

        # Save the plot image to a BytesIO buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # # Show the plot
        # plt.show()

        # Set the appropriate content type
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'image/png'

        return response

    @application.route('/get-results', methods=['POST'])
    def get_results():
        request_json = request.get_json()
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Log")
        r_list = mongo_db_function.get_by_query(collection, request_json, "user_id")
        r_data = {'data': r_list}
        print(r_data)
        response = jsonify(r_data)
        return response

    @application.route('/results', methods=['DELETE'])
    def delete_results():
        request_json = request.get_json()
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Log")
        ids_to_delete = [ObjectId(doc_id) for doc_id in request_json["doc_ids"]]
        result = collection.delete_many({'_id': {'$in': ids_to_delete}})

        return jsonify({'message': f'{result.deleted_count} documents deleted'})

    return application
