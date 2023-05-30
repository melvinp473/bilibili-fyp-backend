import json

from flask import Flask, Response, request, Blueprint, make_response, jsonify
from flask_cors import CORS
from ..db import mongo_db, mongo_db_function
import weka.core.jvm as jvm
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
        print(id)
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")
        store = mongo_db_function.get_by_query(collection, request_json, "DATASET_ID")
        path = mongo_db_function.list_to_csv(store)
        print(path)
        df = pd.read_csv(path)
        print(df.head())
        regr = linear_model.LinearRegression()
        # x = df[['SMOKING', 'DRINKING', 'LACK_EXERCISE', 'AGE65_OVER', 'AGE25_44', 'EARLY_SCHOOL_LEAVERS',
        #         'HCC_HOLDER', 'RAC_PLACE', 'TOTAL_CLIENTS', 'DIABETES', 'MENTAL_DISEASE', 'HYPERTENSION']]
        x = df [request_json["selected_attributes"]]
        y = df[["STROKE"]]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
        regr.fit(train_x, train_y)
        test_y_ = regr.predict(test_x)

        print('Coefficients: ', regr.coef_)
        print('Intercept: ', regr.intercept_)

        print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
        print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
        print("Residual sum of squares (MSE): %.4f" % np.mean((test_y_ - test_y) ** 2))
        print("R2-score: %.4f" % r2_score(test_y, test_y_))

        return_dict = {"Coefficients": regr.coef_.tolist()[0], "Intercept": regr.intercept_.tolist()[0]}
        return_dict.update({"mae":mean_absolute_error(test_y_, test_y)})
        return_dict.update({"mse":mean_squared_error(test_y_, test_y)})
        return_dict.update({"r2_score": r2_score(test_y, test_y_)})
        return_dict = {'data': return_dict}
        json_data = jsonify(return_dict)
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
        response = "received " + f.filename
        return response


    return application


