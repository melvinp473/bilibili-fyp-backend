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
        print(path)
        df = pd.read_csv(path)
        print(df.head())
        regr = linear_model.LinearRegression()
        x = df[['SMOKING', 'DRINKING', 'LACK_EXERCISE', 'AGE65_OVER', 'AGE25_44', 'EARLY_SCHOOL_LEAVERS',
                'HCC_HOLDER', 'RAC_PLACE', 'TOTAL_CLIENTS', 'DIABETES', 'MENTAL_DISEASE', 'HYPERTENSION']]
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
        return_dict.update({"scikit metrics mean absolute error":mean_absolute_error(test_y_, test_y)})
        return_dict.update({"scikit metrics mean squared error":mean_squared_error(test_y_, test_y)})
        return_dict.update({"Residual sum of squares (MSE)": np.mean((test_y_ - test_y) ** 2)})
        return_dict.update({"R2-score": r2_score(test_y, test_y_)})
        json_data = json.dumps(return_dict)
        response = json_data

        mongo_db_function.remove_file(path)

        return response



    return application


