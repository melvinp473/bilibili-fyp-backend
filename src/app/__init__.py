import csv
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bson import ObjectId
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
import numpy as np
from ..db import mongo_db_function
from ..ml import regression, preprocessing, classification, PySAL_SA


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
        request_json = request.get_json()
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")
        list = mongo_db_function.get_by_query(collection, request_json, "DATASET_ID")
        r_data = {'data': list}
        response = jsonify(r_data)
        return response

    @application.route('/machine-learning', methods=['POST'])
    def run_machine_learning():

        np.random.seed(1)
        request_json = request.get_json()
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")
        store = mongo_db_function.get_by_query(collection, request_json, "DATASET_ID")

        # split file

        algo = request_json["algo_name"]
        split_variable = request_json["split_variable"]
        independent_variables = request_json["independent_variables"]
        target_variable = request_json["target_variable"]
        return_list = []
        return_dict = ""

        if split_variable is not None:
            split_dict = {}
            for document in store:
                split_value = document[split_variable]
                if split_value in split_dict:
                    split_dict[split_value].append(document)
                else:
                    split_dict[split_value] = [document]

            split_datasets = [split_dict[key] for key in split_dict]


        else:
            split_datasets = [store]
        for split_data in split_datasets:
            df = mongo_db_function.list_to_df(split_data)
            try:
                if algo not in ['voting_regr', 'voting_cls']:
                    algo_params = {key: value for key, value in request_json["algo_params"].items() if
                                   value is not None and value != ''}
                else:
                    algo_params = request_json["algo_params"]


                if algo == "linear_regr":
                    return_dict = regression.linear_regression(df, target_variable, independent_variables)
                elif algo == "decision_trees_regr":
                    return_dict = regression.decision_trees(df, target_variable, independent_variables, algo_params)
                elif algo == "svm_regr":
                    return_dict = regression.support_vector_machines(df, target_variable, independent_variables)
                elif algo == "knn_regr":
                    return_dict = regression.kth_nearest_neighbors(df, target_variable, independent_variables,algo_params)
                elif algo == "random_forest_regr":
                    return_dict = regression.random_forest(df, target_variable, independent_variables, algo_params)
                elif algo == "bagging_regr":
                    return_dict = regression.bagging_regr(df, target_variable, independent_variables, algo_params)
                elif algo == "voting_regr":
                    return_dict = regression.voting_regressor(df, target_variable, independent_variables, algo_params)
                elif algo == "decision_trees_cls":
                    return_dict = classification.decision_trees_classification(df, target_variable, independent_variables,algo_params)
                elif algo == "random_forest_cls":
                    return_dict = classification.random_forest_classification(df, target_variable, independent_variables,algo_params)
                elif algo == "knn_cls":
                    return_dict = classification.k_nearest_neighbor_classification(df, target_variable,independent_variables, algo_params)
                elif algo == "gauss_naive_bayes_cls":
                    return_dict = classification.gaussian_naive_bayes(df, target_variable, independent_variables)
                elif algo == "voting_cls":
                    return_dict = classification.voting_cls(df, target_variable, independent_variables, algo_params)

                if split_variable is not None:
                    return_dict.update({"split_value": split_data[0][split_variable]})

                if request_json["result_logging"]["save_results"]:
                    metric = return_dict
                    log = mongo_db_function.get_collection(db, "Log")
                    run_id = mongo_db_function.get_run_id(log)

                    run_id += 1

                    data = {
                        "user_id": request_json["user_id"],
                        "dataset_id": request_json["DATASET_ID"],
                        "algo_type": request_json["algo_type"],
                        "algo_name": algo,
                        "run_name": request_json["result_logging"]["runName"],
                        "run_id": run_id,
                        "metrics": metric,
                        "create_date": datetime.now(),
                    }

                    mongo_db_function.update_log(log, data)

                return_list.append(return_dict)

            except BaseException as e:
                e = str(e)
                return_dict = {'error': e}
                json_data = jsonify(return_dict)
                return json_data

        return_dict = {'data': return_list}
        json_data = jsonify(return_dict)

        json_data = json_data


        return json_data

    @application.route('/upload-dataset', methods=['POST'])
    def upload_dataset():
        try:
            max_col_length = 15

            dataset = request.files['dataset']
            dataset.save(dataset.filename)

            # get attribute(column) names and shorten them
            with open(dataset.filename, 'r') as csvfile:
                reader = csv.reader(csvfile)

                attributes = next(reader)
                temp_attributes = []
                for attribute in attributes:
                    temp_attributes.append(attribute[0:max_col_length])

                attributes = temp_attributes

            data = {
                "name": dataset.filename.split('.')[0],
                "user_id": request.form['user_id'],
                "status": "ACTIVE",
                "create_date": datetime.now(),
                "update_date": datetime.now(),
                "attributes": attributes
            }

            db = mongo_db_function.get_database('FIT4701')
            collection = mongo_db_function.get_collection(db, "Dataset")
            dataset_doc_insert_result = collection.insert_one(data)
            dataset_doc_id = str(dataset_doc_insert_result.inserted_id)

            # create new Data documents
            with open(dataset.filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                reader.fieldnames = attributes

                # Skip the first row (header row) in the CSV
                next(reader)

                data_rows = []
                for line in reader:
                    row = {}
                    for key, value in line.items():
                        row[key] = convert_to_numeric(value)
                    row["DATASET_ID"] = dataset_doc_id
                    data_rows.append(row)

            db = mongo_db_function.get_database('FIT4701')
            collection = mongo_db_function.get_collection(db, "Data")
            data_doc_insert_result = collection.insert_many(data_rows)

            if data_doc_insert_result.acknowledged:
                response = "successfully uploaded " + dataset.filename
                flag = True
            else:
                response = "Error when inserting data to database"
                flag = False
            response =  jsonify({
                'response': response,
                'flag': flag
            })
        except BaseException as e:
            e = str(e)
            return_dict = {'error': e}
            response = jsonify(return_dict)

        return response

    @application.route('/preprocessing', methods=['POST'])
    def do_preprocessing():
        try:
            request_json = request.get_json()
            dataset_id = request_json['DATASET_ID']
            preprocessing_code = request_json['preprocessing_code']
            # selected_variables = request_json['variables']
            # print(selected_variables)
            print(dataset_id)
            input = {"DATASET_ID": dataset_id}

            flag = True
            body = []
            if preprocessing_code == 'mean imputation':
                selected_variables = request_json['variables']
                preprocessing.imputation(input, "mean", selected_variables)

            elif preprocessing_code == 'median imputation':
                selected_variables = request_json['variables']
                preprocessing.imputation(input, "median", selected_variables)

            elif preprocessing_code == 'label encoding':
                selected_variables = request_json['variables']
                preprocessing.label(input, selected_variables)

            elif preprocessing_code == 'outlier':
                selected_variables = request_json['variables']
                preprocessing.outliers_removal(input, selected_variables)

            elif preprocessing_code == 'select_k_best':
                k = request_json['params']['k_best']
                selection_type = request_json['params']['selection_type']
                target_attribute = request_json['params']['target_attribute']
                body = preprocessing.k_selection(dataset_id, k, selection_type, target_attribute)

            elif preprocessing_code == 'wrapper':
                k = request_json['params']['k_best']
                selection_type = request_json['params']['selection_type']
                target_attribute = request_json['params']['target_attribute']
                estimator_type = request_json['params']['estimator_type']
                estimator_params = request_json['params']['estimator_params']
                model = request_json['params']['model']
                body = preprocessing.wrapper_selection(dataset_id, k, selection_type, target_attribute, estimator_type,
                                                       estimator_params, model)

            elif preprocessing_code == 'standardization':
                try:
                    selected_variables = request_json['variables']
                    preprocessing.standardization(input, selected_variables)
                except ValueError as e:
                    print(e)
                    flag = False
            elif preprocessing_code == 'normalization':
                selected_variables = request_json['variables']
                preprocessing.normalization(input, selected_variables)

            elif preprocessing_code == 'split dataset':
                selected_variables = request_json['variables'][0]
                preprocessing.split_dataset(input,selected_variables)

            response = {'flag': flag,
                        'body': body}
            response = jsonify(response)

        except KeyError as key_err:
            err_type = type(key_err).__name__
            if str(key_err) == "'DATASET_ID'":
                msg = err_type + ". Please select a dataset from the Datasets page."
            elif str(key_err) == "'preprocessing_code'":
                msg = err_type + ". Please select a preprocessing method."
            else:
                msg = err_type + ". Required value " + str(key_err) + " is missing."
            response = jsonify({'error': msg})

        except BaseException as e:
            e = str(e)
            return_dict = {'error': e}
            response = jsonify(return_dict)

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
        dataset_id = request_json['dataset_id']
        user_id = request_json['user_id']

        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Log")
        # r_list = mongo_db_function.get_by_query(collection, request_json, "user_id")
        cursor = collection.find({
            'dataset_id': dataset_id,
            'user_id': user_id
        })
        r_list = list(cursor)
        for item in r_list:
            item["_id"] = str(item["_id"])
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

    @application.route('/dataset/<string:dataset_id>', methods=['DELETE'])
    def delete_dataset(dataset_id):

        # delete dataset
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Dataset")
        delete_dataset_result = collection.delete_one({'_id': ObjectId(dataset_id)})

        # delete data
        print(dataset_id)
        collection = mongo_db_function.get_collection(db, "Data")
        delete_data_result = collection.delete_many({'DATASET_ID': dataset_id})

        return jsonify({
            'flag': f'{delete_dataset_result.acknowledged}',
            'data_rows_removed': f'{delete_data_result.deleted_count}',
        })

    @application.route('/spatial-analysis', methods=['POST'])
    def do_spatial_analysis():
        request_json = request.get_json()
        dataset_id = request_json['dataset_id']
        user_id = request_json['user_ic']
        area_level = request_json['area_level']
        target_variable = request_json['target_variable']
        file_path = '../shp/aus_pha_shape_files/pha_shape_files/2021'
        db = mongo_db_function.get_database('FIT4701')
        collection = mongo_db_function.get_collection(db, "Data")
        data = mongo_db_function.get_by_query(collection, {'DATASET_ID': dataset_id}, 'DATASET_ID')
        df = pd.DataFrame(data)
        result = PySAL_SA.spatial_analysis(file_path,target_variable,df,user_id,area_level)
        json_data = jsonify(result)

        json_data = json_data


        return json_data

    # @application.route('/feature-selection', methods=['POST'])
    # def feature_selection():
    #     request_json = request.get_json()
    #     dataset_id = request_json['DATASET_ID']
    #     k = request_json['k']
    #     regression_type = request_json['k']
    #     target_attribute = request_json['k']
    #
    #     response = preprocessing.k_selection(dataset_id, k, regression_type, target_attribute)
    #     return jsonify({'attributes': response})

    return application


def convert_to_numeric(value):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

