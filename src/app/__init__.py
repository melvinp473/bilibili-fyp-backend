import json

from flask import Flask, Response, request, Blueprint, make_response, jsonify
from flask_cors import CORS
from ..db import mongo_db, mongo_db_function

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
    return application


