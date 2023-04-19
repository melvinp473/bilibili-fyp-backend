from flask import Flask, Response, request, Blueprint
from flask_cors import CORS
from ..db import mongo_db

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

        db = mongo_db.mongoDB()

        db.print_table()

        return Response(status=200)
    return application


