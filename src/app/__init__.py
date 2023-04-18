from flask import Flask, Response, request
from flask_cors import CORS
import os

def create_app(debug=False):

    # print(os.getenv('FLASK_ENV'))

    application = Flask(__name__)
    application.debug = debug

    @application.route("/", methods=['GET'])
    def home():
        return "Hello, World!"

    return application

